from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from model_registry import build_model, resolve_model
from dataloader import PotholeProposalDataset, DEFAULT_DATA_ROOTS


def apply_nms(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []

    boxes = boxes.float()
    scores = scores.float()

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = scores.sort(descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
        else:
            i = order[0].item()
        keep.append(i)
        
        if order.numel() == 1:
            break
            
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = (iou <= iou_threshold).nonzero(as_tuple=False).squeeze()
        if inds.numel() == 0:
            break
        order = order[inds + 1]
    return keep


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


dataset = PotholeProposalDataset(
    data_root=Path(DEFAULT_DATA_ROOTS.get("pothole", "./data")),
    proposal_type="edge_box",  # or "selective_search" if needed
    proposal_json="Proposal Sample/edge_box_proposals_all.json",
    image_size=(256, 256),
    iou_threshold=0.5,
    positive_ratio=0.25
)

# Check label distribution in the dataset
print(f"Total dataset: {len(dataset)} samples")
label_counts = {0: 0, 1: 0}
for i in range(len(dataset)):
    _, label = dataset[i]
    label_counts[int(label)] += 1
print(f"Label 0 (negative): {label_counts[0]} samples ({100*label_counts[0]/len(dataset):.1f}%)")
print(f"Label 1 (positive): {label_counts[1]} samples ({100*label_counts[1]/len(dataset):.1f}%)")
print()

test_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
print(f"Evaluating on full dataset: {len(dataset)} samples")

checkpoint_path = "lolologs/detcnn_eb_no_vgg_iou05/version_0/checkpoints/epoch=7-step=936.ckpt"

spec = resolve_model("detection_cnn")
model = spec.model_class.load_from_checkpoint(checkpoint_path, **spec.default_params)
model.eval()
model.to(device)
print(f"Model loaded successfully from checkpoint: {checkpoint_path}")

all_results = []
all_scores = []
all_gt_labels = []

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)

        outputs = model(images)
        # Support both dict-based outputs and the classification-only tensor output
        if isinstance(outputs, dict):
            logits = outputs.get("logits")
        else:
            logits = outputs

        if logits is None:
            raise RuntimeError("Model did not return logits. Expected tensor or dict with key 'logits'.")

        probs = torch.softmax(logits, dim=-1)
        scores = probs[..., 1]  # class 1 = pothole
        
        # Collect for statistics
        if scores.dim() == 1:
            all_scores.extend(scores.cpu().tolist())
        elif scores.dim() == 2:
            all_scores.extend(scores.mean(dim=1).cpu().tolist())
        else:
            all_scores.extend(scores.flatten().cpu().tolist())
        
        all_gt_labels.extend(labels.cpu().tolist())

# Group by image and apply NMS per image
print("\nApplying NMS per image...")
for img_idx, annotation in enumerate(dataset.annotations):
    filename = annotation['filename']
    
    # Get all proposals for this image
    image_proposals = []
    image_scores = []
    image_labels = []
    
    for sample_idx, sample in enumerate(dataset.samples):
        if sample['image_idx'] == img_idx:
            proposal = sample['proposal']
            score = all_scores[sample_idx]
            gt_label = all_gt_labels[sample_idx]
            
            image_proposals.append([
                proposal['x_min'], 
                proposal['y_min'], 
                proposal['x_max'], 
                proposal['y_max']
            ])
            image_scores.append(score)
            image_labels.append(gt_label)
    
    if not image_proposals:
        continue
    
    # Convert to tensors
    boxes_tensor = torch.tensor(image_proposals, dtype=torch.float32)
    scores_tensor = torch.tensor(image_scores, dtype=torch.float32)
    
    # Apply NMS
    keep_idx = apply_nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
    
    # Store filtered results
    for idx in keep_idx:
        pred_label = 1 if image_scores[idx] > 0.5 else 0
        all_results.append({
            "image": filename,
            "box": image_proposals[idx],
            "score": float(image_scores[idx]),
            "label": int(pred_label),
            "gt_label": int(image_labels[idx])
        })

import numpy as np

# Calculate accuracy
correct = sum(1 for r in all_results if r['label'] == r['gt_label'])
accuracy = correct / len(all_results) if all_results else 0

print(f"\n" + "="*60)
print("RESULTS AFTER NMS")
print("="*60)
print(f"Total detections: {len(all_results)}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.2%}")

# Count predictions
pred_positives = sum(1 for r in all_results if r['label'] == 1)
pred_negatives = sum(1 for r in all_results if r['label'] == 0)
print(f"\nPredicted positive (pothole): {pred_positives}")
print(f"Predicted negative (not pothole): {pred_negatives}")

# Show some examples
print(f"\n" + "="*60)
if len(all_results) <= 20:
    for i, det in enumerate(all_results):
        print(f"{i}: {det['image']}, Box={det['box']}, Score={det['score']:.3f}, Pred={det['label']}, GT={det['gt_label']}")
else:
    print("(Showing first 10 detections)")
    for i in range(10):
        det = all_results[i]
        print(f"{i}: {det['image']}, Box={det['box']}, Score={det['score']:.3f}, Pred={det['label']}, GT={det['gt_label']}")
