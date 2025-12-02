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
        if order.dim() == 0:
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
    proposal_json="proposals/edge_box_proposals_all.json",
    image_size=(256, 256),
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
all_proposals = []
all_image_indices = []

# Collect all proposal boxes and image indices from the dataset
for idx in range(len(dataset)):
    sample = dataset.samples[idx]
    all_proposals.append(sample['proposal'])
    all_image_indices.append(sample['image_idx'])

sample_idx = 0
with torch.no_grad():
    for batch in test_loader:
        images, labels = batch
        images = images.to(device)

        outputs = model(images)
        # Handle both dict and tensor outputs
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs

        probs = torch.softmax(logits, dim=-1)
        scores = probs[..., 1]  # class 1 = pothole
        
        # Process each sample in the batch
        for b in range(images.size(0)):
            # Handle different output shapes
            if scores.dim() == 1:
                # Single score per sample
                score = scores[b].item()
            elif scores.dim() == 2:
                # Multiple scores per sample (take first or mean)
                score = scores[b].mean().item()
            else:
                score = scores[b].item()
            
            pred_label = 1 if score > 0.5 else 0
            gt_label = labels[b].item()
            proposal = all_proposals[sample_idx]
            
            # Collect for statistics
            all_scores.append(score)
            all_gt_labels.append(gt_label)
            
            all_results.append({
                "box": [proposal['x_min'], proposal['y_min'], proposal['x_max'], proposal['y_max']],
                "score": float(score),
                "label": int(pred_label),
                "gt_label": int(gt_label)
            })
            
            sample_idx += 1

import numpy as np
from torchmetrics.detection import MeanAveragePrecision

print(f"\n" + "="*60)
print("MODEL PREDICTION STATISTICS")
print("="*60)
print(f"Score range: [{min(all_scores):.4f}, {max(all_scores):.4f}]")
print(f"Score mean: {np.mean(all_scores):.4f}")
print(f"Score std: {np.std(all_scores):.4f}")
print(f"Score median: {np.median(all_scores):.4f}")

# Scores for positive vs negative samples
pos_scores = [s for s, l in zip(all_scores, all_gt_labels) if l == 1]
neg_scores = [s for s, l in zip(all_scores, all_gt_labels) if l == 0]

if pos_scores:
    print(f"\nPositive samples (ground truth = 1):")
    print(f"  Count: {len(pos_scores)}")
    print(f"  Score mean: {np.mean(pos_scores):.4f}")
    print(f"  Score range: [{min(pos_scores):.4f}, {max(pos_scores):.4f}]")

if neg_scores:
    print(f"\nNegative samples (ground truth = 0):")
    print(f"  Count: {len(neg_scores)}")
    print(f"  Score mean: {np.mean(neg_scores):.4f}")
    print(f"  Score range: [{min(neg_scores):.4f}, {max(neg_scores):.4f}]")

# Analyze bounding box proposals
print(f"\n" + "="*60)
print("BOUNDING BOX STATISTICS")
print("="*60)
all_boxes_list = [det['box'] for det in all_results]
if all_boxes_list:
    all_boxes_array = np.array(all_boxes_list)
    print(f"Box coordinate means: {all_boxes_array.mean(axis=0)}")
    print(f"Box coordinate stds:  {all_boxes_array.std(axis=0)}")
    print(f"Box coordinate ranges:")
    print(f"  x_min: [{all_boxes_array[:, 0].min():.4f}, {all_boxes_array[:, 0].max():.4f}]")
    print(f"  y_min: [{all_boxes_array[:, 1].min():.4f}, {all_boxes_array[:, 1].max():.4f}]")
    print(f"  x_max: [{all_boxes_array[:, 2].min():.4f}, {all_boxes_array[:, 2].max():.4f}]")
    print(f"  y_max: [{all_boxes_array[:, 3].min():.4f}, {all_boxes_array[:, 3].max():.4f}]")
    
    # Calculate box sizes
    widths = all_boxes_array[:, 2] - all_boxes_array[:, 0]
    heights = all_boxes_array[:, 3] - all_boxes_array[:, 1]
    print(f"\nBox dimensions:")
    print(f"  Width:  mean={widths.mean():.4f}, std={widths.std():.4f}")
    print(f"  Height: mean={heights.mean():.4f}, std={heights.std():.4f}")

# Calculate accuracy
correct = sum(1 for r in all_results if r['label'] == r['gt_label'])
accuracy = correct / len(all_results)

print(f"\n" + "="*60)
print("CLASSIFICATION PERFORMANCE")
print("="*60)
print(f"Total samples: {len(all_results)}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.2%}")

# Count predictions
pred_positives = sum(1 for r in all_results if r['label'] == 1)
pred_negatives = sum(1 for r in all_results if r['label'] == 0)
print(f"\nPredicted positive (pothole): {pred_positives}")
print(f"Predicted negative (not pothole): {pred_negatives}")

# Group by image and apply NMS for mAP computation
print("\nApplying NMS per image...")
preds_list = []
targets_list = []

for img_idx, annotation in enumerate(dataset.annotations):
    # Get all proposals for this image
    image_proposals = []
    image_scores = []
    image_pred_labels = []
    
    for sample_idx in range(len(all_results)):
        if all_image_indices[sample_idx] == img_idx:
            result = all_results[sample_idx]
            image_proposals.append(result['box'])
            image_scores.append(result['score'])
            image_pred_labels.append(result['label'])
    
    if not image_proposals:
        # Empty predictions for this image
        preds_list.append({
            'boxes': torch.empty((0, 4), dtype=torch.float32),
            'scores': torch.empty((0,), dtype=torch.float32),
            'labels': torch.empty((0,), dtype=torch.int64),
        })
    else:
        # Convert to tensors
        boxes_tensor = torch.tensor(image_proposals, dtype=torch.float32)
        scores_tensor = torch.tensor(image_scores, dtype=torch.float32)
        
        # Apply NMS
        keep_idx = apply_nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
        
        # Keep only predicted potholes (class 1) after NMS
        kept_boxes = []
        kept_scores = []
        for idx in keep_idx:
            if image_pred_labels[idx] == 1:  # Only include predicted potholes
                kept_boxes.append(image_proposals[idx])
                kept_scores.append(image_scores[idx])
        
        if len(kept_boxes) > 0:
            preds_list.append({
                'boxes': torch.tensor(kept_boxes, dtype=torch.float32),
                'scores': torch.tensor(kept_scores, dtype=torch.float32),
                'labels': torch.ones((len(kept_scores),), dtype=torch.int64),  # All are pothole class
            })
        else:
            preds_list.append({
                'boxes': torch.empty((0, 4), dtype=torch.float32),
                'scores': torch.empty((0,), dtype=torch.float32),
                'labels': torch.empty((0,), dtype=torch.int64),
            })
    
    # Build ground-truth dict for this image
    gt_bboxes = annotation.get('bboxes', [])
    if gt_bboxes:
        targets_list.append({
            'boxes': torch.tensor([[b['xmin'], b['ymin'], b['xmax'], b['ymax']] for b in gt_bboxes], dtype=torch.float32),
            'labels': torch.ones((len(gt_bboxes),), dtype=torch.int64),  # All GT are pothole class
        })
    else:
        targets_list.append({
            'boxes': torch.empty((0, 4), dtype=torch.float32),
            'labels': torch.empty((0,), dtype=torch.int64),
        })

# Compute mAP using torchmetrics
metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True, iou_thresholds=[0.5])
metric.update(preds_list, targets_list)
map_res = metric.compute()

print('\n' + '='*60)
print('MEAN AVERAGE PRECISION (mAP)')
print('='*60)
if 'map_50' in map_res:
    print(f"AP@50 (IoU=0.5): {map_res['map_50']:.4f}")
print(f"mAP: {map_res['map']:.4f}")
if 'map_per_class' in map_res and map_res['map_per_class'] is not None:
    print(f"mAP per class: {map_res['map_per_class']}")
print('='*60 + '\n')

print(f"\n" + "="*60)
print(f"Total detections: {len(all_results)}")
print("="*60)
if len(all_results) <= 20:
    for i, det in enumerate(all_results):
        print(f"{i}: Box={det['box']}, Score={det['score']:.3f}, Label={det['label']}")
else:
    print("(Showing first 10 and last 10 detections)")
    for i in range(10):
        det = all_results[i]
        print(f"{i}: Box={det['box']}, Score={det['score']:.3f}, Label={det['label']}")
    print("...")
    for i in range(len(all_results)-10, len(all_results)):
        det = all_results[i]
        print(f"{i}: Box={det['box']}, Score={det['score']:.3f}, Label={det['label']}")
