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
    proposal_json=None,
    image_size=(256, 256),
)

test_len = int(len(dataset) * 0.15)
test_len = max(test_len, 1)
train_val_len = len(dataset) - test_len
_, test_dataset = random_split(dataset, [train_val_len, test_len], generator=torch.Generator().manual_seed(42))

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
print(f"Test dataset: {len(test_dataset)} samples")

checkpoint_path = "logs/lightning_logs/detcnn_eb_vgg/version_0/checkpoints/epoch=6-step=98.ckpt"

spec = resolve_model("detection_cnn")
model = spec.model_class.load_from_checkpoint(checkpoint_path, **spec.default_params)
model.eval()
model.to(device)
print(f"Model loaded successfully from checkpoint: {checkpoint_path}")

all_results = []
with torch.no_grad():
    for batch in test_loader:
        images, labels = batch
        images = images.to(device)

        outputs = model(images)
        logits = outputs["logits"]
        boxes = outputs["boxes"]

        probs = torch.softmax(logits, dim=-1)
        scores = probs[..., 1]  # class 1 = pothole
        pred_labels = (scores > 0.5).long()

        for b in range(images.size(0)):
            b_boxes = boxes[b]
            b_scores = scores[b]
            keep_idx = apply_nms(b_boxes, b_scores, iou_threshold=0.5)

            final_boxes = b_boxes[keep_idx].cpu()
            final_scores = b_scores[keep_idx].cpu()
            final_labels = pred_labels[b][keep_idx].cpu()

            for box, score, label in zip(final_boxes, final_scores, final_labels):
                all_results.append({
                    "box": box.tolist(),
                    "score": float(score),
                    "label": int(label)
                })

print(f"Total detections after NMS: {len(all_results)}")
for i, det in enumerate(all_results):
    print(f"{i}: Box={det['box']}, Score={det['score']:.3f}, Label={det['label']}")
