from __future__ import annotations

import argparse
from pathlib import Path
import lightning as l
import torch
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader, random_split

from dataloader import DEFAULT_DATA_ROOTS, PotholeProposalDataset
from model_registry import build_model, get_available_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DetectionCNN on pothole proposals")

    default_root = str(DEFAULT_DATA_ROOTS["pothole"]) if "pothole" in DEFAULT_DATA_ROOTS else "./data"

    parser.add_argument("--data-root", type=str, default=default_root, help="Path to dataset root")
    parser.add_argument("--proposal-type", type=str, default="selective_search", choices=["selective_search", "edge_box"], help="Proposal algorithm to consume")
    parser.add_argument("--proposal-json", type=str, default=None, help="Optional custom proposal JSON path")
    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 256), metavar=("H", "W"), help="Resize each crop to this size")
    parser.add_argument("--iou-threshold", type=float, default=0.7, help="IoU threshold for positive samples")
    parser.add_argument("--positive-ratio", type=float, default=0.25, help="Target ratio of positive samples")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model-name", type=str, default="detection_cnn", help="Registered model name")
    parser.add_argument("--model-checkpoint", type=str, default=None, help="Path to pretrained checkpoint for the model")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--num-queries", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--pretrained-vgg", action="store_true", help="Initialize backbone from ImageNet VGG weights")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze pretrained backbone weights")
    parser.add_argument("--activation", type=str, default="silu", choices=["relu", "silu"])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--include-train-metrics", action="store_true")
    parser.add_argument("--loss-function", type=str, default="cross_entropy", choices=["cross_entropy"], help="Loss applied to averaged query logits")

    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--default-root-dir", type=str, default="./lightning_logs")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--logger-name", type=str, default="detcnn")
    parser.add_argument("--skip-test", action="store_true", help="Skip running the test loop after training")

    parser.add_argument("--list-models", action="store_true", help="Print available models and exit")

    # Early stopping is always enabled; only patience is configurable
    parser.add_argument("--early-stopping-patience", type=int, default=5, help="Epochs with no val/loss improvement before stopping")

    return parser.parse_args()


def build_datasets(args: argparse.Namespace):
    dataset = PotholeProposalDataset(
        data_root=Path(args.data_root),
        proposal_type=args.proposal_type,
        proposal_json=args.proposal_json,
        image_size=tuple(args.image_size),
        iou_threshold=args.iou_threshold,
        positive_ratio=args.positive_ratio,
        seed=args.seed,
    )

    val_len = int(len(dataset) * args.val_ratio)
    test_len = int(len(dataset) * args.test_ratio)
    val_len = max(val_len, 1) if val_len > 0 else 0
    test_len = max(test_len, 1) if test_len > 0 else 0
    train_len = len(dataset) - val_len - test_len
    if train_len <= 0:
        raise ValueError("Train split must be positive; adjust val/test ratios.")

    generator = torch.Generator().manual_seed(args.seed)
    splits = [train_len, val_len, test_len]
    train_dataset, val_dataset, test_dataset = random_split(dataset, splits, generator=generator)
    if val_len == 0:
        val_dataset = None
    if test_len == 0:
        test_dataset = None
    return train_dataset, val_dataset, test_dataset


def build_dataloaders(train_ds, val_ds, test_ds, args: argparse.Namespace):
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs) if val_ds is not None else None
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs) if test_ds is not None else None
    return train_loader, val_loader, test_loader


def main() -> None:
    args = parse_args()

    if args.list_models:
        print("Available models:")
        for name, defaults in get_available_models().items():
            print(f"- {name}: {defaults}")
        return

    l.seed_everything(args.seed, workers=True)

    train_ds, val_ds, test_ds = build_datasets(args)
    train_loader, val_loader, test_loader = build_dataloaders(train_ds, val_ds, test_ds, args)

    model_overrides = dict(
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        base_channels=args.base_channels,
        pretrained_vgg=args.pretrained_vgg,
        train_backbone=not args.freeze_backbone,
        activation=args.activation,
        dropout_p=args.dropout,
        include_train_metrics=args.include_train_metrics,
    )

    model_overrides.update(
        {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "loss_function": args.loss_function,
        }
    )

    model, _model_config = build_model(args.model_name, **model_overrides)

    if args.model_checkpoint:
        ckpt_path = Path(args.model_checkpoint).expanduser().resolve()
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
        print(f"Loaded pretrained weights from {ckpt_path}")

    csv_logger = CSVLogger(save_dir=args.default_root_dir, name=args.logger_name)
    tb_logger = TensorBoardLogger(save_dir=args.default_root_dir, name=f"{args.logger_name}_tb")

    early_stopping = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=args.early_stopping_patience,
        min_delta=0.0,
    )

    trainer = l.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        default_root_dir=args.default_root_dir,
        logger=[csv_logger, tb_logger],
        callbacks=[early_stopping],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume_from_checkpoint)

    if not args.skip_test and test_loader is not None:
        trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
