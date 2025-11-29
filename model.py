import math
import warnings
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Mapping, Optional

import torch
import torch.nn as nn
import lightning as l
from torchvision import models
from torchmetrics import Metric
from torchmetrics import Accuracy, Precision, Recall

def _load_vgg_backbone(
    *,
    target_convs: Sequence[nn.Conv2d],
    target_bns: Optional[Sequence[Optional[nn.Module]]] = None,
    pretrained: bool,
    vgg_variant: str,
    freeze_backbone: bool,
) -> None:
    """Load ImageNet-pretrained VGG weights into the provided conv/backbone stack."""

    if not pretrained or not target_convs:
        return

    if target_bns is None:
        target_bns = [None] * len(target_convs)
    if len(target_bns) != len(target_convs):
        raise ValueError("target_bns must match target_convs length.")

    try:
        weights_enum = getattr(models, f"{vgg_variant.upper()}_Weights")
        weights = getattr(weights_enum, "DEFAULT")
        vgg = getattr(models, vgg_variant)(weights=weights)
    except AttributeError:
        vgg = getattr(models, vgg_variant)(pretrained=True)

    source_convs = [module for module in vgg.features if isinstance(module, nn.Conv2d)]
    vgg_layers = list(vgg.features)

    def _match_conv(dst_conv: nn.Conv2d) -> Optional[nn.Conv2d]:
        for candidate in source_convs:
            if candidate.out_channels == dst_conv.out_channels:
                return candidate
        return None

    def _match_bn(src_conv: nn.Conv2d) -> Optional[nn.BatchNorm2d]:
        start_idx = vgg_layers.index(src_conv)
        for layer in vgg_layers[start_idx + 1 :]:
            if isinstance(layer, nn.BatchNorm2d):
                return layer
            if isinstance(layer, nn.Conv2d):
                break
        return None

    def _adapt_weight(weight: torch.Tensor, dst_channels: int) -> torch.Tensor:
        if weight.shape[1] == dst_channels:
            return weight
        if dst_channels < weight.shape[1]:
            return weight[:, :dst_channels, :, :]
        repeat_factor = math.ceil(dst_channels / weight.shape[1])
        expanded = weight.repeat(1, repeat_factor, 1, 1)
        return expanded[:, :dst_channels, :, :] / repeat_factor

    for idx, (conv, bn_layer) in enumerate(zip(target_convs, target_bns)):
        matched_conv = _match_conv(conv)
        if matched_conv is None:
            warnings.warn(
                f"No compatible VGG conv found for target layer {idx} (out={conv.out_channels}).",
                stacklevel=2,
            )
            continue
        with torch.no_grad():
            copied_weight = _adapt_weight(matched_conv.weight, conv.in_channels)
            if copied_weight.shape != conv.weight.shape:
                resized = copied_weight.view(
                    copied_weight.shape[0], copied_weight.shape[1], *copied_weight.shape[2:]
                )
                conv.weight.copy_(resized[: conv.out_channels, : conv.in_channels])
            else:
                conv.weight.copy_(copied_weight)
            if matched_conv.bias is not None and conv.bias is not None:
                conv.bias.copy_(matched_conv.bias)

        # Logging which VGG layer was used to initialise this target conv
        try:
            print(
                f"[VGG_COPY] target_idx={idx} target_conv_out={conv.out_channels} target_conv_in={conv.in_channels} "
                f"<- source_conv_shape={tuple(matched_conv.weight.shape)}"
            )
        except Exception:
            pass

        if bn_layer is not None:
            matched_bn = _match_bn(matched_conv)
            if matched_bn is None:
                continue
            with torch.no_grad():
                weight_param = getattr(bn_layer, "weight", None)
                bias_param = getattr(bn_layer, "bias", None)
                running_mean = getattr(bn_layer, "running_mean", None)
                running_var = getattr(bn_layer, "running_var", None)
                if isinstance(weight_param, torch.Tensor) and matched_bn.weight is not None:
                    weight_param.copy_(matched_bn.weight)
                if isinstance(bias_param, torch.Tensor) and matched_bn.bias is not None:
                    bias_param.copy_(matched_bn.bias)
                if isinstance(running_mean, torch.Tensor) and matched_bn.running_mean is not None:
                    running_mean.copy_(matched_bn.running_mean)
                if isinstance(running_var, torch.Tensor) and matched_bn.running_var is not None:
                    running_var.copy_(matched_bn.running_var)

            # Log BN mapping
            try:
                print(
                    f"[VGG_COPY] target_bn for target_idx={idx} target_bn_shape={(getattr(bn_layer,'weight').numel() if getattr(bn_layer,'weight',None) is not None else 'None')} "
                    f"<- source_bn_shape={(getattr(matched_bn,'weight').numel() if getattr(matched_bn,'weight',None) is not None else 'None')}"
                )
            except Exception:
                pass

    if freeze_backbone:
        for conv, bn_layer in zip(target_convs, target_bns):
            for param in conv.parameters():
                param.requires_grad = False
            if bn_layer is not None:
                for param in bn_layer.parameters():
                    param.requires_grad = False
                if isinstance(bn_layer, nn.BatchNorm2d):
                    bn_layer.eval()


def _unfreeze_conv_bn_pair(
    conv_bn_pairs: list[tuple[nn.Module, Optional[nn.Module]]],
    freeze_bn_running_stats: bool = True
) -> None:
    """
    Unfreeze multiple conv + BN layer pairs.

    Args:
        conv_bn_pairs: list of tuples (conv_layer, bn_layer). bn_layer can be None.
        freeze_bn_running_stats: if True, keeps BN running stats frozen (calls eval() on BN).
    """
    for conv_layer, bn_layer in conv_bn_pairs:
        # Unfreeze conv
        for p in conv_layer.parameters():
            p.requires_grad = True
        # Unfreeze BN if given
        if bn_layer is not None:
            for p in bn_layer.parameters():
                p.requires_grad = True
            if freeze_bn_running_stats and isinstance(bn_layer, (nn.BatchNorm2d, nn.BatchNorm3d)):
                bn_layer.eval()

    if not freeze_bn_running_stats:
        print(f"[UNFREEZE] Unfroze CONV: {conv_layer.__class__.__name__},\n"
              f"[UNFREEZE] Unfroze BN: {bn_layer.__class__.__name__ if bn_layer is not None else 'None'}")

    else: 
        print(f"[FROZEN] Copied weights and froze BN running stats for CONV: {conv_layer.__class__.__name__},\n,"
              f"[FROZEN] Copied weights and froze BN running stats for BN: {bn_layer.__class__.__name__ if bn_layer is not None else 'None'}")

class BaseModel(l.LightningModule):
    """Base Lightning module that centralizes metric registration and logging."""

    def __init__(
        self,
        *,
        task_type: str = "classification",
        metric_config: Optional[dict[str, Any]] = None,
        optimizer_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.feature_dim: Optional[int] = None
        self.task_type = task_type
        self.train_metrics = nn.ModuleDict()
        self.val_metrics = nn.ModuleDict()
        self.test_metrics = nn.ModuleDict()
        self._metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }
        self._metric_prog_bar = {
            "train": set(),
            "val": set(),
            "test": set(),
        }
        default_optim = {"lr": 1e-3, "weight_decay": 0.0}
        self.optimizer_config = {**default_optim, **(optimizer_config or {})}
        # Detection metrics (e.g., COCO mAP) are currently disabled to
        # keep the training loop simple and avoid extra dependencies.
        # You can re-enable this once the detection target formatting
        # is wired up correctly.
        # self._setup_default_metrics(metric_config or {})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def register_metrics(
        self,
        stage: str,
        metrics: Mapping[str, Metric],
        *,
        prog_bar_keys: Optional[Iterable[str]] = None,
    ) -> None:
        if stage not in self._metrics:
            raise ValueError(f"Unsupported stage '{stage}'. Use train/val/test.")
        stage_attr = f"{stage}_metrics"
        new_collection = nn.ModuleDict(metrics)
        if hasattr(self, stage_attr):
            setattr(self, stage_attr, new_collection)
        self._metrics[stage] = new_collection
        self._metric_prog_bar[stage] = set(prog_bar_keys or set())

    def update_metrics(self, stage: str, *metric_args: Any, **metric_kwargs: Any) -> None:
        if stage not in self._metrics:
            raise ValueError(f"Unsupported stage '{stage}'. Use train/val/test.")
        metric_dict = self._metrics[stage]
        if len(metric_dict) == 0:
            return
        for metric in metric_dict.values():
            if isinstance(metric, Metric):
                metric.update(*metric_args, **metric_kwargs)

    def _log_stage_metrics(self, stage: str) -> None:
        metric_dict = self._metrics.get(stage)
        if metric_dict is None or len(metric_dict) == 0:
            return
        prog_bar_keys = self._metric_prog_bar.get(stage, set())
        for name, metric in metric_dict.items():
            if not isinstance(metric, Metric):
                continue
            value = metric.compute()
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    # Only log scalar-like values; skip vectors, empty tensors, etc.
                    is_scalar = False
                    if isinstance(sub_value, (int, float)):
                        is_scalar = True
                    elif isinstance(sub_value, torch.Tensor) and sub_value.ndim == 0:
                        is_scalar = True
                    if not is_scalar:
                        continue

                    composite_name = f"{name}.{sub_key}"
                    log_name = f"{stage}/{composite_name}"
                    self.log(
                        log_name,
                        sub_value,
                        prog_bar=composite_name in prog_bar_keys,
                        on_step=False,
                        on_epoch=True,
                    )
            else:
                log_name = f"{stage}/{name}"
                self.log(
                    log_name,
                    value,
                    prog_bar=name in prog_bar_keys,
                    on_step=False,
                    on_epoch=True,
                )
            if hasattr(metric, "reset"):
                metric.reset()

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self._log_stage_metrics("train")

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        self._log_stage_metrics("val")

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()
        self._log_stage_metrics("test")

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement _shared_step")

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, "train")
        self.log("train/loss_step", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config.get("lr", 1e-3),
            weight_decay=self.optimizer_config.get("weight_decay", 0.0),
        )

    def _setup_default_metrics(self, metric_config: dict[str, Any]) -> None:
        # Temporarily disabled: detection metrics are not wired up yet.
        # Left here for future use when full detection training targets
        # (boxes + labels per image) are available.
        return

    def _configure_detection_metrics(self, metric_config: dict[str, Any]) -> None:
        # Detection metrics (COCO mAP) are intentionally disabled here.
        # Keep this method as a no-op until detection targets (boxes+labels)
        # are fully wired up in the dataset and training loop.
        return


class DetectionCNN(BaseModel):

    def __init__(
        self,
        num_classes: int = 2,
        *,
        in_channels: int = 3,
        pretrained_vgg: bool = False,
        train_backbone: bool = True,
        vgg_variant: str = "vgg16_bn",
        dropout_p: float = 0.2,
        activation: str = "silu",
        base_channels: int = 64,
        include_train_metrics: bool = False,
        metric_iou_thresholds: Optional[Sequence[float]] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_function: str = "cross_entropy",
    ):
        metric_config = {
            "include_train_metrics": include_train_metrics,
            "metric_iou_thresholds": metric_iou_thresholds,
        }
        optimizer_config = {"lr": learning_rate, "weight_decay": weight_decay}
        super().__init__(
            task_type="detection",
            metric_config=metric_config,
            optimizer_config=optimizer_config,
        )
        self.num_classes = num_classes
        self.loss_function = loss_function.lower()
        if self.loss_function != "cross_entropy":
            raise ValueError("DetectionCNN currently supports only 'cross_entropy' loss.")
        self.criterion = nn.CrossEntropyLoss()
        act_cls = nn.ReLU if activation.lower() == "relu" else nn.SiLU
        widths = (
            [64, 128, 256]
            if pretrained_vgg
            else [base_channels, base_channels * 2, base_channels * 4]
        )
        
        def _conv_block(conv: nn.Module, norm: nn.Module) -> nn.Sequential:
            return nn.Sequential(conv, norm, act_cls(), nn.MaxPool2d(2), nn.Dropout2d(p=dropout_p))
        
        self.feature_dim = widths[-1]

        # Backbone blocks
        self.conv1 = nn.Conv2d(in_channels, widths[0], kernel_size=3, padding=1)
        # Use BatchNorm only (remove GroupNorm/LayerNorm support)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.conv2 = nn.Conv2d(widths[0], widths[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(widths[1])
        self.conv3 = nn.Conv2d(widths[1], widths[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(widths[2])


        self.features = nn.Sequential(
            _conv_block(self.conv1, self.bn1),
            _conv_block(self.conv2, self.bn2),
            nn.Sequential(self.conv3, self.bn3, act_cls()),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.head_dropout = nn.Dropout(p=dropout_p)

        # Classification head (single prediction per crop)
        self.cls_head = nn.Linear(self.feature_dim, num_classes)

        # Optionally bootstrap with ImageNet weights
        _load_vgg_backbone(
            target_convs=[self.conv1, self.conv2, self.conv3],
            target_bns=[self.bn1, self.bn2, self.bn3],
            pretrained=pretrained_vgg,
            vgg_variant=vgg_variant,
            freeze_backbone=not train_backbone,
        )

        if pretrained_vgg and train_backbone:
            _unfreeze_conv_bn_pair(
                conv_bn_pairs=[(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)],
                freeze_bn_running_stats=True,
            )

        # Metric factory to create fresh Metric instances per stage
        def _make_metrics():
            return {
                "acc": Accuracy(task="multiclass", num_classes=self.num_classes),
                "precision": Precision(task="multiclass", average="macro", num_classes=self.num_classes),
                "recall": Recall(task="multiclass", average="macro", num_classes=self.num_classes),
            }

        # Register metrics separately for each stage (don't reuse instances)
        self.register_metrics("val", _make_metrics(), prog_bar_keys={"acc", "precision", "recall"})
        self.register_metrics("test", _make_metrics())
        if include_train_metrics:
            self.register_metrics("train", _make_metrics())

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.features(x)
        feats = self.spatial_pool(feats)
        flat = torch.flatten(feats, 1)
        flat = self.head_dropout(flat)
        logits = self.cls_head(flat)
        return logits

    # decode_predictions removed — this model is classification-only and
    # the training/validation loop works with logits/predictions directly.
    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        images, labels = batch
        logits = self(images)  # shape (B, num_classes)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        # Update registered metrics (if any)
        try:
            self.update_metrics(stage, preds, labels)
        except Exception:
            # fallback: compute simple accuracy for logging if metrics not present
            acc = (preds == labels).float().mean()
            self.log(f"{stage}/acc", acc, on_step=False, on_epoch=True, prog_bar=stage != "test")

        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=stage != "test")
        return loss
