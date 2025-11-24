from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import lightning as L

from model import DetectionCNN


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_class: type[L.LightningModule]
    description: str = ""
    default_params: Optional[dict] = None


MODEL_SPECS: Dict[str, ModelSpec] = {
    "detection_cnn": ModelSpec(
        name="detection_cnn",
        model_class=DetectionCNN,
        description="Compact CNN baseline for proposal-based pothole detection.",
        default_params={
            "num_classes": 2,
            "in_channels": 3,
            "num_queries": 64,
            "pretrained_vgg": False,
            "train_backbone": True,
            "vgg_variant": "vgg16_bn",
            "dropout_p": 0.2,
            "activation": "silu",
            "norm_type": "batch",
            "base_channels": 64,
            "include_train_metrics": False,
            "metric_iou_thresholds": None,
            "box_format": "xyxy",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "loss_function": "cross_entropy",
        },
    ),
}


def resolve_model(name: str) -> ModelSpec:
    if name not in MODEL_SPECS:
        available = ", ".join(MODEL_SPECS)
        raise KeyError(f"Unknown model '{name}'. Available models: {available}")
    return MODEL_SPECS[name]


def get_available_models() -> Dict[str, Dict[str, Any]]:
    return {
        name: {
            "description": spec.description,
            "default_params": spec.default_params,
        }
        for name, spec in MODEL_SPECS.items()
    }


def build_model(
    name: str,
    **model_kwargs,
) -> Tuple[L.LightningModule, dict]:
    spec = resolve_model(name)
    params = spec.default_params.copy() if spec.default_params else {}
    params.update(model_kwargs)

    model = spec.model_class(**params)

    model_config = {
        "name": name,
        "kwargs": params,
    }

    print(f"Built {spec.name}: {spec.description}")
    print(f"  Modes: {params.get('n_modes', 'N/A')}")
    print(f"  Hidden channels: {params.get('hidden_channels', 'N/A')}")
    print(f"  Layers: {params.get('n_layers', 'N/A')}")
    print(f"  Learning rate: {params['learning_rate']}, Loss: {params['loss_function']}")

    return model, model_config


def rebuild_model_from_config(model_config: dict) -> L.LightningModule:
    return build_model(
        name=model_config["name"],
        **model_config["kwargs"],
    )[0]


__all__ = [
    "ModelSpec",
    "MODEL_SPECS",
    "build_model",
    "rebuild_model_from_config",
    "resolve_model",
    "get_available_models",
]
