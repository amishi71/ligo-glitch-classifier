"""Model definitions: single-view baseline CNN and multi-view fusion CNN."""
import torch
import torch.nn as nn
from torchvision import models


_BACKBONES = {
    "resnet18": (models.resnet18, "IMAGENET1K_V1", 512),
    "resnet34": (models.resnet34, "IMAGENET1K_V1", 512),
    "efficientnet_b0": (models.efficientnet_b0, "IMAGENET1K_V1", 1280),
}


def build_single_view_model(num_classes: int, backbone_name: str = "resnet18", pretrained: bool = True) -> nn.Module:
    """Transfer-learning baseline: pretrained CNN backbone + new classification head."""
    if backbone_name not in _BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone_name}'. Choose from {list(_BACKBONES)}")

    fn, weights_name, feat_dim = _BACKBONES[backbone_name]
    weights = weights_name if pretrained else None
    model = fn(weights=weights)

    if backbone_name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone_name.startswith("efficientnet"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    return model


class MultiViewCNN(nn.Module):
    """Stretch-goal architecture matching Gravity Spy's own approach: one CNN tower
    per Omega-scan duration view (0.5s / 1s / 2s / 4s), features concatenated before
    the final classifier. Expects `forward(views)` where `views` is a list of 4
    image tensors, each shaped [batch, 3, H, W]."""

    def __init__(self, num_classes: int, backbone_name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if backbone_name not in _BACKBONES:
            raise ValueError(f"Unknown backbone '{backbone_name}'. Choose from {list(_BACKBONES)}")

        fn, weights_name, feat_dim = _BACKBONES[backbone_name]
        weights = weights_name if pretrained else None

        self.towers = nn.ModuleList()
        for _ in range(4):
            backbone = fn(weights=weights)
            if backbone_name.startswith("resnet"):
                backbone.fc = nn.Identity()
            elif backbone_name.startswith("efficientnet"):
                backbone.classifier = nn.Identity()
            self.towers.append(backbone)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, views: list):
        assert len(views) == 4, "MultiViewCNN expects exactly 4 duration-view images."
        feats = [tower(v) for tower, v in zip(self.towers, views)]
        combined = torch.cat(feats, dim=1)
        return self.classifier(combined)