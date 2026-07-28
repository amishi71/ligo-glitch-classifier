"""Model definitions."""
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(num_classes, pretrained=True):
    """Baseline single-view CNN: ResNet18 fine-tuned from ImageNet weights
    (project brief, section 3.4 — transfer learning). Swap this out for the
    multi-view fusion architecture in section 3.3 once the baseline works."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
