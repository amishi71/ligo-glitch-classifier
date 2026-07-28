"""Grad-CAM sanity check: visualize what part of the spectrogram the model is
actually looking at. Useful to confirm it's reading glitch morphology and not
plot axes/colorbar artifacts.

Usage:
    python -m src.gradcam_demo --config configs/config.yaml --index 0
"""
import argparse
import os

import matplotlib.pyplot as plt
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.dataset import build_dataloaders
from src.models import build_single_view_model
from src.utils import load_config, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--index", type=int, default=0, help="Index into the test set")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    checkpoint_path = os.path.join(
        config["output"]["checkpoint_dir"], config["output"]["best_model_name"]
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]

    _, _, _, _, _, test_ds = build_dataloaders(config, classes)

    model = build_single_view_model(
        num_classes=len(classes),
        backbone_name=config["model"]["backbone"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Only resnet-style backbones supported here (layer4 is the last conv block)
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    img_tensor, label = test_ds[args.index]
    grayscale_cam = cam(input_tensor=img_tensor.unsqueeze(0).to(device))[0]

    rgb_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.title(f"True class: {classes[label]}")
    plt.axis("off")

    os.makedirs(config["output"]["outputs_dir"], exist_ok=True)
    out_path = os.path.join(config["output"]["outputs_dir"], f"gradcam_example_{args.index}.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved Grad-CAM visualization to {out_path}")


if __name__ == "__main__":
    main()