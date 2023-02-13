import json
import typing as t

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from PIL import Image
from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel
from captum.attr import Saliency
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from torchvision import transforms
from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

from saliency.fullgrad import FullGrad
from saliency.misc_functions import save_saliency_map
from saliency.gradcam import GradCAM


def eval_saliency(model: nn.Module, image: torch.Tensor, target: t.Optional[torch.Tensor] = None):
    saliency = Saliency(model)
    attributions_saliency = saliency.attribute(image, target=target)

    default_cmap = LinearSegmentedColormap.from_list(
        "custom blue", [(0, "#ffffff"), (0.25, "#000000"), (1, "#000000")], N=256
    )
    fig, _ = viz.visualize_image_attr_multiple(
        np.transpose(attributions_saliency.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "positive"],
        cmap=default_cmap,
        show_colorbar=True
    )
    fig.savefig("saliency.png")


def eval_integrated_gradient(model: nn.Module, image: torch.Tensor, target: t.Optional[torch.Tensor] = None):
    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(image, target=target, n_steps=200)

    default_cmap = LinearSegmentedColormap.from_list(
        "custom blue", [(0, "#ffffff"), (0.25, "#000000"), (1, "#000000")], N=256
    )
    fig, _ = viz.visualize_image_attr_multiple(
        np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "positive"],
        cmap=default_cmap,
        show_colorbar=True
    )
    fig.savefig("ig.png")


def eval_integrated_gradient_with_noise_tunnel(
        model: nn.Module, image: torch.Tensor, target: t.Optional[torch.Tensor] = None
):
    integrated_gradients = IntegratedGradients(model)
    noise_tunnel = NoiseTunnel(integrated_gradients)
    attributions_ig_nt = noise_tunnel.attribute(image, nt_samples=10, nt_type="smoothgrad_sq", target=target)

    default_cmap = LinearSegmentedColormap.from_list(
        "custom blue", [(0, "#ffffff"), (0.25, "#000000"), (1, "#000000")], N=256
    )
    fig, _ = viz.visualize_image_attr_multiple(
        np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "positive"],
        cmap=default_cmap,
        show_colorbar=True
    )
    fig.savefig("ig_nt.png")


def eval_full_gradient(
        model: nn.Module, image: torch.Tensor, target: t.Optional[torch.Tensor] = None
):
    fullgrad = FullGrad(model)
    fullgrad.checkCompleteness()
    saliency_map = fullgrad.saliency(image, target)

    save_saliency_map(image[0, ...], saliency_map[0, ...], "full_grad.png")


def eval_grad_cam(
        model: nn.Module, image: torch.Tensor, target: t.Optional[torch.Tensor] = None
):
    grad_cam = GradCAM(model)
    saliency_map = grad_cam.saliency(image, target)

    save_saliency_map(image[0, ...], saliency_map[0, ...], "grad_cam.png")


def main():
    with open("./imagenet1000_clsidx_to_labels.json") as json_data:
        idx_to_labels = json.load(json_data)

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    image = torch.unsqueeze(transform(Image.open("./n01820546_14917.JPEG")), dim=0)

    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transformed_image = transform_normalize(image)

    output = model(transformed_image)
    output = functional.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
    print("Predicted:", predicted_label, "(", prediction_score.squeeze().item(), ")")

    # Gradient-based attribution
    # eval_integrated_gradient(model=model, image=transformed_image, target=pred_label_idx)
    # eval_integrated_gradient_with_noise_tunnel(model=model, image=transformed_image, target=pred_label_idx)
    # eval_saliency(model=model, image=image, target=pred_label_idx)
    # eval_full_gradient(model=model, image=image, target=pred_label_idx)
    eval_grad_cam(model=model, image=image, target=pred_label_idx)


if __name__ == "__main__":
    main()
