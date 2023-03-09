import json

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from torchvision import transforms
from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

from iml.attribution import GuidedIG

transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )


conv_layer_outputs = {}


def call_model_function(images, call_model_args=None, expected_keys=None):
    images = np.transpose(images, (0, 3, 1, 2))
    images = torch.tensor(images, dtype=torch.float32)
    images = transform_normalize.forward(images)
    images = images.requires_grad_(True)

    # images = preprocess_images(images)
    target_class_idx = call_model_args["class_idx_str"]
    model = call_model_args["model"]
    output = model(images)
    m = torch.nn.Softmax(dim=1)
    output = m(output)
    if "INPUT_OUTPUT_GRADIENTS" in expected_keys:
        outputs = output[:, target_class_idx]
        grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
        grads = torch.movedim(grads[0], 1, 3)
        gradients = grads.detach().numpy()
        return {"INPUT_OUTPUT_GRADIENTS": gradients}
    else:
        one_hot = torch.zeros_like(output)
        one_hot[:, target_class_idx] = 1
        model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        return conv_layer_outputs


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

    transformed_image = transform_normalize(image)

    output = model(image)
    output = functional.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
    print("Predicted:", predicted_label, "(", prediction_score.squeeze().item(), ")")

    call_model_args = {
        "class_idx_str": pred_label_idx.item(),
        "model": model
    }

    im = np.asarray(image.detach().cpu().numpy()).astype(np.float32)[0, ...]
    im = np.transpose(im, (1, 2, 0))

    guided_ig = GuidedIG()
    baseline = np.zeros(im.shape)

    attribution = guided_ig.get_mask(
        im, call_model_function, call_model_args, x_steps=25, x_baseline=baseline, max_dist=1.0, fraction=0.5
    )

    default_cmap = LinearSegmentedColormap.from_list(
        "custom blue", [(0, "#ffffff"), (0.25, "#000000"), (1, "#000000")], N=256
    )
    fig, _ = viz.visualize_image_attr_multiple(
        attribution,
        np.transpose(transformed_image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "positive"],
        cmap=default_cmap,
        show_colorbar=True
    )
    fig.savefig("guided_ig.png")


if __name__ == "__main__":
    main()
