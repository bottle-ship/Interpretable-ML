#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

"""
    Implement GradCAM

    Original Paper:
    Selvaraju, Ramprasaath R., et al. "Grad-cam: Visual explanations from deep networks
    via gradient-based localization." ICCV 2017.

"""

import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMExtractor(object):
    # Extract tensors needed for GradCAM using hooks
    _features: torch.Tensor
    _feature_gradient: torch.Tensor

    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self.layer = layer

        self.layer.register_backward_hook(self._extract_layer_grads)
        self.layer.register_forward_hook(self._extract_layer_features)

    def get_features_and_gradients(self, x: torch.Tensor, target_class: t.Optional[torch.Tensor]):
        out = self.model(x)

        if target_class is None:
            target_class = out.data.max(1, keepdim=True)[1]

        output_scalar = -F.nll_loss(out, target_class.flatten(), reduction='sum')

        # Compute gradients
        self.model.zero_grad()
        output_scalar.backward()

        return self._features, self._feature_gradient

    def _extract_layer_grads(self, module: nn.Module, in_grad: t.Tuple[torch.Tensor], out_grad: t.Tuple[torch.Tensor]):
        # function to collect the gradient outputs
        self._feature_gradient = out_grad[0]

    def _extract_layer_features(self, module: nn.Module, input: t.Tuple[torch.Tensor], output: torch.Tensor):
        # function to collect the layer outputs
        self._features = output


class GradCAM(object):
    """
    Compute GradCAM
    """

    def __init__(self, model: nn.Module):
        self.model = model

        self._extractor = GradCAMExtractor(self.model, self._select_target_layer())

    def attribution(self, image: torch.Tensor, target_class: t.Optional[torch.Tensor] = None):
        self.model.eval()
        features, intermed_grad = self._extractor.get_features_and_gradients(image, target_class=target_class)

        # GradCAM computation
        grads = intermed_grad.mean(dim=(2, 3), keepdim=True)
        cam = (F.relu(features) * grads).sum(1, keepdim=True)
        attribution = F.interpolate(F.relu(cam), size=image.size(2), mode="bilinear", align_corners=True)

        return attribution

    def _select_target_layer(self) -> nn.Module:
        # Iterate through layers
        prev_module = None
        target_module = None
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                prev_module = m
            elif isinstance(m, nn.Linear):
                target_module = prev_module
                break

        return target_module
