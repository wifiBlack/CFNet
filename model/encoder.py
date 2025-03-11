from typing import List, Tuple
import torch.nn as nn
import torch
import torchvision
from model.utils import *
class Encoder(nn.Module):
    """
    A class for encoding two images using a shared backbone model.

    The class takes in two images and passes them through a shared backbone model.
    The feature maps from each layer are stored and returned as a list of two lists.
    """

    def __init__(self,channel_list):
        """
        Initialize the encoder model.
        """
        super().__init__()
        self._backbone = self._get_backbone('efficientnet_b5', weights='DEFAULT', output_layer_bkbn='4', freeze_backbone=False)
        # self.se_blocks_1 = [SE_Block(channel).to("cuda") for channel in channel_list[1:] ]
        # self.se_blocks_2 = [SE_Block(channel).to("cuda") for channel in channel_list[1:] ]
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through the backbone, but store feature maps from each layer.

        Args:
            x1: Input tensor 1
            x2: Input tensor 2

        Returns:
            List of feature maps from each layer for x1 and x2
        """
        Y1: List[torch.Tensor] = []
        Y2: List[torch.Tensor] = []
        for i, layer in enumerate(self._backbone):
            # Forward pass through the backbone
            x1 = layer(x1)
            x2 = layer(x2)
            
            # Store feature maps from each layer
            if i != 0:
                # x1 = self.se_blocks_1[i](x1)
                # x2 = self.se_blocks_2[i](x2)
                Y1.append(x1)
                Y2.append(x2)
        return Y1, Y2

    @staticmethod
    def _get_backbone(bkbn_name: str, weights: str, output_layer_bkbn: str, freeze_backbone: bool) -> nn.Module:
        """
        Get the backbone model.

        Args:
            bkbn_name: The name of the backbone to use.
            weights: The weights to use for the backbone.
            output_layer_bkbn: The name of the output layer of the backbone.
            freeze_backbone: Whether to freeze the weights of the backbone.

        Returns:
            The backbone model.
        """
        # The whole model:
        entire_model = getattr(torchvision.models, bkbn_name)(weights=weights).features

        # Slicing it:
        derived_model = nn.ModuleList([])
        for name, layer in entire_model.named_children():
            derived_model.append(layer)
            if name == output_layer_bkbn:
                break

        # Freezing the backbone weights:
        if freeze_backbone:
            for param in derived_model.parameters():
                param.requires_grad = False
                # This is important to prevent the frozen weights from being updated
                # when calling the model with the `requires_grad` argument set to `True`.

        return derived_model
