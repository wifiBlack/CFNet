from typing import List
import torch.nn as nn
import torch
from model.utils import *

class ChangeDecoder(nn.Module):
    def __init__(self, channel_list: List[int]):
        """
        Initialize the ChangeDecoder model.

        Args:
            channel_list (List[int]): A list of the number of channels in the feature maps
                from the backbone model, from highest to lowest resolution.
        """
        super().__init__()
        self.fuseconv3ds = nn.ModuleList([FuseConv3d(channel) for channel in reversed(channel_list)])
        self.aggregations = nn.ModuleList([Aggregation(in_channel=in_channel, out_channel=out_channel)
                                          for in_channel, out_channel in zip(reversed(channel_list), reversed(channel_list[1:-1]))])
        # self.upconv = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
        #                           CBA1x1(channel_list[1],1))
        self.upconv = nn.Sequential(nn.ConvTranspose2d(in_channels=channel_list[1], out_channels=1, kernel_size=3, stride=2, padding=1,output_padding=1,bias=False))
        
    def forward(self, X1: List[torch.Tensor], X2: List[torch.Tensor],focuses:List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the ChangeDecoder model.

        Args:
            X1 (List[torch.Tensor]): The feature maps from the first image at different scales.
            X2 (List[torch.Tensor]): The feature maps from the second image at different scales.

        Returns:
            List[torch.Tensor]: The feature maps at different scales after the change detection.
        """
        # Initialize an empty list to store the feature maps at different scales
        Y = []

        # Iterate over the feature maps at different scales
        for i in range(len(X1)):
            # Compute the feature map at the current scale using MixConv3d
            Fusion = self.fuseconv3ds[i](X1[i], X2[i])
            Fusion = torch.mul(focuses[i].unsqueeze(1),Fusion)
            # If it is not the first scale, aggregate the feature maps at the current and previous scales
            if i>0:
                Fusion = self.aggregations[i-1](Y[i-1], Fusion)
            # Append the feature map at the current scale to the list
            Y.append(Fusion)

        # Return the last of list of feature maps at different scales
        return torch.tanh(self.upconv(Y[-1]).squeeze(1))


class FuseConv3d(nn.Module):
    def __init__(self, ch_out: int):
        
        super().__init__()
        self.conv = nn.Sequential(
            # Convolve the input with a 3D kernel
            nn.Conv3d(ch_out, ch_out, kernel_size=(2, 3, 3),
                      stride=(2, 1, 1), padding=(0, 1, 1)),
            # Normalize the output
            nn.BatchNorm3d(ch_out),
            # Apply ReLU activation
            nn.ReLU(),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MixConv3d layer.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Unsqueeze the input tensors to 3D
        x1 = x1.unsqueeze(2)
        x2 = x2.unsqueeze(2)

        # Concatenate the input tensors in the channel dimension
        x = torch.cat([x1, x2], dim=2)

        # Apply the convolutional layer
        x = self.conv(x)

        # Squeeze the output tensor to 2D
        x = x.squeeze(2)

        return x
