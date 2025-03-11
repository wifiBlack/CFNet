from typing import List
import torch
import torch.nn as nn
from model.utils import *

class ContentDecoder(nn.Module):
    def __init__(self, channel_list: List[int]):
        """
        Initialize the ContentDecoder model.

        The ContentDecoder model takes a list of feature maps from the backbone model,
        from highest to lowest resolution, and passes them through a series of aggregation layers and SE blocks.

        Args:
            channel_list (List[int]): A list of the number of channels in the feature maps
                from the backbone model, from highest to lowest resolution.
        """
        super().__init__()
        self.channel_list = channel_list
        
        # Initialize the aggregation layers
        self.aggregations = nn.ModuleList([Aggregation(in_channel=in_channel, out_channel=out_channel)
                                          for in_channel, out_channel in zip(reversed(channel_list), reversed(channel_list[1:-1]))])
        
        # Initialize the SE blocks
        # self.seblocks = nn.ModuleList([SEBlock(in_channel) for in_channel in reversed(channel_list[1:])])
        self.cbam = nn.ModuleList([CBAM(in_channel) for in_channel in reversed(channel_list[1:])])
    def forward(self, X: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through the ContentDecoder model.

        The ContentDecoder model takes a list of feature maps from the backbone model,
        from highest to lowest resolution, and passes them through a series of aggregation layers and SE blocks.

        Args:
            X (List[torch.Tensor]): A list of feature maps from the backbone model,
                from highest to lowest resolution.

        Returns:
            List[torch.Tensor]: The final output of the ContentDecoder model.
        """
        
        # X[-1] = self.cbam[0](X[-1])
        Y = [X[-1]]  # The output of the highest resolution layer is the input of the first aggregation layer.
        
        for idx,x in enumerate(reversed(X[:-1])):  # Iterate over the feature maps, from highest to lowest resolution.
            y = self.aggregations[idx](Y[idx], x)  # Pass the feature map through the aggregation layer.
            # y = self.cbam[idx+1](y)
            Y.append(y)  # Append the output of the aggregation layer to the list of feature maps.
        
        for idx,y in enumerate(Y):  # Iterate over the feature maps, from highest to lowest resolution.
            # y = self.seblocks[idx](y)  # Pass the feature map through the SE block.
            y = self.cbam[idx](y)
        
        return Y
    
