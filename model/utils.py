import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os

class CBA3x3(nn.Module):
    """
    A class for a 3x3 convolutional layer with batch normalization and ReLU activation.
    """
    def __init__(self, in_channel: int, out_channel: int):
        """
        Initialize the 3x3 convolutional layer with batch normalization and ReLU activation.

        Args:
            in_channel (int): The number of input channels.
            out_channel (int): The number of output channels.
        """
        super().__init__()

        # Convolutional layer
        self.block = nn.Sequential(
            # 3x3 convolution with padding
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=1),
            # Batch normalization
            nn.BatchNorm2d(out_channel),
            # ReLU activation
            nn.ReLU()
        )
        
class CBA1x1(nn.Module):
    """
    A class for a 1x1 convolutional layer with batch normalization and ReLU activation.
    """
    def __init__(self, in_channel: int, out_channel: int):
        """
        Initialize the 1x1 convolutional layer with batch normalization and ReLU activation.

        Args:
            in_channel (int): The number of input channels.
            out_channel (int): The number of output channels.
        """
        super().__init__()

        # Convolutional layer
        self.block = nn.Sequential(
            # 1x1 convolution with padding
            nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, stride=1),
            # Batch normalization
            nn.BatchNorm2d(out_channel),
            # ReLU activation
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Forward pass through the layer
        return self.block(x)


class Aggregation(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        """
        Initialize the aggregation layer.

        The aggregation layer takes two feature maps as input and performs the following operations:
            1. Upsample the input feature map to match the resolution of the other feature map.
            2. Concatenate the two feature maps in the channel dimension.
            3. Reduce the number of channels in the concatenated feature map using a convolutional layer with a kernel size of 1x1.

        Args:
            in_channel (int): The number of channels in the input feature map.
            out_channel (int): The number of channels in the output feature map.
        """
        super().__init__()
        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample = nn.ConvTranspose2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=2, padding=1,output_padding=1,bias=False)
        self.conv1 = CBA1x1(in_channel=in_channel+out_channel, out_channel=out_channel)
        self.residual1 = BasicBlock(in_channel=out_channel, out_channel=out_channel)
        self.residual2 = BasicBlock(in_channel=out_channel, out_channel=out_channel)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the aggregation layer.

        Args:
            x1 (torch.Tensor): The low-resolution feature map.
            x2 (torch.Tensor): The high-resolution feature map.

        Returns:
            torch.Tensor: The aggregated feature map.
        """
        x1_up = self.upsample(x1)
        y = self.conv1(torch.cat([x1_up, x2], dim=1))
        y = self.residual1(y)
        y = self.residual2(y)
        return y
        
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        """
        Initialize the SEBlock module.

        Args:
            in_channels (int): The number of input channels.
            reduction_ratio (int): The reduction ratio for the number of channels in the SEBlock.
        """
        super().__init__()
        
        # Compute the average of the input feature map across spatial dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Reduce the number of channels to `num_channels // reduction_ratio`
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)
        
        # Increase the number of channels back to `num_channels`
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        
        # Apply the sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the SEBlock module.

    
        Args:
            x (torch.Tensor): The input feature map.

        Returns:
            torch.Tensor: The output feature map.
        """
        
        # Compute the average of the input feature map across spatial dimensions
        batch_size, num_channels, _, _ = x.size()
        avg_out = self.global_avg_pool(x).view(batch_size, num_channels)
        
        # Reduce the number of channels to `num_channels // reduction_ratio`
        attention = self.fc1(avg_out)
        
        # Increase the number of channels back to `num_channels`
        attention = F.relu(attention)
        attention = self.fc2(attention)
        
        # Apply the sigmoid activation function
        attention = self.sigmoid(attention).view(batch_size, num_channels, 1, 1)
        
        # Multiply the input feature map with the output of the sigmoid activation function, element-wise
        return x * attention

class BasicBlock(nn.Module):
    expansion = 1 

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):#downsample对应虚线残差结构
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x 
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # MLP: (C -> C/r -> C)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        # The kernel size is usually set to 7 for better performance
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Apply channel attention
        x_out = self.channel_attention(x) * x
        
        # Apply spatial attention
        x_out = self.spatial_attention(x_out) * x_out
        return x_out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # MLP: (C -> C/r -> C)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        # The kernel size is usually set to 7 for better performance
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Apply channel attention
        x_out = self.channel_attention(x) * x
        
        # Apply spatial attention
        x_out = self.spatial_attention(x_out) * x_out
        return x_out
    
def save_feature_maps_with_label(Y1, Y2, label, save_path, filename):
    """
    Process and save Y1[3], Y2[3], and label, automatically managing batch indices, with each batch in a separate folder.

    Parameters:
        Y1, Y2: feature map tensors (batch_size, channel, W, H)
        label: ground truth mask (batch_size, 1, 2W, 2H) - black and white image
        save_path: overall save path
    """
    def get_next_batch_idx(save_path):
        """Get the next batch index, i.e., the number of subfolders in the current folder"""
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            return 0
        existing_folders = [name for name in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, name))]
        return len(existing_folders)

    def save_feature_map(Y, file_path):
        """
        Process the feature map and save it as a grayscale image
        """
        batch_size, _, W, H = Y.shape
        Y_avg = Y.mean(dim=1)  # Average over the channel dimension (batch_size, W, H)

        # Normalize to [0, 255]
        Y_min, Y_max = Y_avg.min(), Y_avg.max()
        Y_norm = (Y_avg - Y_min) / (Y_max - Y_min + 1e-6)
        Y_vis = (Y_norm * 255).clamp(0, 255).byte()  # (batch_size, W, H)

        # Save only the first image in the batch (assuming batch_size >= 1)
        img = Image.fromarray(Y_vis[0].cpu().numpy(), mode='L')
        img.save(file_path)

    def save_label(label, file_path):
        """
        Process the label and save it as a black and white image
        :param label: Tensor, shape (batch_size, 2W, 2H)
        :param file_path: save path
        """
        # Save only the first image in the batch (assuming batch_size >= 1)
        label_vis = (label[0] * 255).clamp(0, 255).byte()
        img = Image.fromarray(label_vis.cpu().numpy(), mode='L')
        img.save(file_path)
        

    # Get batch index and create subfolder
    batch_folder = os.path.join(save_path, filename[0])
    os.makedirs(batch_folder, exist_ok=True)

    # Process and save Y1[3], Y2[3], and label
    save_feature_map(Y1[3], os.path.join(batch_folder, "Y1_3.png"))
    save_feature_map(Y2[3], os.path.join(batch_folder, "Y2_3.png"))
    save_label(label, os.path.join(batch_folder, "label.png"))

