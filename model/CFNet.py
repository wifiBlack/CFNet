import torch
import torch.nn as nn
from model.encoder import *
from model.content_decoder import *
from model.change_decoder import *
from model.focuser import *
from torch.amp import autocast

class CFNet(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        """
        Initialize the network.

        :param in_channel: The number of channels of the input tensor.
        :param out_channel: The number of channels of the output tensor.
        """
        super().__init__()
        self.channel_list = [in_channel,24,40,64,128]
        # Initialize the encoder
        self.encoder = Encoder(channel_list=self.channel_list)
        # Initialize the content decoders
        self.content_decoder_1 = ContentDecoder(channel_list=self.channel_list)
        self.content_decoder_2 = ContentDecoder(channel_list=self.channel_list)
        # Initialize the change decoder
        self.change_decoder = ChangeDecoder(channel_list=self.channel_list)
        # Initialize the focus network
        self.focuser = Focuser()
    
    def forward(self,x1: torch.Tensor, x2: torch.Tensor,device:torch.device,is_RM:bool=False,is_content:bool=False )-> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through the network.

        :param x1: The first input tensor.
        :param x2: The second input tensor.
        :return: A tuple containing the change map, the decoded feature maps of the first image, and the decoded feature maps of the second image.
        """
        with autocast(device.type+":"+str(device.index)):
            # Get the feature maps from the encoder
            Y1, Y2 = self.encoder(x1, x2)
            
            
            # Decode the feature maps of the first image
            Y1 = self.content_decoder_1(Y1)
            
            # Decode the feature maps of the second image
            Y2 = self.content_decoder_2(Y2)
            
            # folder_path = "content"
            
            # save_feature_maps_with_label(Y1,Y2,label,folder_path)
            
            # Get the focus maps from the focus network
            RM,Y1_change,Y2_change,Y1_unchange,Y2_unchange = self.focuser(Y1, Y2)
        
            # Get the change map from the change decoder
            y_change = self.change_decoder(Y1, Y2, RM)
            
            if is_RM:
                return RM
            
            if is_content:
                return Y1,Y2
            
            return y_change, Y1_change, Y2_change,Y1_unchange,Y2_unchange



