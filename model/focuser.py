from typing import List
import torch.nn as nn
import torch

class Focuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def forward(self, X1: List[torch.Tensor], X2: List[torch.Tensor], mode: str = "change") -> List[torch.Tensor]:
        RM = []
        for x1, x2 in zip(X1, X2):
            cosin_distance = 1 - self.cos_sim(x1, x2)
            if mode == "change":
                RM.append(torch.tanh(cosin_distance))
            elif mode == "unchange":
                RM.append(1 - torch.tanh(cosin_distance))
            
        Y1_change = []
        Y2_change = []
        Y1_unchange = []
        Y2_unchange = []
        # Apply the focus to the decoded feature maps
        for i in range(len(RM)):
            Y1_change.append(torch.mul(RM[i].unsqueeze(1), X1[i]))
            Y2_change.append(torch.mul(RM[i].unsqueeze(1), X2[i])) 
    
        for i in range(len(RM)):
            Y1_unchange.append(torch.mul(1-RM[i].unsqueeze(1), X1[i]))
            Y2_unchange.append(torch.mul(1-RM[i].unsqueeze(1), X2[i]))
            
        return RM,Y1_change,Y2_change,Y1_unchange,Y2_unchange
