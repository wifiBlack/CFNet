import torch.nn as nn
import torch

class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cossim = nn.CosineSimilarity(dim=1)

    def forward(self, x1:torch.Tensor, x2:torch.Tensor,reduction:int = 1,mode:str = "unchange")->torch.Tensor:
        batch_size,num_channels,W,H = x1.size()
        x1 = x1.view(batch_size,num_channels,-1)
        x2 = x2.view(batch_size,num_channels,-1)
        n = int(W/reduction)
        rand_int1 = torch.randint(0,W*H,(n,))
        rand_int2 = torch.randint(0,W*H,(n,))
        
        D1 = torch.zeros(batch_size,n)
        D2 = torch.zeros(batch_size,n)

        for i in range(n):
            D1[:,i] = self.cossim(x1[:,:,rand_int1[i]],x1[:,:,rand_int2[i]])
            D2[:,i] = self.cossim(x2[:,:,rand_int1[i]],x2[:,:,rand_int2[i]])
        if mode=="unchange":
            return torch.mean(torch.abs(D1-D2))
        elif mode=="change":
            return 1-torch.mean(torch.abs(D1-D2))