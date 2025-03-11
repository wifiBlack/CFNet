from typing import List
from loss.change_loss import *
from loss.content_loss import *
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self) :
        super().__init__()
        self.change_loss = ChangeLoss()
        self.content_loss = ContentLoss()
        
    """
    Compute the loss of the model. The loss is the sum of the change loss and
    the auxiliary loss. The change loss is the mean squared error between the
    predicted change map and the target change map. The auxiliary loss is the
    sum of the content loss between the predicted feature maps of the first
    image and the predicted feature maps of the second image at each scale.
    """
    def forward(self, y_change: torch.Tensor, Y1_change: List[torch.Tensor],
                Y2_change: List[torch.Tensor], Y1_unchange: List[torch.Tensor],
                Y2_unchange: List[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss of the model.

        Args:
            y_change: The predicted change map.
            Y1_change: The predicted feature maps of the first image at each scale.
            Y2_change: The predicted feature maps of the second image at each scale.
            Y1_unchange: The predicted feature maps of the first image at each scale.
            Y2_unchange: The predicted feature maps of the second image at each scale.
            target: The target change map.

        Returns:
            The loss of the model.
        """
        alpha = 1
        beta = 0.1
        main_loss = self.change_loss(y_change, target)
        change_content_loss = 0
        unchange_content_loss = 0
        n = len(Y1_change)
        for i in range(n):
            change_content_loss = change_content_loss + self.content_loss(Y1_change[i], Y2_change[i],mode="change")
            unchange_content_loss = unchange_content_loss + self.content_loss(Y1_unchange[i], Y2_unchange[i],mode="unchange")
        
        return alpha*main_loss + beta / n * (change_content_loss + unchange_content_loss)
