import torch.nn as nn
import torch
class ChangeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, y_change: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean squared error between the predicted change map and the target change map.

        Args:
            y_change: The predicted change map.
            target: The target change map.

        Returns:
            The mean squared error between the predicted change map and the target change map.
        """
        # Compute the mean squared error between the predicted change map and the target change map
        loss = nn.MSELoss()(y_change, target)

        return loss
