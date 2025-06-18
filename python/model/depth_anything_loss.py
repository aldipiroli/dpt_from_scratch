import torch.nn as nn


class AffineInvariantDepthLoss(nn.Module):
    def __init__(self):
        super(AffineInvariantDepthLoss, self).__init__()

    def forward(self, pred, gt):
        pred = pred.reshape(-1, 3)
        gt = gt.reshape(-1, 3)
        loss_fn = nn.MSELoss()
        loss = loss_fn(pred, gt)
        return loss
