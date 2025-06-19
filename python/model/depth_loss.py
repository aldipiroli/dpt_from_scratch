import torch
import torch.nn as nn


class AffineInvariantDepthLoss(nn.Module):
    def __init__(self):
        super(AffineInvariantDepthLoss, self).__init__()

    def forward(self, pred, gt):
        pred_norm = self.normalize(pred)
        gt_norm = self.normalize(gt)

        loss = torch.mean(torch.abs(pred_norm - gt_norm))
        return loss

    def normalize(self, x):
        b, h, w = x.shape
        assert h > 0 and w > 0
        translation = torch.median(x)
        scale = (1 / (h * w)) * torch.sum(torch.abs(x - translation)) + 1e-6

        x_norm = (x - translation) / scale
        return x_norm
