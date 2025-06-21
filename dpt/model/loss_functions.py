import torch.nn as nn


class AffineInvariantDepthLoss(nn.Module):
    def __init__(self):
        super(AffineInvariantDepthLoss, self).__init__()

    def forward(self, pred, gt):
        if len(pred.shape) == 4:
            pred = pred.squeeze(1)
        gt = self.normalize(gt)
        pred = self.normalize(pred)
        loss_fn = nn.SmoothL1Loss()
        return loss_fn(pred, gt)

    def normalize(self, x):
        b, h, w = x.shape
        x_flat = x.view(b, -1)

        translation = x_flat.mean(dim=1, keepdim=True)
        scale = (x_flat - translation).abs().mean(dim=1, keepdim=True) + 1e-6

        x_norm = (x_flat - translation) / scale
        return x_norm.reshape(-1)


class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()

    def forward(self, pred, gt):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, gt)
        return loss
