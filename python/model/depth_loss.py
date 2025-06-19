import torch.nn as nn


class AffineInvariantDepthLoss(nn.Module):
    def __init__(self):
        super(AffineInvariantDepthLoss, self).__init__()

    def forward(self, pred, gt):
        # gt = torch.clamp(gt, min=1e-2)
        # disp = 1.0 / gt
        # disp = self.normalize(disp)
        # pred = self.normalize(pred)
        loss_fn = nn.MSELoss()
        return loss_fn(pred, gt)

    def normalize(self, x):
        b, h, w = x.shape
        x_flat = x.view(b, -1)

        translation = x_flat.mean(dim=1, keepdim=True)
        scale = (x_flat - translation).abs().mean(dim=1, keepdim=True) + 1e-6

        x_norm = (x_flat - translation) / scale
        return x_norm.reshape(-1)
