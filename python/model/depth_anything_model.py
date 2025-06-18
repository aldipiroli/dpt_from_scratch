import torch
import torch.nn as nn


class DepthAnythingModel(nn.Module):
    def __init__(
        self,
        model_cfg,
    ):
        super(DepthAnythingModel, self).__init__()
        self.model_cfg = model_cfg
        self.ln = nn.Linear(3, 1)

    def forward(self, x):
        x = self.ln(x)
        return x


if __name__ == "__main__":
    x = torch.rand(1, 640, 480, 3)
    model = DepthAnythingModel(model_cfg={})
    preds = model(x)
