import torch.nn as nn
class DiceCoefficient(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, prediction, target):
        numerator = 2 * (prediction * target).sum()
        denominator = (prediction ** 2).sum() + (target ** 2).sum()
        return numerator / denominator.clamp(min=self.eps)