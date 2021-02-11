import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LmCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=None, reduction='batchmean') -> None:
        super(LmCrossEntropyLoss, self).__init__()
        assert reduction in ['none', 'batchmean', 'sum', 'mean']
        self.reduction = reduction
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = self.compute_loss(input, target)
        return self._reduce(loss)

    def compute_loss(self, input: Tensor, target: Tensor) -> Tensor:
        batch_size, _, num_embeddings = input.shape
        loss = self.criterion(
            input.contiguous().view(-1, num_embeddings),
            target.contiguous().view(-1)
        ).view(batch_size, -1)
        return loss

    def _reduce(self, loss: Tensor) -> Tensor:
        if self.reduction == 'batchmean':
            return loss.sum(dim=1).mean(dim=0)
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'mean':
            return loss.mean()
        return loss