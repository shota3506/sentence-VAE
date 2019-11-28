import torch
import torch.nn as nn


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, tgt, mask):
        batch_size, max_sequence_length, _ = x.shape
        tgt = tgt[:, :max_sequence_length]
        mask = mask[:, :max_sequence_length]
        x = x.view(batch_size * max_sequence_length, -1)
        tgt = tgt.flatten()
        mask = mask.flatten()

        loss = self.loss(x, tgt)
        loss = loss.masked_select(mask).sum()
        return loss / batch_size


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, mean, logvar):
        batch_size, _ = mean.shape
        loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return loss / batch_size
