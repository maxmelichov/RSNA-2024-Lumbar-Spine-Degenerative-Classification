from typing import Any
import torch.nn as nn
import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.nn.modules.loss import _Loss


class CFG():
    pass

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class WeightedLosses(nn.Module):
    def __init__(self, losses, weights):
        super().__init__()
        self.losses = losses
        self.weights = weights

    def forward(self, *input: Any, **kwargs: Any):
        cum_loss = 0
        for loss, w in zip(self.losses, self.weights):
            cum_loss += w * loss.forward(*input, **kwargs)
        return cum_loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


class FocalLossWithWeights(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLossWithWeights, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=torch.tensor([1.0, 2.0, 4.0]).to(device))
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()
    


# Custom Loss for this competition

class SevereLoss(_Loss):
    """
    For RSNA 2024
    criterion = SevereLoss()     # you can replace nn.CrossEntropyLoss
    loss = criterion(y_pred, y)
    """
    def __init__(self, temperature=1.0):
        """
        Use max if temperature = 0
        """
        super().__init__()
        self.t = temperature
        assert self.t >= 0
    
    def __repr__(self):
        return 'SevereLoss(t=%.1f)' % self.t

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
          y_pred (Tensor[float]): logit             (batch_size, 3, 25)
          y      (Tensor[int]):   true label index  (batch_size, 25)
        """
        assert y_pred.size(0) == y.size(0)
        assert y_pred.size(1) == 3 and y_pred.size(2) == 25
        assert y.size(1) == 25
        assert y.size(0) > 0
        
        slices = [slice(0, 5), slice(5, 15), slice(15, 25)] 
        w = 2 ** y  # sample_weight w = (1, 2, 4) for y = 0, 1, 2 (batch_size, 25)

        loss = F.cross_entropy(y_pred, y, reduction='none')  # (batch_size, 25)

        # Weighted sum of losses for spinal (:5), foraminal (5:15), and subarticular (15:25)
        wloss_sums = []
        for k, idx in enumerate(slices):
            wloss_sums.append((w[:, idx] * loss[:, idx]).sum())

        # any_severe_spinal
        #   True label y_max:      Is any of 5 spinal severe? true/false
        #   Prediction y_pred_max: Max of 5 spinal severe probabilities y_pred[:, 2, :5].max(dim=1)
        #   any_severe_spinal_loss is the binary cross entropy between these two.
        y_spinal_prob = y_pred[:, :, :5].softmax(dim=1)             # (batch_size, 3,  5)
        w_max = torch.amax(w[:, :5], dim=1)                         # (batch_size, )
        #y_max = torch.amax(y[:, :5] == 2, dim=1).to(torch.float32)  # 0 or 1
        y_max = torch.amax(y[:, :5] == 2, dim=1).to(y_pred.dtype)   # 0 or 1

        if self.t > 0:
            # Attention for the maximum value
            attn = F.softmax(y_spinal_prob[:, 2, :] / self.t, dim=1)  # (batch_size, 5)

            # Pick the sofmax among 5 severe=2 y_spinal_probs with attn
            y_pred_max = (attn * y_spinal_prob[:, 2, :]).sum(dim=1)   # weighted average among 5 spinal columns 
        else:
            # Exact max; this works too
            y_pred_max = y_spinal_prob[:, 2, :].amax(dim=1)

        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        loss_max = criterion(y_pred_max, y_max, )
        wloss_sums.append((w_max * loss_max).sum())

        # See below about these numbers
        loss = (wloss_sums[0] / 6.084050632911392 +
                wloss_sums[1] / 12.962531645569621 + 
                wloss_sums[2] / 14.38632911392405 +
                wloss_sums[3] / 1.729113924050633) / (4 * y.size(0))

        return loss