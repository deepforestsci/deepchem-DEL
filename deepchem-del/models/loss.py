import torch
import torch.nn.functional as F
from typing import Callable, Optional
from deepchem.models.losses import Loss


class MultiLabelLoss(Loss):
    """
    """

    def __init__(self,
                 epsilon: float = 1e-7,
                 macro: bool = False,
                 gamma: float = 2.0,
                 nan_mask: int = 2):
        super().__init__()
        self.epsilon = epsilon
        self.macro = macro
        self.gamma = gamma
        self.nan_mask = nan_mask

    def _create_pytorch_loss(
            self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:

        def loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            # y_true: (B, T), values in {0,1,nan_mask}
            # y_pred: (B, T), probs or logits depending on from_logits
            y_true_int = y_true.long()
            y_pred_prob = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)

            # mask missing labels
            mask = (y_true_int != self.nan_mask).float()  # (B, T)
            y_true_bin = (y_true_int == 1).float() * mask

            # dynamic alpha
            if self.macro:
                # freq per task and class -> (T, 2)
                freq0 = ((y_true_int == 0) & (mask.bool())).sum(dim=0).float()
                freq1 = ((y_true_int == 1) & (mask.bool())).sum(dim=0).float()
                freq = torch.stack([freq0, freq1], dim=-1)  # (T,2)

                alpha_cls = torch.where(freq == 0.0, torch.zeros_like(freq), torch.rsqrt(freq))
                denom = (alpha_cls * freq).sum(dim=1, keepdim=True).clamp_min(self.epsilon)
                numer = freq.sum(dim=1, keepdim=True)
                alpha_cls = alpha_cls * numer / denom  # (T,2)

                # select alpha by true class per element
                alpha = alpha_cls[:, 0].unsqueeze(0) * (1.0 - y_true_bin) + \
                        alpha_cls[:, 1].unsqueeze(0) * y_true_bin  # (B,T)
            else:
                freq0 = ((y_true_int == 0) & (mask.bool())).sum().float()
                freq1 = ((y_true_int == 1) & (mask.bool())).sum().float()
                freq = torch.stack([freq0, freq1], dim=0)  # (2,)

                alpha_cls = torch.where(freq == 0.0, torch.zeros_like(freq), torch.rsqrt(freq))
                denom = (alpha_cls * freq).sum().clamp_min(self.epsilon)
                numer = freq.sum()
                alpha_cls = alpha_cls * numer / denom  # (2,)

                alpha = alpha_cls[0] * (1.0 - y_true_bin) + alpha_cls[1] * y_true_bin  # (B,T)

            pt = y_true_bin * y_pred_prob + (1.0 - y_true_bin) * (1.0 - y_pred_prob)
            elem = -alpha * ((1.0 - pt) ** self.gamma) * torch.log(pt) * mask  # (B,T)

            if self.macro:
                num = elem.sum(dim=0)  # (T,)
                den = (alpha * mask).sum(dim=0).clamp_min(self.epsilon)  # (T,)
                out = (num / den).mean()  # scalar
                return out.unsqueeze(0).unsqueeze(0)
            else:
                num = elem.sum(dim=1)  # (B,)
                den = (alpha * mask).sum(dim=1).clamp_min(self.epsilon)  # (B,)
                out = num / den  # (B,)
                return out.unsqueeze(-1)  # (B,1)

        return loss


class CategoricalLoss(Loss):
    """
    """

    def __init__(self,
                 epsilon: float,
                 mask: int,
                 vocab_size: int,
                 gamma: float = 2.0):
        super().__init__()
        self.epsilon = epsilon
        self.mask = mask
        self.vocab_size = vocab_size
        self.gamma = gamma


    def _create_pytorch_loss(
            self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:

        def loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            # unpack
            unmasked = y_true[:, :, 1].long()      # (B,L)
            y_true_masked = y_true[:, :, 0].long() # (B,L)

            # flatten
            target = y_true_masked.reshape(-1)             # (B*L,)
            pred = y_pred.reshape(-1, y_pred.shape[-1])    # (B*L,V)

            # keep only masked positions
            keep = (target != self.mask)
            target = target[keep]   # (N,)
            pred = pred[keep]       # (N,V)

            if target.numel() == 0:
                # no masked tokens in batch
                return torch.zeros((1, 1), device=y_pred.device, dtype=y_pred.dtype)

            # alpha from full unmasked stream
            freq = torch.bincount(unmasked.reshape(-1), minlength=self.vocab_size).float()
            freq = freq.to(y_pred.device)
            # set [PAD],[MASK] frequencies to zero
            if self.vocab_size >= 2:
                freq[:2] = 0.0
            alpha = torch.where(freq == 0.0, torch.zeros_like(freq), torch.rsqrt(freq))  # (V,)

            # probs
            pred_prob = torch.clamp(pred, self.epsilon, 1.0 - self.epsilon)

            # one-hot targets
            one_hot = F.one_hot(target, num_classes=self.vocab_size).float()  # (N,V)

            pt = one_hot * pred_prob + (1.0 - one_hot) * (1.0 - pred_prob)
            elem = -alpha.unsqueeze(0) * ((1.0 - pt) ** self.gamma) * (one_hot * torch.log(pred_prob))

            num = elem.sum()
            den = (alpha.unsqueeze(0) * one_hot).sum().clamp_min(self.epsilon)
            out = num / den  # scalar

            return out.unsqueeze(0).unsqueeze(0)  # (1,1)

        return loss


class BinaryLoss(Loss):
    """
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def _create_pytorch_loss(
            self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:

        def loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            y_true = y_true.float().reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)
            p = torch.clamp(y_pred, 1e-7, 1.0 - 1e-7)
            pt = y_true * p + (1.0 - y_true) * (1.0 - p)
            focal = -((1.0 - pt) ** self.gamma) * torch.log(pt)  # alpha=None, reduction=mean
            out = focal.mean()

            return out.unsqueeze(0).unsqueeze(0)  # (1,1)

        return loss
