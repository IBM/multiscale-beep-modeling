import logging
from typing import Callable, Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.tokenization_utils import PreTrainedTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class NumberTokenLoss(nn.Module):
    """Class for NTL."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        squash: Optional[float] = None,
        use_logit_weight: bool = False,
        device: str = "cuda",
        **kwargs,  # noqa: ARG002
    ):
        """
        NTL constructor.

        Args:
            tokenizer: PreTrainedTokenizer that tokenize each digit independandly
            squash: The optional squashing factor for the NTL.
            use_logit_weight: Whether to scale the NTL using the logit weight on number tokens. Defaults to False.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.squash = squash
        self.use_logit_weight = use_logit_weight
        self.create_ntl_dep(tokenizer=tokenizer, device=device)

        if squash is not None:
            # We are squashing the absolute differences to the GT token to a given range
            # The squashing factor indicates by what factor the farthest number token
            # is worse than the closest, incorrect number token. If this value is not
            # set, with 10 NTs, we have a squashing factor of 9
            num_ids = torch.nonzero(self.is_number_token, as_tuple=True)[0]
            vocab_to_num = torch.full((len(tokenizer),), -1, dtype=torch.long)
            vocab_to_num[num_ids] = torch.arange(num_ids.size(0), dtype=torch.long)

            # Build NxN abs-diff matrix
            vals = self.number_values_dense.unsqueeze(0)  # (1 x N)
            diff = torch.abs(vals - vals.t())  # (N x N)

            # Mask out zeros → find smallest nonzero diff
            inf = torch.finfo(diff.dtype).max
            diff_nonzero = diff.masked_fill(diff == 0, inf)
            global_min_nz = diff_nonzero.min()
            global_max = diff.max()

            scale = (squash - 1) / (global_max - global_min_nz)
            lookup = 1 + (diff - global_min_nz) * scale
            lookup[diff == 0] = 0.0

            self.register_buffer("vocab_to_num", vocab_to_num)
            self.register_buffer("squash_lookup", lookup)
            logger.info(f"Set up squashing for NTL with factor {squash}")

    def __call__(self, *args, **kwargs):
        """Alias to self.forward"""
        return self.forward(*args, **kwargs)

    def create_ntl_dep(self, tokenizer: PreTrainedTokenizer, device: str = "cuda") -> None:
        """Assign the values to self.is_number_token and  self.number_values_dense based on the tokenizer"""
        self.number_values = torch.full((len(tokenizer),), float("nan"))
        ids = tokenizer("".join(str(x) for x in range(10)), return_attention_mask=False)[
            "input_ids"
        ]
        assert isinstance(ids, list)
        for value, at_idx in enumerate(ids):
            if not (-1 <= value <= 9):
                continue
            self.number_values[at_idx] = value
        self.is_number_token = ~torch.isnan(self.number_values)
        self.number_values_dense = self.number_values[self.is_number_token]
        self.number_values = self.number_values.to(device=device)
        self.is_number_token = self.is_number_token.to(device=device)
        self.number_values_dense = self.number_values_dense.to(device=device)

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        loss_mask: Optional[Tensor] = None,
        reduction: str = "mean",
    ) -> Tensor:
        """
        Computes the NTL.

        Args:
            logits: Tensor of shape BS x T x V
            labels: Tensor of shape BS x T
            loss_mask: Optional tensor of BS x T
            ce_loss: Optional tensor of BS x T with previously computed CE for logits.
            reduction: Optional string specifying the reduction to apply to the
                output. Defaults to "mean", options are "mean", "sum", "none".

        Returns:
            Loss tensor
                0-D if reduction=="mean"|"sum"
                BS x T if reduction=="none"

        """

        if logits.numel() == 0:
            raise ValueError("Logits passed to the NumberTokenLoss are empty!")
        if labels.numel() == 0:
            raise ValueError("Labels passed to the NumberTokenLoss are empty!")

        labels = labels.masked_fill(labels == -100, 0)

        # Create a mask to filter out non-digit tokens
        y = self.number_values[labels]
        valid_positions = ~torch.isnan(y)

        # If no digit tokens in batch, no need for upcoming calculations
        if torch.count_nonzero(valid_positions) == 0:
            if (reduction == "mean") | (reduction == "mean"):
                loss = torch.tensor(0, dtype=labels.dtype, device=labels.device)
            elif reduction == "none":
                loss = torch.zeros_like(valid_positions)
            else:
                raise ValueError(f"{reduction} is not a valid value for reduction")

            return loss

        # apply softmax and get number labels
        bs, seq_len, _ = logits.size()
        nt_logits = logits[:, :, self.is_number_token]
        softmax_probs = F.softmax(nt_logits, dim=-1)

        # compute absolute difference between the true numbers and all possible number values
        if self.squash is None:
            abs_diff = torch.abs(y[valid_positions].unsqueeze(-1) - self.number_values_dense)
        else:
            abs_diff = self.squash_lookup[self.vocab_to_num[labels[valid_positions]]]  # type:ignore

        # loss is the absolute difference weighted by the softmax probs
        loss = (abs_diff * softmax_probs[valid_positions]).sum(axis=-1)  # type:ignore

        # If use_logit_weight: compute weights for NTL based on logits
        if self.use_logit_weight:
            softmax_probs_all = F.softmax(logits, dim=-1)
            ntl_weight = torch.sum(softmax_probs_all[:, :, self.is_number_token], dim=-1)[
                valid_positions
            ]

            # Apply weights for NTL element-wise
            loss *= ntl_weight

        # Apply the loss_mask to lower importance of number tokens before the final answer
        label_mask = (
            loss_mask[valid_positions].to(dtype=loss.dtype)
            if loss_mask is not None
            else torch.ones_like(loss)
        )

        if reduction == "mean":
            # Mean pooling (weighted by loss mask)
            loss = torch.dot(loss.flatten(), label_mask.flatten()) / torch.count_nonzero(label_mask)
        elif reduction == "sum":
            loss = torch.dot(loss.flatten(), label_mask.flatten())
        elif reduction == "none":
            # Cast loss for number tokens back to Tensor of size BS x T
            loss_ = torch.zeros(valid_positions.view(-1).size()).to(loss.device)
            loss_[valid_positions.view(-1)] = loss * label_mask
            loss = loss_.view(bs, seq_len)

            assert torch.sum(loss[~valid_positions]) == 0, (
                "NumberTokenLoss computed for non-digit tokens!"
            )

        else:
            raise ValueError(f"{reduction} is not a valid value for reduction")

        return loss


class NumberTokenLossMSE(NumberTokenLoss):
    """Class for NTL-MSE. Inherits from NumberTokenLoss."""

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        loss_mask: Optional[Tensor] = None,
        reduction: str = "mean",
        loss_function: Callable = F.mse_loss,
    ) -> Tensor:
        """
           Computes the NTL-MSE.
        self,
           logits: Tensor,
           labels: Tensor,
           loss_mask: Optional[Tensor] = None,
           ce_loss: Optional[Tensor] = None,
           reduction: str = "mean",
           Args:
               logits: Tensor of shape BS x T x V
               labels: Tensor of shape BS x T
               loss_mask: Optional tensor of BS x T
               reduction: Optional string specifying the reduction to apply to the
                   output. Defaults to "mean", options are "mean", "sum", "none".
               loss_function: Callable function specifying loss function to be
                   used. Defaults to torch.nn.functional.mse_loss

           Returns:
               Loss tensor
                   0-D if reduction=="mean"|"sum"
                   BS x T if reduction=="none"
        """
        if logits.numel() == 0:
            raise ValueError("Logits passed to the NumberTokenLossMSE are empty!")
        if labels.numel() == 0:
            raise ValueError("Labels passed to the NumberTokenLossMSE are empty!")

        labels = labels.masked_fill(labels == -100, 0)

        # Create a mask to filter out non-digit tokens
        y = self.number_values[labels]
        valid_positions = ~torch.isnan(y)

        # If no digit tokens in batch, no need for upcoming calculations
        if torch.count_nonzero(valid_positions) == 0:
            if (reduction == "mean") | (reduction == "mean"):
                loss = torch.tensor(0, dtype=labels.dtype, device=labels.device)
            elif reduction == "none":
                loss = torch.zeros_like(valid_positions)
            else:
                raise ValueError(f"{reduction} is not a valid value for reduction")

            return loss

        # apply softmax and get number labels
        bs, seq_len, _ = logits.size()
        nt_logits = logits[:, :, self.is_number_token]
        softmax_probs = F.softmax(nt_logits, dim=-1)

        # compute the weighted average of number tokens
        yhat = torch.sum(softmax_probs[valid_positions] * self.number_values_dense, dim=-1)

        # Apply specified loss function to y and yhat
        loss = loss_function(yhat, y[valid_positions], reduction="none")

        # If use_logit_weight: compute weights for NTL based on logits
        if self.use_logit_weight:
            softmax_probs_all = F.softmax(logits, dim=-1)
            ntl_weight = torch.sum(softmax_probs_all[:, :, self.is_number_token], dim=-1)[
                valid_positions
            ]

            # Apply weights for NTL element-wise
            loss *= ntl_weight

        # Apply the loss_mask to lower importance of number tokens before the final answer
        label_mask = (
            loss_mask[valid_positions].to(dtype=loss.dtype)
            if loss_mask is not None
            else torch.ones_like(loss)
        )

        if reduction == "mean":
            # Mean pooling (weighted by loss mask)
            loss = torch.dot(loss.flatten(), label_mask.flatten()) / torch.count_nonzero(label_mask)
        elif reduction == "sum":
            loss = torch.dot(loss.flatten(), label_mask.flatten())
        elif reduction == "none":
            # Cast loss for number tokens back to Tensor of size BS x T
            loss_ = torch.zeros(valid_positions.view(-1).size()).to(loss.device)
            loss_[valid_positions.view(-1)] = loss * label_mask
            loss = loss_.view(bs, seq_len)

            assert torch.sum(loss[~valid_positions]) == 0, (
                "NumberTokenLossMSE computed for non-digit tokens!"
            )

        else:
            raise ValueError(f"{reduction} is not a valid value for reduction")

        return loss


LOSS_FACTORY: Dict[str, Type[NumberTokenLoss]] = {
    "ntl": NumberTokenLoss,
    "ntl-mse": NumberTokenLossMSE,
}
