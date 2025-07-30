from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml
from einops import rearrange
from mblm.model.config import MBLMEncoderModelConfig, MBLMReturnType
from mblm.model.mblm import MBLMEncoder
from torch import nn

from multiscale_beep_encoder.loss.ntl import NumberTokenLoss


@dataclass
class MLMDataCollator:
    """
    Data collator for Masked Language Modeling.
    It stacks features and ensures the output dictionary keys match
    what WrapperMBLMEncoder via HuggingFace Trainer expects.
    """

    def __call__(
        self, features: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Dict[str, Any]:
        """Map from the dataset output to the WrapperMBLMEncoder input"""
        is_label_present = len(features[0]) == 3
        input_ids = torch.stack([feature[0] for feature in features])
        attention_mask = torch.stack([feature[1] for feature in features]).bool()
        labels = None
        if is_label_present:
            labels = torch.stack([feature[2] for feature in features])
        return {
            k: v
            for k, v in zip(
                ["input_ids", "attention_mask", "labels"], [input_ids, attention_mask, labels]
            )
            # Includes the labels key, value if present
            if v is not None
        }


class WrapperMBLMEncoder(nn.Module):
    """Wrap the MBLMEncoder model for hugging face usage"""

    def __init__(self, encoder_config: MBLMEncoderModelConfig, is_embedder: bool = False):
        """Initialize the WrapperMBLMEncoder"""
        super().__init__()
        self.cfg = encoder_config
        self.encoder = MBLMEncoder(encoder_config)
        self.is_embedder = is_embedder

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass of the model.
        args:
            input_ids: The input tensor of shape (batch_size, sequence_length)
            attention_mask: The attention mask tensor of shape (batch_size, sequence_length)
            labels: The labels tensor of shape (batch_size, sequence_length)
        """
        if self.is_embedder:
            assert attention_mask is None and labels is None
            return self.get_embeddings(input_ids)
        if labels is not None:
            loss_logit = self.encoder.forward(
                input_ids, attention_mask, labels, return_type=MBLMReturnType.LOSS_LOGITS
            )
            return {"loss": loss_logit[0], "logits": loss_logit[1]}
        logits = self.encoder(input_ids, attention_mask, return_type=MBLMReturnType.LOGITS)
        return {"logits": logits}

    def get_embeddings(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns the hidden state for the given input (embeddings)
        args:
            x: The input tensor that will be embedded, shape (batch_size, sequence_length), it contains the token_id
        returns:
            The Embeddings tensor of shape (batch_size, sequence_length, hidden_size)
        """
        hidden_state = self.encoder.forward(x, return_type=MBLMReturnType.HIDDEN_STATE)
        if isinstance(hidden_state, torch.Tensor):
            return {"hidden_state": hidden_state}
        raise ValueError("Unexpected return type from encoder")

    @staticmethod
    def load_cfg(path_to_mblm_encoder_model_config: str, is_embedder: bool = False):
        """Load a configuration created with save_cfg
        args:
            path_to_MBLMEncoderModelConfig: A path to a yaml configuration
        returns:
            A WrapperMBLMEncoder ready for training or ready to load weight and use
        """
        with Path(path_to_mblm_encoder_model_config).open("rt") as f:
            cfg = yaml.safe_load(f)
            cfg_obj = MBLMEncoderModelConfig.validate(cfg)
            return WrapperMBLMEncoder(cfg_obj, is_embedder=is_embedder)

    def save_cfg(self, output_file: str) -> None:
        """Save the config to a yaml file, ready to be loaded using the load_cfg static method"""
        with Path(output_file).open("w") as f:
            yaml.dump(self.cfg.model_dump(), f)


class NTLMBLMEncoder(nn.Module):
    """A WrapperMBLMEncoder that compute the loss as a weighted average of ntl and CE"""

    def __init__(
        self,
        encoder_config: MBLMEncoderModelConfig,
        ntl: NumberTokenLoss,
        ntl_weight: float = 0.66,
        is_embedder: bool = False,
    ):
        """Create the NTLMBLMEncoder"""
        super().__init__()
        self.cfg = encoder_config
        self.encoder = MBLMEncoder(encoder_config)
        self.is_embedder = is_embedder
        self.loss_fn = ntl
        self.ntl_weight = ntl_weight

    def get_embeddings(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns the hidden state for the given input (embeddings)
        args:
            x: The input tensor that will be embedded, shape (batch_size, sequence_length), it contains the token_id
        returns:
            The Embeddings tensor of shape (batch_size, sequence_length, hidden_size)
        """
        hidden_state = self.encoder.forward(x, return_type=MBLMReturnType.HIDDEN_STATE)
        if isinstance(hidden_state, torch.Tensor):
            return {"hidden_state": hidden_state}
        raise ValueError("Unexpected return type from encoder")

    @staticmethod
    def load_cfg(path_to_mblm_encoder_model_config: str, is_embedder: bool = False):
        """Load a configuration created with save_cfg
        args:
            path_to_MBLMEncoderModelConfig: A path to a yaml configuration
        returns:
            A WrapperMBLMEncoder ready for training or ready to load weight and use
        """
        with Path(path_to_mblm_encoder_model_config).open("rt") as f:
            cfg = yaml.safe_load(f)
            cfg_obj = MBLMEncoderModelConfig.validate(cfg)
            return WrapperMBLMEncoder(cfg_obj, is_embedder=is_embedder)

    def save_cfg(self, output_file: str) -> None:
        """Save the config to a yaml file, ready to be loaded using the load_cfg static method"""
        with Path(output_file).open("w") as f:
            yaml.dump(self.cfg.model_dump(), f)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass of the model.
        args:
            input_ids: The input tensor of shape (batch_size, sequence_length)
            attention_mask: The attention mask tensor of shape (batch_size, sequence_length)
            labels: The labels tensor of shape (batch_size, sequence_length)
        """
        if self.is_embedder or labels is None:
            assert attention_mask is None and labels is None
            return self.get_embeddings(input_ids)

        assert attention_mask is not None
        logits = self.encoder.forward(
            input_ids, attention_mask, labels, return_type=MBLMReturnType.LOGITS
        )
        loss_ntl = self.loss_fn(logits=logits, labels=labels, loss_mask=attention_mask)
        # CROSS ENTROPY
        labels[~attention_mask] = self.cfg.mask_token_id
        logits = rearrange(logits, "b s v -> b v s")
        # target is Batch, Seq_len
        assert isinstance(logits, torch.Tensor), "Logits should be a tensor"
        loss_ce = torch.nn.functional.cross_entropy(
            input=logits, target=labels, ignore_index=self.cfg.mask_token_id
        )

        return {
            "loss": loss_ntl * self.ntl_weight + loss_ce * (1 - self.ntl_weight),
            "logits": logits,
        }
