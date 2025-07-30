import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd
import torch
from mblm.data.types import BatchMaskedForMLM
from pydantic import BaseModel
from torch.utils.data import ConcatDataset, Dataset
from transformers.models.auto.tokenization_auto import (
    AutoTokenizer,
)
from transformers.tokenization_utils import PreTrainedTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def create_attention_mask(
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    skip_attention_on: Literal["none", "special", "sep", "all"],
    sep_token: Optional[List[str]],
) -> torch.Tensor:
    """
    Creates a custom attention mask based on the specified strategy.

    Args:
        input_ids: The tokenized input IDs.
        tokenizer: The tokenizer instance.
        skip_attention_on: Strategy for masking ('none', 'special', 'sep', 'all', ).
                           'none': ignore no tokens.
                           'special': ignore all special tokens.
                           'sep': ignore both the sep token (if used) and the special tokens.
                           'all': ignore all tokens.
        sep_token: The separator token string used during concatenation, Each value will be joint using it,
                    If you have the following data 0.3, 0.02... you will get tokenised(0.3) sep_token tokenised(0.02).

    Returns:
        A boolean tensor where True indicates the token should be attended to,
        and False indicates it be ignored in the loss computation.
    """
    # The final attention mask should attend (True) only to tokens that are
    # 1. Not padding (handled by the tokenizer's default attention_mask)
    # 2. Not marked to be ignored by our custom logic using (`skip_attention_on` )

    mask_to_ignore = torch.zeros_like(input_ids, dtype=torch.bool)

    if skip_attention_on == "special" or skip_attention_on == "sep":
        special_token_ids = tokenizer.all_special_ids
        for token_id in special_token_ids:
            mask_to_ignore |= input_ids == token_id
    if skip_attention_on == "sep":
        if sep_token and len(sep_token) != 0:
            # Convert the separator token string to its corresponding ID
            # Note: Ensure the sep_token was actually added to the tokenizer vocabulary (the token ID is valid and not the unknown token ID)
            sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)
            # Get a list of sep tokens that are known
            sep_tokens = (
                [sep_t for sep_t in sep_token_id if sep_t != tokenizer.unk_token_id]
                if isinstance(sep_token_id, list)
                else [sep_token_id]
            )
            if len(sep_tokens) == 0 or sep_tokens[0] == tokenizer.unk_token_id:
                logger.warning(f"Separator token '{sep_token}' not found in tokenizer vocabulary.")
            for t in set(sep_tokens):
                mask_to_ignore |= input_ids == t
        else:
            logger.warning(
                "'skip_attention_on' is 'sep', but 'sep' token is None or empty. No tokens will be masked."
            )
    elif skip_attention_on == "all":
        # Mark all positions as True to ignore all tokens, returns ~ones
        mask_to_ignore = torch.ones_like(input_ids, dtype=torch.bool)
    elif skip_attention_on == "none":
        # Keep the mask as all False, meaning no tokens are specifically ignored by this logic
        pass  # mask_to_ignore remains all False
    elif skip_attention_on == "special":
        pass
    else:
        # Raise an error if an invalid strategy is provided
        raise ValueError(f"Invalid value for skip_attention_on: {skip_attention_on}")

    return ~mask_to_ignore


class BeepConfig(BaseModel):
    """
    Configuration for the beep dataset
    Params:
        data_dir: A path to the directory containings the pickle files.
        tokenizer_uri: str An subclass of PreTrainedTokenizer, you can implement your own
            strategy through it.
        masking_proba: (float) A masking proba, saying how much of the output should be masked
        batttery_stream: A identifier of the column in which we store the string representation of the device
        seq_len: (int) the sequence length that the dataset should produce, we should have
    """

    data_dir: str
    tokenizer_uri: str | None
    masking_proba: float
    battery_stream: str
    seq_len: int


def verify_tokenizer(tokeniser: PreTrainedTokenizer) -> bool:
    """Check that the tokeniser has all the necessary tokens set (pad_token_id, sep_token_id, unk_token_id, mask_token_id).
    If you used the preprocessing, an updated tokenized should have been saved
    args:
        - Tokeniser: the tokeniser that should have all the required tokens
    return:
        True if the tokenizer has all the tokens set, false otherwise"""

    return (
        tokeniser.pad_token_id is not None
        and tokeniser.sep_token_id is not None
        and tokeniser.unk_token_id is not None
        and tokeniser.mask_token_id is not None
    )


class BatteryHandler(Dataset):
    """A dataset storing one battery"""

    def __init__(
        self,
        data_path: str,
        stream_id: str = "battery_stream",
        sequence_len: int = 1024,
        time_id: str = "cycle_number",
    ):
        """Init the dataset, with a sequence length and the column id where the stream is"""
        self.sequence_length = sequence_len
        self.stream_id = stream_id
        self.data_path = data_path
        self.time_id = time_id

    @staticmethod
    @lru_cache(maxsize=45)
    def load_data(data_path: str):
        """Fill the data only once"""
        return pd.read_csv(data_path)

    def __getitem__(self, index) -> Tuple[str, str] | str:
        """Return a sequence from the data. Stream from each timestamp are joined together and sliced to the correct size"""
        data = "".join(BatteryHandler.load_data(self.data_path)[self.stream_id])[
            index * self.sequence_length : (index + 1) * self.sequence_length
        ]
        return data

    def get_item_and_meta(self, index: int) -> Dict[str, str | int]:
        """Return a row of the dataset as a dict"""
        cellkey_cyclenumber_batterystream = BatteryHandler.load_data(self.data_path).iloc[index]
        return cellkey_cyclenumber_batterystream.to_dict()

    def get_number_of_cycles(self) -> int:
        """In case we want to iterate over the cycles, this methods returns the number of cycle in the file (as the number of row)"""
        return BatteryHandler.load_data(self.data_path).shape[0]

    def __len__(self):
        """Returns the number of sequence available in the dataset"""
        str_len = len("".join(BatteryHandler.load_data(self.data_path)[self.stream_id]))
        return max(str_len // self.sequence_length, 1)


class BeepDataloader(Dataset):
    """This class hold the logic to serve an already tokenized beep dataset.
    Tokenized data should be stored in a pickle file as {<battery_name (str)>:  data (tokenizer_data)}
    the tokenizer_data should have the following keys:value
     - input_ids:torch.Tensor
     - attention_mask:torch.Tensor
    """

    TOKENISATION_SHRINKAGE_FACTOR: float = 1.09
    _tokenizer: PreTrainedTokenizer

    def __init__(
        self,
        config: BeepConfig,
        tokenizer: PreTrainedTokenizer | None = None,
        return_name: bool = False,
    ):
        """Initialise the Beep dataset using"""
        super().__init__()
        assert not (tokenizer is None and config.tokenizer_uri is None)
        if tokenizer is not None:
            assert config.tokenizer_uri is None, (
                "When a tokenizer is provided config.tokenizer_uri should be none"
            )
            logger.info(f"Using provided tokenizer, {config.tokenizer_uri} is not used")
            self._tokenizer = tokenizer
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_uri)
        self.return_name = return_name
        assert config.masking_proba >= 0.0 and config.masking_proba <= 1.0
        self._config = config
        assert verify_tokenizer(self._tokenizer), (
            "Your tokeniser doesn't have all the necessary tokens"
        )
        self._pad_token_id = self._tokenizer.pad_token_id
        self._sep_token_id = self._tokenizer.sep_token_id
        self._unk_token_id = self._tokenizer.unk_token_id
        self._mask_token_id = self._tokenizer.mask_token_id
        self._data = self._load_dataset(config.data_dir)

    def getitem_name(self, index):
        """Return the battery name associated with the data at index index"""
        data, name = self._data[index]
        return name

    def _load_dataset(self, dir: str) -> ConcatDataset[str] | ConcatDataset[Tuple[str, str]]:
        return ConcatDataset(
            datasets=[
                BatteryHandler(
                    data_path=filename.resolve().as_uri(),
                    stream_id=self._config.battery_stream,
                    sequence_len=int(self._config.seq_len * self.TOKENISATION_SHRINKAGE_FACTOR),
                )
                for filename in Path(dir).iterdir()
                if filename.name.endswith(".csv")
            ]
        )

    def __len__(self):
        """Return the len of the dataset, i.e the number of sequence on which we can learn"""
        return len(self._data)

    def get_vocab_size(self):
        """Get the size of the vocab, should be used to create in your model to create an embedding layer big enough and a projection to the vocab size"""
        return len(self._tokenizer)

    @staticmethod
    def apply_mlm_masking(
        sample_window: torch.Tensor,
        initial_candidate_mask: torch.Tensor,
        masking_proba: float,
        mask_token_id: int,
        vocab_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies MLM masking strategy (80% [MASK], 10% random, 10% original) to the sample window.

        Args:
            sample_window: The original token sequence (potentially truncated).
            initial_candidate_mask: A boolean mask indicating which tokens are candidates for MLM
                                     (e.g., not special tokens). True for candidates.

        Returns:
            A tuple containing:
                - tokens_masked: The token sequence with MLM applied.
                - mlm_loss_mask: A boolean mask indicating which tokens are part of the MLM objective (where loss should be computed).
        """
        # Determine which tokens (from the candidates) will actually be part of the MLM objective
        one_indices = torch.nonzero(initial_candidate_mask).view(-1)  # Indices of candidate tokens
        num_ones = int(one_indices.shape[0])  # Number of candidate tokens
        logger.debug(f"Number of MLM candidates: {num_ones=}")
        num_to_mask_for_mlm = int(round(num_ones * masking_proba))
        logger.debug(f"Number of tokens to select for MLM objective: {num_to_mask_for_mlm=}")

        tokens_masked = sample_window.clone()
        # This mask indicates which tokens will contribute to the MLM loss
        mlm_loss_mask = torch.zeros_like(initial_candidate_mask, dtype=torch.bool)

        if num_ones > 0 and num_to_mask_for_mlm > 0:
            logger.debug("Applying MLM masking")
            perm = torch.randperm(num_ones)
            # These are the specific token indices (from sample_window) that will be part of the MLM objective
            indices_for_mlm_objective = one_indices[perm[:num_to_mask_for_mlm]]

            len_indices_for_mlm_objective = indices_for_mlm_objective.shape[0]

            # 80% of selected tokens are replaced with [MASK]
            idx_mask_token = indices_for_mlm_objective[: int(len_indices_for_mlm_objective * 0.8)]
            tokens_masked[idx_mask_token] = mask_token_id
            mlm_loss_mask[idx_mask_token] = True

            # 10% of selected tokens are replaced with a random token
            idx_random_token = indices_for_mlm_objective[
                int(len_indices_for_mlm_objective * 0.8) : int(len_indices_for_mlm_objective * 0.9)
            ]
            tokens_masked[idx_random_token] = torch.randint(
                high=vocab_size,
                size=(len(idx_random_token),),
                device=tokens_masked.device,
            )
            mlm_loss_mask[idx_random_token] = True

            # 10% of selected tokens remain original (but are still predicted)
            idx_original_token = indices_for_mlm_objective[
                int(len_indices_for_mlm_objective * 0.9) :
            ]
            # tokens_masked at idx_original_token remain unchanged from sample_window
            mlm_loss_mask[idx_original_token] = True
        elif num_to_mask_for_mlm == 0:
            logger.debug(
                "No tokens selected for MLM objective based on masking probability or candidate count."
            )
            # mlm_loss_mask remains all False, tokens_masked is a clone of sample_window

        return tokens_masked, mlm_loss_mask

    @staticmethod
    def pad_tensors(
        tokens_masked: torch.Tensor,
        mlm_loss_mask: torch.Tensor,
        original_tokens: torch.Tensor,
        seq_len: int,
        padding_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pads the input tensors to the configured sequence length.

        Args:
            tokens_masked: The token sequence with MLM applied.
            mlm_loss_mask: The boolean mask for MLM loss.
            original_tokens: The original token sequence (labels).

        Returns:
            A tuple containing the padded versions of:
                - tokens_masked
                - mlm_loss_mask
                - original_tokens (labels)
        """
        current_len = original_tokens.size(-1)
        target_len = seq_len

        if current_len >= target_len:
            # No padding needed if current_len is already target_len (or greater, though truncation should prevent >)
            return tokens_masked, mlm_loss_mask, original_tokens

        num_to_pad = target_len - current_len
        # Ensure pad_token_id is available (checked in __init__)
        pad_values = padding_token_id * torch.ones(
            num_to_pad, dtype=original_tokens.dtype, device=original_tokens.device
        )

        logger.debug(f"Padding {num_to_pad} tokens. Pad tensor size: {pad_values.size()}")

        tokens_masked_padded = torch.concat((tokens_masked, pad_values))

        # Padding for loss mask should be False (or 0), as padded tokens don't contribute to loss
        pad_mask = torch.zeros(num_to_pad, dtype=mlm_loss_mask.dtype, device=mlm_loss_mask.device)
        mlm_loss_mask_padded = torch.concat((mlm_loss_mask, pad_mask))

        original_tokens_padded = torch.concat((original_tokens, pad_values))

        return tokens_masked_padded, mlm_loss_mask_padded, original_tokens_padded

    def __getitem__(self, index: int) -> BatchMaskedForMLM:
        """
        Retrieves a single data sample, applies MLM, and prepares it for the model.
        A sample consists of masked input tokens, a loss mask, and original labels.
        """
        seq = self._data[index]

        tokenized_output: Dict = self._tokenizer(
            seq, return_tensors="pt", return_attention_mask=False
        )
        del seq

        # Truncate tokenized input_ids to the configured sequence length.
        original_tokens_truncated = tokenized_output["input_ids"][0][: self._config.seq_len]
        logger.debug(f"Original tokens (truncated to seq_len): {original_tokens_truncated.shape=}")

        initial_mlm_candidate_mask = create_attention_mask(
            original_tokens_truncated,
            self._tokenizer,
            skip_attention_on="sep",
            sep_token=self._tokenizer.sep_token,
        )
        logger.debug(f"Initial MLM candidate mask: {initial_mlm_candidate_mask.shape=}")

        masked_token, attention_window = BeepDataloader.apply_mlm_masking(
            sample_window=original_tokens_truncated,
            initial_candidate_mask=initial_mlm_candidate_mask,
            masking_proba=self._config.masking_proba,
            mask_token_id=self._tokenizer.mask_token_id,
            vocab_size=len(self._tokenizer),
        )
        logger.debug(f"Tokens for model (unpadded): {masked_token.shape=}")
        logger.debug(f"MLM loss mask (unpadded): {attention_window.shape=}")

        final_model_input, final_mlm_loss_mask, final_labels = BeepDataloader.pad_tensors(
            masked_token,
            attention_window,
            original_tokens_truncated,
            seq_len=self._config.seq_len,
            padding_token_id=self._tokenizer.pad_token_id,
        )

        # Ensure correct dtypes for the BatchMaskedForMLM tuple
        return final_model_input.long(), final_mlm_loss_mask.bool(), final_labels.long()
