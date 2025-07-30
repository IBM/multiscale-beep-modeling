import logging
import unittest
from unittest.mock import patch

import polars as pl
import torch
from mblm import MBLMEncoderModelConfig
from mblm.model.config import MBLMModelConfig
from mblm.model.transformer import TransformerEncoderBlock
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from multiscale_beep_encoder.cli.run_inference import embedd_cycle
from multiscale_beep_encoder.data.preprocessing.beep_preprocessor import (
    update_tokenizer,
)
from multiscale_beep_encoder.utils.model_wrapper import WrapperMBLMEncoder

tokenizer_id: str = "ibm-granite/granite-3.3-8b-base"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@patch("polars.DataFrame.write_csv")
class TestInference(unittest.TestCase):
    """Test class for the inference (embeddings creation),"""

    test_column_id = "col_0"
    tokenizer: PreTrainedTokenizer
    model: WrapperMBLMEncoder

    @classmethod
    def setup_class(cls):
        """
        Dynamically compute MASKED_TOKEN_ID before tests run for TestBEEPDataset.
        The MASKED_TOKEN_ID is derived from the tokenizer
        """
        cls.tokenizer, _ = update_tokenizer(AutoTokenizer.from_pretrained(tokenizer_id))

        mblm_cfg = MBLMModelConfig(
            # Total number of tokens in the vocabulary. WARNING DON'T USE tokenizer.vocab_size as it isn't updated
            num_tokens=len(cls.tokenizer),
            pad_token_id=cls.tokenizer.pad_token_id,
            hidden_dims=[16, 8, 4],
            seq_lens=[2048, 8, 4],
            num_layers=[1, 1, 1],
            train_checkpoint_chunks=None,
            block=[
                TransformerEncoderBlock(
                    attn_head_dims=8,
                    attn_num_heads=2,
                    attn_use_rot_embs=True,
                    use_flash_attn=False,
                    pos_emb_type="fixed",
                ),
                TransformerEncoderBlock(
                    attn_head_dims=8,
                    attn_num_heads=1,
                    attn_use_rot_embs=True,
                    use_flash_attn=False,
                    pos_emb_type="fixed",
                ),
                TransformerEncoderBlock(
                    attn_head_dims=8,
                    attn_num_heads=1,
                    attn_use_rot_embs=True,
                    use_flash_attn=False,
                    pos_emb_type="fixed",
                ),
            ],
        )
        cfg = MBLMEncoderModelConfig(
            mask_token_id=cls.tokenizer.mask_token_id, mblm_config=mblm_cfg
        )
        cls.model = WrapperMBLMEncoder(cfg, is_embedder=True)

    def test_embedd_cycle(self, mock_write):  # noqa: ARG002
        """Tests that the basic embedd_cycle function runs without crashing."""
        stream = (
            pl.read_csv("tests/data/beep/preprocessed/b2c11.csv")
            .get_column("battery_stream")
            .to_list()[0]
        )

        embedd_cycle(
            stream,
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_function=lambda x: torch.mean(x, dim=0),
        )
        assert True
