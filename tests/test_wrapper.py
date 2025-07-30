import logging

import numpy as np
import pytest
import torch
from mblm.model.config import MBLMEncoderModelConfig, MBLMModelConfig
from mblm.model.transformer import TransformerEncoderBlock

from multiscale_beep_encoder.utils.model_wrapper import WrapperMBLMEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TestWrapperMBLMEncoder:
    """Test the WrapperMBLMEncoder model"""

    @pytest.fixture
    def model(self, num_stages=3, seed=42):
        """Fixture to  generate models with num_layer stages"""
        torch.manual_seed(seed)
        num_token = int(torch.randint(0, 2**10, size=(1,)).item())
        mblm_cfg = MBLMModelConfig(
            # Total number of tokens in the vocabulary. WARNING DON'T USE tokenizer.vocab_size as it isn't updated
            num_tokens=num_token,
            pad_token_id=num_token - 1,
            hidden_dims=[int(torch.randint(1, 2**10, size=(1,)).item())] * num_stages,
            seq_lens=[int(torch.randint(4, 10, size=(1,)).item())] * num_stages,
            num_layers=[1] * num_stages,
            train_checkpoint_chunks=None,
            block=[
                TransformerEncoderBlock(
                    attn_head_dims=8,
                    attn_num_heads=2,
                    attn_use_rot_embs=True,
                    use_flash_attn=False,
                    pos_emb_type="fixed",
                )
            ]
            * num_stages,
        )
        logger.info(mblm_cfg)
        return WrapperMBLMEncoder(
            encoder_config=MBLMEncoderModelConfig(mask_token_id=0, mblm_config=mblm_cfg)
        )

    def test_save_load_cfg(self, model: WrapperMBLMEncoder, tmp_path_factory):
        """Check that saving the model config and loading it into a model yields the same config"""
        fn = tmp_path_factory.mktemp("data") / "config.yaml"
        model.save_cfg(fn)

        new_model = WrapperMBLMEncoder.load_cfg(fn)
        assert new_model.cfg == model.cfg

    @pytest.mark.parametrize("batch,seq_len_divisor", [(1, 1), (2, 2), (3, 3), (9, 4)])
    def test_embeddings(self, model: WrapperMBLMEncoder, batch: int, seq_len_divisor: int):
        """Test shape of the embeddings based on a random model config"""
        seq_len_max = int(np.prod(model.cfg.mblm_config.seq_lens))
        seq_len = max(seq_len_max // seq_len_divisor, 2)
        input = torch.randint(0, model.cfg.mblm_config.num_tokens, size=(batch, seq_len))
        embeddings = model.get_embeddings(input)
        assert embeddings["hidden_state"].size(0) == batch
        assert embeddings["hidden_state"].size(1) == seq_len
        assert embeddings["hidden_state"].size(-1) == model.cfg.mblm_config.hidden_dims[-1]
        assert embeddings["hidden_state"].ndim == 3
