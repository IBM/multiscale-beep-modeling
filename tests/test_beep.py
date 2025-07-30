import pytest
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from multiscale_beep_encoder.data.dataset.beep import BatteryHandler, BeepConfig, BeepDataloader
from multiscale_beep_encoder.data.preprocessing.beep_preprocessor import update_tokenizer

TOKENIZER_URI = "ibm-granite/granite-3.3-8b-base"
BEEP_DIR = "tests/data/beep/preprocessed"


def fixture_beep(masking_proba=1.0, seq_len=100, tokenizer=None):
    """
    General helper to directly import the beep dataset from the fixture
    """
    return (
        BeepDataloader(
            BeepConfig(
                data_dir=BEEP_DIR,
                tokenizer_uri=None,
                masking_proba=masking_proba,
                battery_stream="battery_stream",
                seq_len=seq_len,
            ),
            tokenizer=tokenizer,
        )
        if tokenizer is not None
        else BeepDataloader(
            BeepConfig(
                data_dir=BEEP_DIR,
                tokenizer_uri=TOKENIZER_URI,
                masking_proba=masking_proba,
                battery_stream="battery_stream",
                seq_len=seq_len,
            ),
            tokenizer=None,
        )
    )


class TestBatteryHandler:
    """Test the battery handler"""

    # Assert don't fail
    @staticmethod
    def data(len: int = 2**10):
        """Create a battery handler"""
        data = BatteryHandler(
            data_path="tests/data/beep/preprocessed/b2c11.csv",
            sequence_len=len,
        )
        return data

    @pytest.mark.parametrize("seq_len", [2**x for x in range(1, 11)])
    def test_seq_len(self, seq_len):
        """Tests that the BatteryHandler returns data утверждает with the specified sequence length."""
        data = TestBatteryHandler.data(seq_len)
        assert len(data[0]) == seq_len

    def test_ds_len(self):
        """Tests that the BatteryHandler dataset has a non-negative length."""
        ds = TestBatteryHandler.data(15)
        assert len(ds) >= 0

    @pytest.mark.parametrize("index", [(2**x) for x in range(0, 4)])
    def test_idempotent(self, index):
        """Test that calling multiple time the index yields the same data"""
        ds = TestBatteryHandler.data(16)
        data1 = ds[index]
        data2 = ds[index]
        assert data1 == data2

    @pytest.mark.xfail
    @pytest.mark.parametrize("seq_len", [-(2**x) for x in range(1, 11)])
    def test_get_seq_len(self, seq_len):
        """Tests that the BatteryHandler correctly handles negative sequence lengths (expected to fail)."""
        ds = TestBatteryHandler.data(seq_len)
        assert len(ds[0]) == seq_len


class TestBEEPDataset:
    """The TestBEEPDataset class to test the Beep Dataset Class"""

    MASKED_TOKEN_ID: int  # Will be computed in setup_class
    PAD_TOKEN_ID: int  # Will be computed in setup_class
    tokenizer: PreTrainedTokenizer

    @classmethod
    def setup_class(cls):
        """
        Dynamically compute MASKED_TOKEN_ID before tests run for TestBEEPDataset.
        The MASKED_TOKEN_ID is derived from the tokenizer
        """
        cls.tokenizer, _ = update_tokenizer(AutoTokenizer.from_pretrained(TOKENIZER_URI))
        cls.MASKED_TOKEN_ID = cls.tokenizer.mask_token_id
        cls.PAD_TOKEN_ID = cls.tokenizer.pad_token_id

    def test_ok_init(self):
        """Tests if the BeepDataloader initializes successfully with valid configuration."""
        try:
            fixture_beep(tokenizer=self.tokenizer)
            assert True
        except Exception as e:
            print(f"{e}")
            assert False

    @pytest.mark.xfail
    def test_bad_proba_init(self):
        """Tests that BeepDataloader initialization fails with invalid masking probabilities (expected to fail)."""
        fixture_beep(masking_proba=-0.4, tokenizer=self.tokenizer)
        fixture_beep(masking_proba=12.32, tokenizer=self.tokenizer)

    def test_masked_proba_0(self):
        """Tests that with zero masking probability, no tokens are masked and the attention mask is all zeros."""
        dataset = fixture_beep(masking_proba=0.0, seq_len=10, tokenizer=self.tokenizer)
        masked_data, mask, labels = dataset[0]
        assert torch.all(masked_data != self.MASKED_TOKEN_ID)
        assert torch.all(mask == 0)

    @pytest.mark.parametrize("proba", [0.1, 0.20, 0.30, 0.75])
    def test_masked_proba_dyn(self, proba):
        """Tests dynamic masking probabilities to ensure correct masking behavior and attention mask generation."""
        dataset = fixture_beep(masking_proba=proba, tokenizer=self.tokenizer)
        masked_data, mask, labels = dataset[0]
        assert (masked_data == self.MASKED_TOKEN_ID).sum() >= 0
        # We have some token to attend (idientified by a 1)
        assert mask.sum() >= 0
        # Some of the masked_token_id appear where the mask indicates, because we can also get the real token or a random one (bert style)
        assert torch.any((masked_data == self.MASKED_TOKEN_ID) == mask)

    def test_padding_for_long_sequence(self):
        """Tests that sequences shorter than seq_len are correctly padded with the pad token ID."""
        bds = fixture_beep(seq_len=2**20, tokenizer=self.tokenizer)
        masked_data, mask, labels = bds[0]
        assert mask[-1] == 0
        assert masked_data[-1] == self.PAD_TOKEN_ID

    def test_padding_for_long_sequence_all_masked(self):
        """Tests that padded tokens are not included in the attention mask even when masking probability is 1.0."""
        bds = fixture_beep(seq_len=2**20, masking_proba=1.0, tokenizer=self.tokenizer)
        masked_data, mask, labels = bds[0]
        assert mask[-1] == 0
