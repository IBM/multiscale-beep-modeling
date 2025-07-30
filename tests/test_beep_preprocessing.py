import unittest
from unittest.mock import Mock, patch

import polars as pl
import pytest
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from multiscale_beep_encoder.data.preprocessing.beep_preprocessor import (
    preprocess,
    update_tokenizer,
)

tokenizer_id: str = "ibm-granite/granite-3.3-8b-base"
csv_data = "tests/data/beep/raw/example_raw_data.csv"
sep = "<|sep|>"


@patch("polars.DataFrame.write_csv")
class TestPreprocessing(unittest.TestCase):
    """Test class for the preprocessing, write is mocked"""

    test_column_id = "col_0"

    def test_preprocess(self, mock_polars_write_csv: Mock):
        """Tests that the basic preprocessing function runs without crashing."""
        try:
            preprocess(
                path=csv_data,
                tokenizer_id=tokenizer_id,
                column_id=self.test_column_id,
                time_sep=sep,
                column_time=["cycle_number"],
                smooth_charge_capacity=False,
                output_dir="to_delete",
                value_sep=",",
            )
            assert mock_polars_write_csv.call_count == 2
        except Exception as e:
            self.fail(f"An error occurred: {e}")

    @patch("polars.read_csv")
    def test_preprocess_dummy_row(
        self,
        mock_read_csv,
        mock_polars_write_csv: Mock,
    ):
        """Tests preprocessing with a mocked DataFrame to ensure correct handling of unique IDs and output file writing."""
        mock_read_csv.return_value = pl.DataFrame({
            "cycle_number": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
            self.test_column_id: ["a1", "a1", "a1", "a1", "b2", "b2", "b2", "b2", "c3", "c3"],
            "charge_policy_Q1": [0.16] * 10,
            "charge_policy_Q2": [0.2, 0.1] * 5,
            "charge_policy_Q3": [1] * 10,
            "charge_policy_Q4": [0.99] * 10,
            "voltage": [0.1] * 9 + [1],
            "charge_capacity": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "current": [0.5] * 10,
            "cycle_life": [370, 1025, 126, 370, 1025, 126, 370, 1025, 126, 4],
        })
        res, _ = preprocess(
            path=csv_data,
            tokenizer_id=tokenizer_id,
            column_id=self.test_column_id,
            time_sep="",
            column_time=["cycle_number"],
            output_dir="to_delete",
            value_sep="",
        )
        unique_ids = res.get_column(self.test_column_id).unique().to_list()
        assert len(unique_ids) == 3
        assert mock_polars_write_csv.call_count == len(unique_ids)

    @patch("polars.read_csv")
    def test_preprocess_with_null(
        self,
        mock_read_csv: Mock,
        mock_polars_write_csv: Mock,
    ):
        """Tests preprocessing with a mocked DataFrame containing null values to ensure robust handling."""
        mock_read_csv.return_value = pl.DataFrame({
            "cycle_number": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
            self.test_column_id: ["a1", "a1", "a1", "a1", "b2", "b2", "b2", "b2", "c3", "c3"],
            "charge_policy_Q1": [0.16] * 10,
            "charge_policy_Q2": [0.2, 0.1] * 5,
            "charge_policy_Q3": [1] * 5 + [None] * 5,
            "charge_policy_Q4": [0.99] * 10,
            "voltage": [0.1] * 9 + [None],
            "charge_capacity": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "current": [0.5] * 10,
            "cycle_life": [370, 1025, 126, 370, 1025, 126, 370, 1025, 126, 4],
        })
        res, _ = preprocess(
            path=csv_data,
            tokenizer_id=tokenizer_id,
            column_id=self.test_column_id,
            time_sep="",
            column_time=["cycle_number"],
            output_dir="to_delete",
            value_sep="",
        )
        unique_ids = res.get_column(self.test_column_id).unique().to_list()
        assert len(unique_ids) == 3
        assert set(unique_ids) == set(res.to_numpy()[:, 0])
        assert mock_polars_write_csv.call_count == len(unique_ids)


@patch("transformers.tokenization_utils_base.PreTrainedTokenizerBase.save_pretrained")
def test_update_tokenizer(mock_save_pretrained: Mock):
    """Tests that the tokenizer is updated with special tokens and saved correctly."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    t_sep = update_tokenizer(
        tokenizer,
    )
    updated_tokenizer: PreTrainedTokenizer = t_sep[0]
    assert updated_tokenizer.unk_token is not None
    assert updated_tokenizer.pad_token is not None
    assert updated_tokenizer.mask_token is not None
    assert updated_tokenizer.sep_token is not None
    mock_save_pretrained.assert_not_called()


@patch("transformers.tokenization_utils_base.PreTrainedTokenizerBase.save_pretrained")
@pytest.mark.parametrize("id", [tokenizer_id])
def test_update_tokenizer_different_path(mock_save_pretrained: Mock, id):
    """Tests that the tokenizer update and save process works correctly even when using a parameterized tokenizer ID."""
    tokenizer = AutoTokenizer.from_pretrained(id)
    t_sep = update_tokenizer(
        tokenizer,
    )
    updated_tokenizer: PreTrainedTokenizer = t_sep[0]
    assert updated_tokenizer.unk_token is not None
    assert updated_tokenizer.pad_token is not None
    assert updated_tokenizer.mask_token is not None
    assert updated_tokenizer.sep_token is not None
    mock_save_pretrained.assert_not_called()


@patch("transformers.tokenization_utils_base.PreTrainedTokenizerBase.save_pretrained")
@pytest.mark.parametrize("id", [tokenizer_id])
def test_update_tokenizer_different_path_with_write(mock_save_pretrained: Mock, id):
    """Tests that the tokenizer update and save process works correctly even when using a parameterized tokenizer ID."""
    tokenizer = AutoTokenizer.from_pretrained(id)
    t_sep = update_tokenizer(tokenizer, output_dir="mocked_dir")
    updated_tokenizer: PreTrainedTokenizer = t_sep[0]
    assert updated_tokenizer.unk_token is not None
    assert updated_tokenizer.pad_token is not None
    assert updated_tokenizer.mask_token is not None
    assert updated_tokenizer.sep_token is not None
    mock_save_pretrained.assert_called_once()
