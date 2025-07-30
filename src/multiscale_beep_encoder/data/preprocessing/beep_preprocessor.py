import logging
from typing import List, Optional, Sequence, Tuple

import polars as pl
from transformers.models.auto.tokenization_auto import (
    AutoTokenizer,
)
from transformers.tokenization_utils import PreTrainedTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def smooth(
    df: pl.LazyFrame,
    column_target: str = "charge_capacity",
    column_id: str = "cell_key",
) -> pl.DataFrame:
    """
    Smooths the target column within a Polars DataFrame, grouped by an identifier column.

    The smoothing process involves applying a cumulative maximum to the target
    column for each group, after clipping its values between 0 and 1.1.
    This is typically used for features like 'charge_capacity' to ensure
    they are monotonically non-decreasing up to a certain point within each group.
    **! This function assumes that the data is sorted such that the cumulative maximum works**
    For charge_capacity, You should ensure the data is sorted decreasing by time

    Args:
        df: The input Polars DataFrame.
        column_target: The name of the column to be smoothed.
                       Defaults to "charge_capacity".
        column_id: The name of the column used to group the DataFrame.
                   Operations are performed independently for each unique value
                   in this column. Defaults to "cell_key".

    Returns:
        A new Polars DataFrame with the target column smoothed for each group.
    """
    dfs = []
    for device in df.select(pl.col(column_id).unique()).collect().get_column(column_id).unique():
        df_dev = (
            df.filter(pl.col(column_id) == device)
            .with_columns(pl.col(column_target).clip(0, 1.1).cum_max())
            .collect()
        )
        df = df.filter((pl.col(column_id) == device).not_())
        dfs.append(df_dev)
    res: pl.DataFrame = pl.concat(dfs)
    return res


def update_tokenizer(
    tokenizer: PreTrainedTokenizer, time_sep: Optional[str] = "<|sep|>", output_dir: str | None = ""
) -> Tuple[PreTrainedTokenizer, str]:
    """Make sure that the tokenizer has all the necessary values, if the tokenizer is updated, it is saved to output_dir"""
    is_tokenizer_modified = False
    if tokenizer.sep_token is None:
        if time_sep is None or time_sep == "":
            time_sep = ""
            tokenizer.add_special_tokens({"sep_token": "<|sep|>"})
        elif time_sep is not None and time_sep != "":
            tokenizer.add_special_tokens({"sep_token": time_sep})
        else:
            # Needed for safe join operations
            time_sep = ""
    else:
        time_sep = (
            tokenizer.sep_token[0] if isinstance(tokenizer.sep_token, list) else tokenizer.sep_token
        )
    if time_sep is None:
        time_sep = ""
    unk_token = tokenizer.unk_token
    if unk_token is None:
        is_tokenizer_modified = True
        unk_token = "<|UNK|>"
        tokenizer.add_special_tokens({"unk_token": unk_token})
    pad_token = tokenizer.pad_token
    if pad_token is None:
        is_tokenizer_modified = True
        pad_token = "<|PAD|>"
        tokenizer.add_special_tokens({"pad_token": pad_token})
    mask_token = tokenizer.mask_token
    if mask_token is None:
        is_tokenizer_modified = True
        mask_token = "<|MASK|>"
        tokenizer.add_special_tokens({"mask_token": mask_token})

    assert tokenizer.unk_token is not None and tokenizer.pad_token is not None
    if is_tokenizer_modified:
        if output_dir is not None and output_dir != "":
            logger.warning(
                f"Tokenizer modifed, load it from {output_dir}/tokenizer for the rest of the pipeline"
            )
            tokenizer.save_pretrained(f"{output_dir}/tokenizer")
        else:
            logger.warning(
                f"Tokenizer modifed but not saved due to having output_dir set to '{output_dir}'"
            )
    return tokenizer, time_sep


def preprocess(
    path: str,
    tokenizer_id: str,
    column_id: str,
    time_sep: Optional[str],
    column_time: List[str],
    output_dir: str | None = None,
    value_sep: str = ",",
    smooth_charge_capacity: bool = True,
    exclude_features: List[str] = [],
    decimal_precision: int = 4,
    time_cutoff: Optional[Sequence[int]] = None,
    is_training_shuffled: bool = True,
) -> Tuple[pl.DataFrame, PreTrainedTokenizer]:
    """
    Preprocesses time-series data from a CSV file for language model consumption.

    Args:
        path: Path to the input CSV file, a glob pattern (data/dir/*.csv) can also be used assuming all the csv have the same columns names.
        tokenizer_id: Identifier for the Hugging Face tokenizer to be loaded
                      (e.g., "bert-base-uncased").
        column_id: Name of the column that identifies unique entities/time-series
                   (e.g., "battery_name").
        time_sep: Token used to separate different time steps (rows) in the
                  final string representation. If None or empty, no separator
                  is used between time steps. If the tokenizer lacks a SEP token,
                  a default or this token will be added.
        column_time: Sequence of column names to sort the data by,
                     representing the time dimension. They should be ordered from bigger timespan to smaller once, like [`day`, `hour`, `minute`], or [`Year`, `Month`], etc.
        output_dir: Directory where the modified tokenizer (if any) will be saved.
        value_sep: Token used to separate values within a single time step (row)
                   when creating the cross-section string. Defaults to ",".
        smooth_charge_capacity: Boolean flag indicating if charge capacity smoothing
                                should be applied. Defaults to True.
        exclude_features: Indicate which features should not be added to the stream (for example metadata such as the final battery life)
        decimal_precision: Define how many decimal your float should have. If decimal_precision is 1, 0.313... will get mapped to 0.3, if it is set to 2, you will get 0.31 and so on
        time_cutoff: define if you want to process all the battery or only recordings before the cutoff
        is_training_shuffled: Defines if we shuffle cross section from a battery. If the raw data is CS11:CS12....CS1N for the first battery and is_training_shuffled is set to true we will randomly get
        CSj1:CSl2....CScN where j,l,c are different cross section identifier belonging to battery 1.

    Returns:
        The resulting dataframe with 2 columns:column_id and battery_stream, the first has a unique identifiers from `column_id` and
        the second column contains the corresponding processed data streams as strings with the different separators (between value and between time steps)
    """
    if isinstance(column_time, str):
        column_time = [column_time]
    assert len(column_time) > 0
    if time_cutoff is not None:
        if len(time_cutoff) != len(column_time):
            logger.warning(
                f"You don't have the same number of value for the cuttoff and the time. All values in {column_time[len(time_cutoff) :]} will be accepted"
            )
    logger.info(f"Loading tokenizer {tokenizer_id}")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer, time_sep = update_tokenizer(
        tokenizer=tokenizer, time_sep=time_sep, output_dir=output_dir
    )
    logger.info(f"Reading data from {path}")
    # read_csv accepts a glob pattern: *.cvs for example
    dataset_df = pl.read_csv(path)
    battery_list = dataset_df.select(pl.col(column_id).unique()).get_column(column_id).to_list()
    logger.info(f"Loaded {len(battery_list)} batteries")
    # If we want to select a slice of the data we need to have a column
    # Indicating where the time is, in that case we filter
    if time_cutoff is not None:
        logger.info("Selecting everything before cutoff")
        dataset_df = dataset_df.filter(
            pl.col(col) <= cutoff for col, cutoff in zip(column_time, time_cutoff)
        )
    logger.debug(f"Data size: {dataset_df.shape}")

    if smooth_charge_capacity:
        dataset_df = dataset_df.sort(pl.col(column_time), descending=True, maintain_order=False)
        dataset_df = smooth(
            df=dataset_df.lazy(),
            column_target="charge_capacity",
            column_id=column_id,
        )
    dataset_df = dataset_df.sort(pl.col(column_time), descending=False, maintain_order=False)

    dataset_df_lazy: pl.LazyFrame = dataset_df.lazy()
    unk_token = tokenizer.unk_token

    def format_with(precision: int = 4):
        def format_element(el):
            if isinstance(el, float):
                # Banker's rounding, 2.5 -> 2, 3.5 -> 4, 4.5 -> 4
                return f"{round(el, precision)}"
            if el is None:
                return unk_token
            return str(el)

        return format_element

    logger.info("Formatting dataset")
    formatted_df = dataset_df_lazy.select(
        pl.all().map_elements(
            format_with(decimal_precision), skip_nulls=False, return_dtype=pl.Utf8
        )
    )
    logger.info("Merging to string")
    out_df: pl.DataFrame = pl.DataFrame()
    if is_training_shuffled:
        aggregated_df = (
            formatted_df.select([
                pl.col(column_id),
                # Concatenate all feature columns into a single string for the cross-section
                pl.concat_str(
                    pl.all().exclude(exclude_features).exclude(column_id), separator=value_sep
                ).alias("formated_cross_section"),
            ])
            # Group by the battery identifier
            .group_by([pl.col(column_id)])
            # Aggregate the cross-sections into a list for each battery
            .agg(pl.col("formated_cross_section"))
        ).collect()

        max_len = aggregated_df.select(pl.col("formated_cross_section").list.len()).max().item()

        # Create the order
        shuffle_index = pl.arange(0, max_len, eager=True).sample(
            fraction=1.0, shuffle=True, seed=24
        )

        out_df = (
            aggregated_df.lazy().select([
                pl.col(column_id),
                pl.col("formated_cross_section")
                # Apply the same shuffle_index to every list.
                .list.gather(shuffle_index, null_on_oob=True)
                # Shorter lists will have nulls for out-of-bounds indices; remove them.
                .list.drop_nulls()
                # Join the consistently shuffled list into a single stream string.
                .list.join(separator=time_sep)
                .alias("battery_stream"),
            ])
        ).collect()
    else:
        formatted_df = (
            formatted_df.select(
                [
                    pl.col(column_id),
                    pl.concat_str(
                        pl.all().exclude(exclude_features).exclude(column_id), separator=value_sep
                    ).alias("formated_cross_section"),
                ]
                + column_time  # type: ignore
            )
            .group_by([pl.col(column_id), pl.col(column_time[0])])
            .agg(pl.col("formated_cross_section"))
            .select([
                pl.all().exclude("formated_cross_section"),
                pl.col("formated_cross_section")
                .list.sample(fraction=1.0, shuffle=True)
                .list.join(separator=time_sep)
                .alias("battery_stream"),
            ])
        )
        # Cast to float to get a numeric sort not an alphabetic one (1,11,12...2,21,....)
        out_df = formatted_df.collect().sort(by=pl.col(column_time[0]).cast(pl.Float32))

    if output_dir and output_dir != "":
        logger.info(f"Writing resulting csv files to {output_dir}")
        for battery_name, data in out_df.group_by(column_id):
            data.write_csv(f"{output_dir}/{battery_name[0]}.csv")

    return out_df, tokenizer
