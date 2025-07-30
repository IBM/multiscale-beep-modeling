import json
import logging
import random
from pathlib import Path
from typing import Dict, List

import polars as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _copy_files_from_config(
    input_dir_str: str, output_dir_str: str, config: Dict[str, List[str]]
) -> Dict[str, List[pl.DataFrame]]:
    """
    Copies files from input_dir to subdirectories of output_dir (train, validation, test)
    based on the provided configuration. CSV files are copied using Polars, other
    file types are copied using shutil.

    Args:
        input_dir_str: Path to the input directory.
        output_dir_str: Path to the main output directory.
        config: Dictionary with "train", "validation", "test" keys,
                each containing a list of filenames.

    Returns:
        A dictionary where keys are split names ("train", "validation", "test")
        and values are lists of full Path objects to the CSV files
        that were successfully copied, for potential merging.
    """
    input_path = Path(input_dir_str)
    output_path = Path(output_dir_str)
    copied_csv: Dict[str, List[pl.DataFrame]] = {"train": [], "validation": [], "test": []}
    files_processed_count = 0

    for split_type in ["train", "validation", "test"]:
        filenames = config.get(split_type, [])
        if len(filenames) == 0:
            logger.info(f"No files found for split type '{split_type}'")
            continue

        target_subdir = output_path / split_type
        target_subdir.mkdir(parents=True, exist_ok=True)

        for filename in filenames:
            src_file = input_path / filename
            dst_file = target_subdir / filename

            if not src_file.exists():
                logger.warning(
                    f"File '{filename}' listed in config not found in '{input_path}', skipping."
                )
                continue

            try:
                logger.info(f"Copying CSV: {src_file.as_posix()} -> {dst_file.as_posix()}")
                df = pl.read_csv(src_file)
                df.write_csv(dst_file)
                copied_csv[split_type].append(df)
                files_processed_count += 1
            except Exception as e:
                logger.error(f"Error processing '{src_file}' to '{dst_file}': {e}")

    logger.info(f"Successfully processed/copied {files_processed_count} files based on config.")
    return copied_csv


def _generate_split_config_by_size(
    input_dir_str: str, output_dir_str: str, ratio_train: float, ratio_validation: float
) -> Dict[str, List[str]]:
    """
    Generates a split configuration for '*.csv' files from input_dir into train,
    validation, and test sets based on file sizes. Saves this configuration to
    output_dir/split_cfg.json.
    """
    input_path = Path(input_dir_str)
    output_path = Path(output_dir_str)
    output_json_path = output_path / "split_cfg.json"

    files_with_sizes = [(p.name, p.stat().st_size) for p in input_path.glob("*.csv") if p.is_file()]
    total_csv_size = sum(size for _, size in files_with_sizes)
    split_config: Dict[str, List[str]] = {"train": [], "validation": [], "test": []}
    if not files_with_sizes or total_csv_size == 0:
        info = f"No '*.csv' files found in '{input_path}' to generate split config."
        logger.info(info)
        raise ValueError(info)

    random.shuffle(files_with_sizes)
    train_target_size = total_csv_size * ratio_train
    val_target_size = total_csv_size * ratio_validation
    current_train_size, current_val_size = 0, 0

    for filename, size in files_with_sizes:
        if current_train_size < train_target_size:
            split_config["train"].append(filename)
            current_train_size += size
        elif current_val_size < val_target_size:
            split_config["validation"].append(filename)
            current_val_size += size
        else:
            split_config["test"].append(filename)
    logger.info(
        f"CSV files per split in generated config: Train: {len(split_config['train'])}, "
        f"Validation: {len(split_config['validation'])}, Test: {len(split_config['test'])}."
    )

    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(split_config, f, indent=4)
    logger.info(f"Saved split configuration for '*.csv' files to '{output_json_path}'.")
    return split_config


def _load_existing_config(config_path_str: str) -> Dict[str, List[str]] | None:
    """Loads an existing JSON configuration file."""
    config_path = Path(config_path_str)
    if not config_path.is_file():
        logger.warning(f"Provided JSON config path is not a file: '{config_path_str}'.")
        return None
    try:
        with config_path.open("r", encoding="utf-8") as f:  # Indent Level 2
            config = json.load(f)
        logger.info(f"Successfully loaded JSON configuration from: '{config_path_str}'")
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from '{config_path_str}': {e}.")
    except Exception as e:
        logger.error(f"An unexpected error occurred reading '{config_path_str}': {e}.")
    return None


def organize_files(
    input_dir: str,
    output_dir: str,
    existing_json_config_path: str | None = None,
    ratio_train: float = 0.8,
    ratio_validation: float = 0.05,
    merge_files: bool = False,
):
    """
    Organizes files from input_dir into train, validation, and test sets in output_dir.

    If existing_json_config_path is provided, uses it to copy files.
    Otherwise, splits only '*.csv' files by size, saves the new configuration,
    and then copies CSV files based on this new configuration.

    Args:
        input_dir: Path to the input directory.
        output_dir: Path to the output directory.
        existing_json_config_path: Optional path to an existing JSON config.
        ratio_train: Proportion for training set (if generating CSV split).
        ratio_validation: Proportion for validation set (if generating CSV split).
        merge_files: If True, merges copied CSV files within each split.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        logger.error(f"Input directory not found: '{input_dir}'")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    for subdir_name in ["train", "validation", "test"]:
        (output_path / subdir_name).mkdir(parents=True, exist_ok=True)

    config_to_use = None
    if existing_json_config_path:
        config_to_use = _load_existing_config(existing_json_config_path)

    if config_to_use is None:
        logger.info("Generating new split configuration for CSV files based on size.")
        config_to_use = _generate_split_config_by_size(
            input_dir, output_dir, ratio_train, ratio_validation
        )
    else:
        logger.error("No valid configuration found or generated. Aborting file organization.")
        return

    copied_csv_files_by_split = _copy_files_from_config(input_dir, output_dir, config_to_use)

    if not merge_files:
        return

    logger.info("Merging CSV files within each split...")
    for split_name in ["train", "validation", "test"]:  # Indent Level 1
        csv_files_to_merge = copied_csv_files_by_split.get(split_name, [])
        if len(csv_files_to_merge) == 0:
            logger.warning(f"No data for {split_name}")
        merged_df_path = output_path / f"{split_name}.csv"
        try:
            merged_df = pl.concat(csv_files_to_merge, how="vertical")
            merged_df.write_csv(merged_df_path)
            logger.info(f"Merged {len(csv_files_to_merge)} CSV files into '{merged_df_path}'")
        except Exception as e:
            logger.error(f"Error merging CSV files for split '{split_name}': {e}")
