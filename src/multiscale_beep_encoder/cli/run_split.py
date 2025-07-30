import click

from multiscale_beep_encoder.data.preprocessing.split import organize_files


@click.option(
    "--input-dir",
    type=str,
    required=True,
    help="Input directory where all the .csv files are stored (from run_preprocessor)",
)
@click.option(
    "--output-dir",
    type=str,
    required=True,
    help="Output directory where test train and validation will be created",
)
@click.option(
    "--config",
    type=str,
    required=False,
    help="Existing JSON config path defining the mapping name->folder_type (train/test/validation)",
)
@click.option(
    "--ratio-train",
    type=float,
    default=0.8,
    required=False,
    help="Ratio of data attributed to train set",
)
@click.option(
    "--ratio-validation",
    type=float,
    default=0.05,
    required=False,
    help="Ratio of data attributed to validation set, test is 1- ratio-validation - ratio-train",
)
@click.option(
    "--merge-files",
    type=bool,
    default=True,
    required=False,
    help="Will result in 1 file per folder",
)
@click.command()
def main(input_dir, output_dir, config, ratio_train, ratio_validation, merge_files):
    """Organise the files in a train test evaluation split ready"""
    organize_files(
        input_dir,
        output_dir,
        existing_json_config_path=config,
        ratio_train=ratio_train,
        ratio_validation=ratio_validation,
        merge_files=merge_files,
    )


if __name__ == "__main__":
    main()
