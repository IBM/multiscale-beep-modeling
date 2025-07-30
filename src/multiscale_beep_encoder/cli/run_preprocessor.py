from pathlib import Path

import click

from multiscale_beep_encoder.data.preprocessing.beep_preprocessor import preprocess


@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    default="/data/encoder-mblm/beep-dataset/fullds-cutoff-100/inference/",
    help="Path were the csv files are stored. We expect a dir with 1 cell per file and the following key: cell_key,battery_stream. cell_key has the same name as the file and the battery_stream contains the string version of the data",
)
@click.option(
    "--model-config-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the model configuration file, this is not the weights, only the number of layers, dimention etc",
)
@click.option(
    "--model-weight",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    help="Path to the model weights if it's a directory, the file model.safetensors will be loaded from the dir, if it is a path to a file, the file is directly loaded ",
)
@click.option(
    "--tokenizer-uri",
    type=Path,
    help="Path to the directory where the patched tokenizer is stored or a Hugging Face uri (make sure to have the same token id than the tokenizer used during training)",
)
@click.option(
    "--output-dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    help="Dir where to write the resulting files eg: /your/path/",
)
@click.option(
    "--aggregation",
    type=str,
    default="mean",
    help="The aggregation method to use (mean,median), see the registry for a complet list",
)
@click.option(
    "--time-id",
    type=str,
    default="cycle_number",
    help="The column containing time information",
)
@click.option(
    "--stream-id",
    type=str,
    default="battery_stream",
    help="The column containing the sequence to embedd",
)
@click.option(
    "--name-id",
    type=str,
    default="cell_key",
    help="The column containing the sequence to embedd",
)
@click.option(
    "--unified-dir",
    type=str,
    required=True,
    help="A path to the original data, to append additional information like targets for supervised learning",
)
@click.option(
    "--targets",
    type=str,
    default="charge_policy_Q1,charge_policy_Q2,charge_policy_Q3,charge_policy_Q4,charge_capacity",
    help="The name of the target to collate, comma separated string",
)
@click.command()
def main(
    input_path,
    tokenizer_id,
    column_id,
    time_sep,
    column_time,
    output_dir,
    value_sep,
    smooth_charge_capacity,
    exclude_features,
    decimal_precision,
    time_cutoff,
    is_training_shuffled,
):
    """Preprocess the data"""
    preprocess(
        path=input_path,
        tokenizer_id=tokenizer_id,
        column_id=column_id,
        time_sep=time_sep,
        column_time=column_time.split(",") if column_time else [],
        output_dir=output_dir,
        value_sep=value_sep,
        smooth_charge_capacity=smooth_charge_capacity,
        exclude_features=exclude_features.split(",") if exclude_features else [],
        decimal_precision=decimal_precision,
        time_cutoff=time_cutoff.split(",") if time_cutoff else None,
        is_training_shuffled=is_training_shuffled,
    )


if __name__ == "__main__":
    main()
