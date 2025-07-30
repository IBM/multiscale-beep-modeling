import logging
from pathlib import Path
from typing import List

import click
import numpy as np
import polars as pl
import torch
from safetensors.torch import load_file
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from multiscale_beep_encoder.data.dataset.beep import (
    BatteryHandler,
    verify_tokenizer,
)
from multiscale_beep_encoder.modeling.registry.embedding_aggregation_registry import get_registry
from multiscale_beep_encoder.utils.model_wrapper import WrapperMBLMEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def embedd_cycle(
    data: str,
    model: WrapperMBLMEncoder,
    tokenizer: PreTrainedTokenizer,
    aggregation_function,
):
    """Embedd a cycle
    Params:
        data: A tuple containing cycle_number, the stream of character
        model: the model to use to embedd
        tokenizer: the model to tokenize the stream of character
        aggregation_function:  the function to apply on the list of embeddings
    returns:
        the embedding tensor
    """
    # We first tokenize the data, then splits it in chunks of size np.prod(model.cfg.mblm_config.seq_lens) . We padd the last chunk using BeepDataloader.padd and the tokenizer.pad_token_id
    # We embed each chunk using the model
    # We concatenate each embedded chunk
    # we apply the aggregation function
    # we return the tuple cycle_number, embeddings
    with torch.no_grad():
        tokenized_data: torch.Tensor = tokenizer(
            data, return_tensors="pt", return_attention_mask=False
        )["input_ids"].squeeze(0)  # type: ignore
        seq_len = int(np.prod(model.cfg.mblm_config.seq_lens))
        chunks = [tokenized_data[i : i + seq_len] for i in range(0, len(tokenized_data), seq_len)]
        if len(chunks[-1]) != seq_len:
            # Pad on the right (end) the last chunk
            logger.debug("Padding")
            chunks[-1] = torch.nn.functional.pad(
                chunks[-1],
                (0, seq_len - chunks[-1].size(-1)),
                "constant",
                tokenizer.pad_token_id,  # type:ignore
            )
        input = chunks[0].unsqueeze(0)
        if len(chunks) > 1:
            input = torch.stack(chunks)
        logger.debug(f"{input.size()=}")
        device = next(model.parameters()).device
        input = input.to(device)
        embeddings: torch.Tensor = model(input)["hidden_state"]
        logger.debug(f"{embeddings.size()=}")
        # Merge the batch and sequence length
        embeddings = embeddings.view(-1, embeddings.size(-1))
        embeddings = aggregation_function(embeddings)
        logger.debug(f"Aggregated embeddings: {embeddings.size()=}")
    return embeddings


def embed_battery(
    ds: BatteryHandler,
    model: WrapperMBLMEncoder,
    tokenizer: PreTrainedTokenizer,
    name_id: str = "cell_key",
    stream_id: str = "battery_stream",
    time_id: str = "cycle_number",
    aggregation: str = "mean",
) -> pl.DataFrame:
    """"""
    all_row = []
    for index in range(ds.get_number_of_cycles()):
        data = ds.get_item_and_meta(index)
        cycle_number = data[time_id]
        battery_name = data[name_id]
        # Embedding is of shape (n,)
        data_stream: str = data[stream_id]  # type:ignore

        embedding = embedd_cycle(
            data_stream, model, tokenizer, get_registry(dim=0).get(aggregation)
        )
        row = {name_id: battery_name, time_id: cycle_number} | {
            f"val_{i}": embedding[i].item() for i in range(len(embedding))
        }
        all_row.append(row)
    # Register the battery_name, cycle number and embeddings (data[col_id],data[time_id] ) in a df
    return pl.DataFrame(all_row)


def embed_dataset(
    dss: List[BatteryHandler],
    model: WrapperMBLMEncoder,
    tokenizer: PreTrainedTokenizer,
    aggregation: str = "mean",
    name_id: str = "cell_key",
    stream_id: str = "battery_stream",
    time_id: str = "cycle_number",
) -> pl.DataFrame:
    """Embed the data available in each ds of the dss list using the model"""

    dfs = []
    for battery_handler in tqdm(dss):
        df = embed_battery(
            battery_handler,
            model,
            tokenizer,
            aggregation=aggregation,
            name_id=name_id,
            stream_id=stream_id,
            time_id=time_id,
        )

        dfs.append(df)
    return pl.concat(dfs, how="vertical")


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
def run_inference(
    tokenizer_uri: str,
    data_dir: Path,
    model_config_path: Path,
    model_weight: Path,
    output_dir: Path,
    aggregation: str,
    stream_id: str,
    time_id: str,
    name_id: str,
    unified_dir: str,
    targets: str,
):
    """Create and store the embeddings associated with each csv file stored in the data-dir"""
    assert output_dir.is_dir(), "output-dir must be a directory"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        logger.info("MPS available")
        if not torch.backends.mps.is_built():
            logger.info("PyTorch not built with mps support")
        else:
            device = "mps"

    logger.info(f"Using device: {device}")
    target_list = targets.split(",")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_uri)
    assert verify_tokenizer(tokenizer), (
        "Tokenizer verification failed. Ensure all special tokens are set."
    )
    assert isinstance(tokenizer.mask_token_id, int)
    assert isinstance(tokenizer.pad_token_id, int), "Tokenizer must have a pad_token_id set."

    model = WrapperMBLMEncoder.load_cfg(model_config_path.as_posix(), is_embedder=True)

    if model.cfg.mask_token_id != tokenizer.mask_token_id:
        logger.warning(
            "Config and tokenizer do not share the same mask_token_id, this might indicate a mismatch"
        )

    model_weight_path = (
        (model_weight / "model.safetensors") if model_weight.is_dir() else model_weight
    )
    logger.info(f"Loading model weights from: {model_weight_path}")
    dic_tensor_weights = load_file(model_weight_path, device="cpu")  # Load to CPU first
    res = model.load_state_dict(dic_tensor_weights)

    if res.missing_keys or res.unexpected_keys:
        logger.info(
            f"Model loading results - Missing keys: {res.missing_keys}, Unexpected keys: {res.unexpected_keys}"
        )
        logger.warning(
            "There were missing or unexpected keys when loading model weights. Check compatibility."
        )

    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    names = [path for path in Path(data_dir).iterdir() if path.name.endswith(".csv")]
    dss = [
        BatteryHandler(
            path.resolve().as_uri(),
            sequence_len=int(np.prod(model.cfg.mblm_config.seq_lens)),
            stream_id=stream_id,
            time_id=time_id,
        )
        for path in names
    ]

    logger.info(f"Found {len(dss)} files to process in {data_dir}.")

    # final_data returns a a list of embeddings for each device_id
    # Each embedding in the list is the average embedding of one sequence for that device
    battery_cycle_components_df = embed_dataset(
        dss,
        model,
        tokenizer,
        aggregation=aggregation,
        name_id=name_id,
        stream_id=stream_id,
        time_id=time_id,
    )
    additional_information = (
        pl.scan_csv(Path(unified_dir) / "*.csv")
        .select(pl.col(target_list), pl.col(name_id), pl.col(time_id))
        .group_by(pl.col(name_id), pl.col(time_id))
        .agg(pl.col(target_list).max())
    ).collect()

    logger.info(f"Null values after embedding: {battery_cycle_components_df.null_count().sum()}")
    battery_cycle_components_df = additional_information.join(
        battery_cycle_components_df, on=[name_id, time_id], how="left"
    )
    logger.info(
        f"Null values after adding information to embeddings: {battery_cycle_components_df.null_count().sum()}"
    )

    for name, data in battery_cycle_components_df.group_by(name_id):
        logger.info(f"Writing {name[0]}")
        data.with_columns(index=pl.col(name_id)).write_csv(output_dir / f"{name[0]}.csv")

    logger.info("Inference run completed.")


if __name__ == "__main__":
    run_inference()
