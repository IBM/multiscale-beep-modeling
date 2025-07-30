# Introduction
This repo implements the Multiscale Encoder, an encoder-only transformer model leveraging representation learning, for hyper-long sequence learning.
We assume some form of time series data such as: batteries condition through time. But any data that can be seen as such is usable with our model.

# Workflow:
In order to use this repo with the BEEP dataset, you should follow the following steps:
If you plan to use another dataset, the process is highly similar but the processing will be done by you. 
1. Download data and format it.
2. Generate a unified csv containing the data.
3. Generate training dataset & train on it.
4. Generate embeddings
5. (Optionally) Downstream tasks evaluation.


## Download data and format it.
First, get your hand on the data [here](https://data.matr.io/1/). The link has two projects, each linked to a specific dataset, we merged both to have more training data.
You can use their code to generate the pickle file using [their repo](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation), if you have some issue with their code, use the [updated code](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/pull/37). Generate the pickle file using their code, you should generate it for the all batches in the [first project](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204) and the [validation set of the second project.](https://data.matr.io/1/projects/5d80e633f405260001c0b60a/batches/5dcef1fe110002c7215b2c94) as they are the only batches containing data for more than 100 cycles. If your use case allows it, fell free to use the other batches.

## Generate a unified csv containing the data.
That step relies on the [beep-data-utils](https://github.com/IBM/beep-data-utils) repos, clone it, then `poetry install` and you should be able to run Step 1. Once done, you can continue our README.md
For ease of use, Step 1 is reproduced here:
```bash
# In the beep-data-utils root folder
poetry run beep-dataset-creation \
    --pickle-data-path "/path/to/pickle/data" \
    --output-path "output/beep_unified_dataset.csv" \
```

## Generate training dataset & train on it.
We now create a string representation from the data, this will be used for training.
You can run it the following way:
```bash
# In the multiscale repo:
uv run preprocess\
        --input-path "output/beep_unified_csv/*.csv"\
        --tokenizer-id "ibm-granite/granite-3.3-8b-base"\
        --column-id "cell_key"\
        --time-sep "<|sep|>"\
        --value-sep ","\
        --column-time "cycle_numer,time"#comma separated list of values\
        --output-dir "/datasets/beep"\
        --smooth-charge-capacity\
        --time-cutoff "1"\
        --is-training-shuffled
```
We recommend using the `--is-training-shuffled` for training, this shuffles the data for pre-training otherwise you might encounter a situation where the models interpolate and doesn't learn anything usefull, again feel free to adapt depending on your dataset and usecase.
You can use `uv run preprocess --help` to have the full list of arguments and their meaning. The tokenizer will be patched with any missing tokens, and you should load it from `output-dir/tokenizer` for all subsequent steps requiring a tokenizer.

### Splitting the data:
You must split the data in train, test and validation, you can provide your own config as json or split it based on the amount of data available.
```bash
uv run split --input-dir /datasets/beep --ouput-dir /dataset/beep-split/
```
You can specify the ratio used for train, test and validation it by using the `--ratio-train` and `--ratio-validation` options. Test ratio is defined as 1-test-validation. This generate a split configuration in the same folder, we recommend using the same config for the downstream tasks.

### Training

The file `src/multiscale_encoder/cli/run_training.py` contains all the value to specify to run a training. We don't provide a cli as the model architecture would require you to run the code once to generate a dump of the architecture, defeating the purpose of the cli.
The values to change are:
```
DATA_DIR = "/dataset/beep-split/"# Should be set to the output-dir of the previous uv command
OUTPUT_DIR = "training_output"
TOKENIZER_URI = "/datasets/beep/tokenizer"# The updated tokenizer saved during preprocessing.
IS_NTL = True #Compute a loss that is the mix of CE and NTL.
```

## Generating embeddings:
We can now use the model to create embeddings.
```
uv run embedd
--data-dir /dataset/beep/ \
--model-config-path training_output/config.yaml \
--model-weight training_output/checkpoint-your-timestep/\
--tokenizer-uri /datasets/beep/tokenizer\
--output-dir /dataset/beep-embeddings/\
--unified-dir output/beep_unified_csv\
--targets "charge_capacity,other_features_to_join"\
```
You can now use the embeddings for your prefered task. In order to be ready for downstream task, you can specify a directory using `--unified-dir` where the "`targets` will be joined to the embeddings on the `name-id` and time-id`. Again use the `--help` to see the full list of options.

# (Optionally) Downstream tasks evaluation. 
Before starting the downstream task evaluation, you need to split the data for the supervised training into train, test and validation. We recommend using the configuration created in the step "splitting the data".

```
uv run split --input-dir dataset/beep-embeddings/ --output-dir dataset/beep-embeddings-for-supervised-learning/ --config /dataset/beep-split/split_cfg.json
```
We can then train using any of the models available in [Beep-data-utils](https://github.com/IBM/beep-data-utils)
