import logging
from time import time

import numpy as np
import torch
from mblm.model.config import MBLMEncoderModelConfig, MBLMModelConfig
from mblm.model.transformer import TransformerEncoderBlock
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.training_args import TrainingArguments

from multiscale_beep_encoder.data.dataset.beep import BeepConfig, BeepDataloader
from multiscale_beep_encoder.loss.ntl import NumberTokenLossMSE
from multiscale_beep_encoder.utils.model_wrapper import (
    MLMDataCollator,
    NTLMBLMEncoder,
    WrapperMBLMEncoder,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DATA_DIR = "/dataset/beep-split/"
OUTPUT_DIR = "training_output"
TOKENIZER_URI = "/datasets/beep/tokenizer"
IS_NTL = True
NTL_WEIGHT = 0.3  # Only used if IS_NTL is True
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_URI)
MB_CONF = MBLMModelConfig(
    # Total number of tokens in the vocabulary. WARNING DON'T USE tokenizer.vocab_size as it isn't updated
    num_tokens=len(tokenizer),
    pad_token_id=tokenizer.pad_token_id,
    hidden_dims=[256, 64, 64],
    seq_lens=[2048, 8, 4],
    num_layers=[8, 4, 2],
    train_checkpoint_chunks=None,
    block=[
        TransformerEncoderBlock(
            attn_head_dims=8,
            attn_num_heads=32,
            attn_use_rot_embs=True,
            use_flash_attn=False,
            pos_emb_type="fixed",
        ),
        TransformerEncoderBlock(
            attn_head_dims=8,
            attn_num_heads=16,
            attn_use_rot_embs=True,
            use_flash_attn=False,
            pos_emb_type="fixed",
        ),
        TransformerEncoderBlock(
            attn_head_dims=8,
            attn_num_heads=4,
            attn_use_rot_embs=True,
            use_flash_attn=False,
            pos_emb_type="fixed",
        ),
    ],
)
OUTPUT_FOLDER = (
    OUTPUT_DIR
    + f"model_{'x'.join(str(x) for x in MB_CONF.seq_lens)}_{'L'.join(str(x) for x in MB_CONF.num_layers)}_{time():.0f}"
)
training_args = TrainingArguments(
    output_dir=OUTPUT_FOLDER,
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=4,
    learning_rate=0.001,
    max_steps=1_000_000,
    dataloader_num_workers=1,
    logging_strategy="steps",
    logging_steps=1_000,
    eval_strategy="steps",
    eval_steps=20_000,
    save_strategy="steps",
    save_steps=20_000,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="mlflow",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    # use_cpu=True,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    use_mps_device=False,
)


def train_encoder_mblm_hf():
    """Train the encoder model using Hugging Face Trainer"""

    ds = {
        folder: BeepDataloader(
            BeepConfig(
                data_dir=f"{DATA_DIR}/{folder}",
                tokenizer_uri=None,
                masking_proba=0.15,
                battery_stream="battery_stream",
                seq_len=int(np.prod(MB_CONF.seq_lens)),
            ),
            tokenizer=tokenizer,
        )
        for folder in ["train", "validation"]
    }

    # Initialize model
    # The MBLMEncoderModelConfig needs the base MBLMModelConfig and a return type for training
    model = (
        NTLMBLMEncoder(
            encoder_config=MBLMEncoderModelConfig(
                mblm_config=MB_CONF,
                mask_token_id=tokenizer.mask_token_id,
            ),
            ntl=NumberTokenLossMSE(
                tokenizer=tokenizer,
                use_logit_weight=False,
                device="cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu",
            ),
            ntl_weight=NTL_WEIGHT,
        )
        if IS_NTL
        else WrapperMBLMEncoder(
            encoder_config=MBLMEncoderModelConfig(
                mblm_config=MB_CONF, mask_token_id=tokenizer.mask_token_id
            )
        )
    )
    logger.info(f"model size : {sum(p.numel() for p in model.parameters())}")
    logger.info(f"model config : {MB_CONF}")
    # Data collator
    data_collator = MLMDataCollator()

    # Initialize Trainer
    logger.info("Initialize Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Start training
    logger.info("Starting training with HuggingFace Trainer...")
    try:
        trainer.train()
    except Exception as e:
        logger.info(f"Error while training: {e}, saving config model and quitting")
        model.save_cfg(OUTPUT_FOLDER + "/config.yaml")
    model.save_cfg(OUTPUT_FOLDER + "/config.yaml")
    logger.info("Training Done")
    del ds
    # Evaluate on the test set
    test_dataset = BeepDataloader(
        BeepConfig(
            data_dir=f"{DATA_DIR}/{'test'}",
            tokenizer_uri=None,
            masking_proba=0.15,
            battery_stream="battery_stream",
            seq_len=int(np.prod(MB_CONF.seq_lens)),
        ),
        tokenizer=tokenizer,
    )
    logger.info("Evaluating on the test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    logger.info(f"Test results: {test_results}")


if __name__ == "__main__":
    train_encoder_mblm_hf()
