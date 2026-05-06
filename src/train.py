"""
Course training script (simplified from nanoGPT).

Focus:
- Train a small GPT-style model from scratch on a tiny dataset.
- Students will integrate sustainability tracking themselves.

Source: https://github.com/karpathy/nanoGPT
"""

# Batch size [16, 32, 64, 128]: BS: 32
# N layers [4, 8, 12, 16] BS: 4
# N embed [32, 64, 128, 256] BS: 128

import os
import time
import pickle
from dataclasses import asdict
from codecarbon import EmissionsTracker

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

from model import GPTConfig, GPT
from pathlib import Path

# -----------------------------------------------------------------------------
# Experiment configuration

PROJECT_NAME = "training_phase"
OUTPUT_DIR = "./emissions_corrected"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# I/O
OUT_DIR = "out"
DATA_DIR = os.path.join("data")
EVAL_INTERVAL = 200
EVAL_ITERS = 50
LOG_INTERVAL = 50
SAVE_CHECKPOINT = True

# Model (main tunables)

## We tune these:
BS_BATCH_SIZE = 32
BS_N_LAYER = 16  # 4
BS_N_EMBD = 256  # 128

BATCH_SIZEs = [16, 32, 64, 128]  # Number of sequences processed in parallel.
N_LAYERs = [4, 8, 12, 16]
N_EMBDs = [32, 64, 128, 256]


N_HEAD = 4
DROPOUT = 0.1
BIAS = True

# Training (main parameters you can also experiment with)
SEED = 1
DEVICE = "mps"  # If you can, try also seeing consumption when using gpu (change this to 'cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = "float32"
BLOCK_SIZE = 256  # Maximum context length for predictions (e.g. 128 or 256). The longer the block size, the more memory and compute it requires, but it can also lead to better performance.
N_EPOCHS = 5
MAX_ITERS = 2000  # Total number of training iterations. The more iterations, the better the model can perform, but it also takes more time and energy to train.
LEARNING_RATE = (
    3e-4  # the standard starting learning rate, often good enough for a first try
)
WEIGHT_DECAY = 0.1  # L2 Regularization
GRAD_CLIP = 1.0  # To prevent exploding gradients


# -----------------------------------------------------------------------------
def print_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print("model size: {:.3f}MB".format(size_all_mb))


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_meta(data_dir: str):
    meta_path = os.path.join(data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "rb") as f:
        return pickle.load(f)


class MemmapTextDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, split: str, block_size: int):
        bin_path = os.path.join(data_dir, f"{split}.bin")

        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        # The total number of valid starting indices
        return len(self.data) - self.block_size

    def __getitem__(self, i):
        # Grab the block of text + 1 extra token for the targets
        chunk = self.data[i : i + self.block_size + 1].astype(np.int64)

        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])

        return x, y


def create_dataloader(
    split: str,
    data_dir: str,
    block_size: int,
    device: str,
    batch_size: int,
    num_workers: int = 0,
):
    dataset = MemmapTextDataset(data_dir, split, block_size)

    is_train = split == "train"
    is_not_cpu = device != "cpu"

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=is_not_cpu,
        drop_last=True,
    )

    return dataloader


def save_checkpoint(
    out_dir: str,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    iter_num: int,
    config: dict,
):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "iter_num": iter_num,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "config": config,
    }
    torch.save(ckpt, os.path.join(out_dir, "ckpt.pt"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)

    run_params_main = dict()

    meta = load_meta(DATA_DIR)
    vocab_size = meta["vocab_size"] if meta and "vocab_size" in meta else 50304
    run_params_main["vocab_size"] = vocab_size

    cfg = GPTConfig(
        block_size=BLOCK_SIZE,
        vocab_size=vocab_size,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
        bias=BIAS,
    )

    # create the model and move it to the device
    model = GPT(cfg).to(DEVICE)

    # create the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )
    run_params_main["optimizer"] = optimizer.__class__.__name__

    # (optional) uncomment this for printing model size once
    print(f"Device: {DEVICE}")
    print(f"Model parameters: {model.get_num_params():,}")
    run_params_main["n_model_params"] = model.get_num_params()
    run_params_main["model_size_mb"] = get_model_size(model)

    train_dataloader = create_dataloader(
        "train", DATA_DIR, BLOCK_SIZE, DEVICE, BATCH_SIZE
    )
    val_dataloader = create_dataloader("val", DATA_DIR, BLOCK_SIZE, DEVICE, BATCH_SIZE)

    print(
        f"Training for {N_EPOCHS} epochs | Training on {len(train_dataloader)} data batches| batch={BATCH_SIZE} | block={BLOCK_SIZE} | Total tokens={(len(train_dataloader)+len(val_dataloader))*BLOCK_SIZE*BATCH_SIZE}"
    )

    for epoch in range(N_EPOCHS):
        epoch_start_time = time.time()
        epoch_train_loss = 0
        epoch_val_loss = 0
        for x_train, y_train in tqdm(train_dataloader):
            x_train, y_train = x_train.to(DEVICE), y_train.to(DEVICE)
            model.train()

            # training step
            _, train_loss = model(x_train, y_train)

            optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            epoch_train_loss += train_loss.item()

            if GRAD_CLIP and GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()
        for x_val, y_val in tqdm(val_dataloader):
            x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
            with torch.no_grad():
                model.eval()
                _, val_loss = model(x_val, y_val)
            epoch_val_loss += val_loss.item()

        epoch_duration = time.time() - epoch_start_time
        print(
            f"Epoch {epoch:5d} | avg train loss {(epoch_train_loss / len(train_dataloader)):.4f} | val loss {(epoch_val_loss / len(val_dataloader)):.4f} | elapsed {epoch_duration:.1f}s"
        )

        if SAVE_CHECKPOINT and epoch > 0:
            config_dump = {
                "data_dir": DATA_DIR,
                "train": {
                    "batch_size": BATCH_SIZE,
                    "block_size": BLOCK_SIZE,
                    "max_iters": MAX_ITERS,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "grad_clip": GRAD_CLIP,
                    "dtype": DTYPE,
                    "device": DEVICE,
                },
                "model": asdict(cfg),
            }
            save_checkpoint(OUT_DIR, model, optimizer, epoch, config_dump)
        torch.mps.empty_cache()

    print("Training completed.")

    # Save final checkpoint
    if SAVE_CHECKPOINT:
        config_dump = {
            "data_dir": DATA_DIR,
            "train": {
                "batch_size": BATCH_SIZE,
                "block_size": BLOCK_SIZE,
                "max_iters": MAX_ITERS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "grad_clip": GRAD_CLIP,
                "dtype": DTYPE,
                "device": DEVICE,
            },
            "model": asdict(cfg),
        }
        save_checkpoint(OUT_DIR, model, optimizer, MAX_ITERS, config_dump)

    return run_params_main


if __name__ == "__main__":

    batch_size_configs = [
        {"BATCH_SIZE": batch_size, "N_LAYER": BS_N_LAYER, "N_EMBD": BS_N_EMBD}
        for batch_size in BATCH_SIZEs
    ]
    n_layer_configs = [
        {"BATCH_SIZE": BS_BATCH_SIZE, "N_LAYER": n_layer, "N_EMBD": BS_N_EMBD}
        for n_layer in N_LAYERs
    ]
    n_embd_configs = [
        {"BATCH_SIZE": BS_BATCH_SIZE, "N_LAYER": BS_N_LAYER, "N_EMBD": n_embd}
        for n_embd in N_EMBDs
    ]

    model_configs = batch_size_configs + n_layer_configs + n_embd_configs

    for model_config in tqdm(model_configs):
        BATCH_SIZE = model_config["BATCH_SIZE"]
        N_LAYER = model_config["N_LAYER"]
        N_EMBD = model_config["N_EMBD"]

        tracker = EmissionsTracker(
            project_name=PROJECT_NAME, log_level="critical", output_dir=OUTPUT_DIR
        )
        tracker.start()
        try:
            start_time = time.time()
            run_params_main = main()
            end_time = time.time()
            training_time = end_time - start_time

        finally:
            emissions = tracker.stop()
            if emissions is None:
                emissions = 0

            run_params = {
                "project_name": PROJECT_NAME,
                "run_id": str(tracker.run_id),
                "start_time": start_time,
                "end_time": end_time,
                "training_time_secs": training_time,
                "emissions": emissions,
                "BATCH_SIZE": BATCH_SIZE,
                "BLOCK_SIZE": BLOCK_SIZE,
                "MAX_ITERS": MAX_ITERS,
                "LEARNING_RATE": LEARNING_RATE,
                "WEIGHT_DECAY": WEIGHT_DECAY,
                "GRAD_CLIP": GRAD_CLIP,
                "DTYPE": DTYPE,
                "DEVICE": DEVICE,
                "EVAL_INTERVAL": EVAL_INTERVAL,
                "EVAL_ITERS": EVAL_ITERS,
                "LOG_INTERVAL": LOG_INTERVAL,
                "SAVE_CHECKPOINT": SAVE_CHECKPOINT,
                "N_LAYER": N_LAYER,
                "N_HEAD": N_HEAD,
                "N_EMBD": N_EMBD,
                "DROPOUT": DROPOUT,
                "BIAS": BIAS,
            }

            run_params.update(run_params_main)
            run_params.update(tracker.get_detected_hardware())

            df = pd.DataFrame([run_params], index=[0])
            df.to_csv(
                f"./emissions/run_params_{PROJECT_NAME}.csv",
                index=False,
                mode="a",
                header=not os.path.exists(f"./emissions/run_params_{PROJECT_NAME}.csv"),
            )
