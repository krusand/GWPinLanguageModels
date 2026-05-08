"""
Inference / prompting script (Tiny Shakespeare, char-level).
Students will integrate sustainability tracking themselves.

Source: https://github.com/karpathy/nanoGPT
"""

# MAX_NEW_TOKENS = [100, 200, 300, 400] BS: 200
# TEMPERATURE = [0.8, 0.9, 0.95, 1.0] BS: 1.0
# TOP_K = [25, 50, 75, 100] BS: 50


import os
import pickle
import torch
import pandas as pd
from codecarbon import EmissionsTracker

from tqdm import tqdm
from model import GPT, GPTConfig
import time
from pathlib import Path
Path("./emissions_corrected").mkdir(exist_ok=True)


PROJECT_NAME = "prompting_phase"


# ----------------------------
# Edit these
# ----------------------------
OUT_DIR = "out"
CKPT_PATH = os.path.join(OUT_DIR, "ckpt.pt")

PROMPT = "To be, or not to be"
MAX_NEW_TOKENSs = [100, 200, 300, 400]
TEMPERATUREs = [0.8, 0.9, 0.95, 1.0]
TOP_Ks = [25, 50, 75, 100]

BS_MAX_NEW_TOKENS = 200
BS_TEMPERATURE = 1.0
BS_TOP_K = 50

DEVICE = "mps"
# ----------------------------


def load_meta(data_dir: str):
    meta_path = os.path.join(data_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        return pickle.load(f)



def main():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    # train.py should store config with model parameters and data_dir
    data_dir = ckpt["config"]["data_dir"]
    model_cfg = ckpt["config"]["model"]

    meta = load_meta(data_dir)
    stoi = meta["stoi"]         # char to index mapping
    itos = meta["itos"]         # index to char mapping

    def encode(s: str):
        # map unknown chars to a safe fallback if needed
        return [stoi.get(ch, stoi[" "]) for ch in s]

    def decode(tokens):
        return "".join([itos[t] for t in tokens])

    config = GPTConfig(**model_cfg)
    model = GPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])

    model.eval()

    idx = torch.tensor([encode(PROMPT)], dtype=torch.long, device=DEVICE)

    out = model.generate(
        idx,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K
    )

    print(decode(out[0].tolist()))


if __name__ == "__main__":

    max_new_tokens_configs = [{'MAX_NEW_TOKENS': max_new_tokens, 'TEMPERATURE': BS_TEMPERATURE, 'TOP_K': BS_TOP_K} for max_new_tokens in MAX_NEW_TOKENSs]
    temperature_configs = [{'MAX_NEW_TOKENS': BS_MAX_NEW_TOKENS, 'TEMPERATURE': temperature, 'TOP_K': BS_TOP_K} for temperature in TEMPERATUREs]
    top_k_configs = [{'MAX_NEW_TOKENS': BS_MAX_NEW_TOKENS, 'TEMPERATURE': BS_TEMPERATURE, 'TOP_K': top_k} for top_k in TOP_Ks]

    model_configs = max_new_tokens_configs + temperature_configs + top_k_configs
    
    for model_config in tqdm(model_configs):
        MAX_NEW_TOKENS = model_config["MAX_NEW_TOKENS"]
        TEMPERATURE = model_config["TEMPERATURE"]
        TOP_K = model_config["TOP_K"]

        tracker = EmissionsTracker(project_name=PROJECT_NAME
                                , log_level='critical'
                                , output_dir="./emissions_corrected")
        tracker.start()
        try:
            start_time = time.time()
            main()
            end_time = time.time()
            training_time = end_time - start_time

        finally:
            emissions = tracker.stop()
            if emissions is None:
                emissions = 0

            run_params = {  "project_name": PROJECT_NAME,
                        "run_id": str(tracker.run_id),
                        "start_time": start_time,
                        "end_time": end_time,
                        "training_time_secs": training_time,
                        "emissions": emissions,
                        "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
                        "TEMPERATURE": TEMPERATURE,
                        "TOP_K": TOP_K
                    }

            run_params.update(tracker.get_detected_hardware())

            df = pd.DataFrame([run_params], index=[0])
            df.to_csv(f"./emissions_corrected/run_params_{PROJECT_NAME}.csv", index=False, mode='a', header=not os.path.exists(f"./emissions_corrected/run_params_{PROJECT_NAME}.csv"))



