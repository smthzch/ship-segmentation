import sys
import os
import tempfile
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from pathlib import Path
import yaml

import torch

import library.data as ld
import library.model as lm
import library.eval as le

def main(config_pth):
    config = ld.load_config(config_pth)
    data_dir = config["data_dir"]

    # load data
    print("loading data")
    test = ld.load_labels("test", data_dir)


    # load model
    with open(config["model_dir"] / config["app_kwargs"]["run_id"] / "model.pkl", "rb") as f:
        model = pickle.load(f)
        model.reinit()

    if config["model_kwargs"]["use_cuda"] and torch.cuda.is_available():
        model.to_cuda()
        model.device = "cuda"
    else:
        model.to_cpu()
        model.device = "cpu"

    # predict
    print("predicting")
    preds = model.predict_df(test, config["img_dir"])

    test["prob"] = list(preds["prob"])

    # save
    out_pth = config["predict_dir"] / config["app_kwargs"]["run_id"]
    out_pth.mkdir(parents=True, exist_ok=True)
    test.to_csv(out_pth / "test_preds.csv", index=False)

if __name__ == "__main__":
    assert len(sys.argv) == 2, "must provide path to model config file"
    main(sys.argv[1])
