import sys
import os
import tempfile
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from PIL import Image
from tqdm import tqdm
from pathlib import Path
import yaml

import library.data as ld
import library.model as lm
import library.eval as le

def main(config_pth):
    mlflow.start_run()
    run_id = mlflow.active_run().info.run_id
    print(f"run id: {run_id}")

    config = ld.load_config(config_pth)
    data_dir = config["data_dir"]

    mlflow.log_params(config)

    # load data
    print("loading data")
    train = ld.load_data("train", data_dir)

    # build model
    print("building model")
    assert hasattr(lm, config["model_name"])
    model = getattr(lm, config["model_name"])(**config["model_kwargs"])

    # train
    print("training")
    model.train(train, config["img_dir"], log_metric=mlflow.log_metric, **config["train_kwargs"])

    # eval
    print("evaluating")
    preds = model.predict_df(train, config["img_dir"])
    
    probs = preds["prob"]
    val_enc = model.class_encoder.transform(val["label"].values[:,None])
    
    metrics = le.eval_preds(val_enc, probs)
    mlflow.log_metrics(metrics)
    
    cm = le.confusion_matrix_plot(val_enc, probs, labels)
    with tempfile.TemporaryDirectory() as d:
        f = Path(d, "confusion_matrix.png")
        cm.savefig(f)
        mlflow.log_artifact(f)

    # save
    model.prepare_to_save()
    out_pth = config["model_dir"] / run_id 
    out_pth.mkdir(parents=True, exist_ok=False)
    with open(out_pth / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    mlflow.end_run()


if __name__ == "__main__":
    assert len(sys.argv) == 2, "must provide path to model config file"
    main(sys.argv[1])
