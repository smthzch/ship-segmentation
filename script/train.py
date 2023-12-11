import sys
import mlflow
import yaml
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

import shipseg.data as sd
import shipseg.model as sm
from shipseg.dataset import ShipDataset
from shipseg.transform import img_transforms

def main(config_pth):
    mlflow.start_run()
    run_id = mlflow.active_run().info.run_id
    print(f"run id: {run_id}")

    config = sd.load_config(config_pth)
    data_dir = config["data_dir"]
    img_dir = config["img_dir"]
    train_kwargs = config["train_kwargs"]

    mlflow.log_params(config)

    # load data
    print("loading data")
    train_df = sd.load_data(data_dir)
    train_ds = ShipDataset(train_df, img_dir, transforms=img_transforms)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=12, pin_memory=True)

    # build model
    device = "cuda" if train_kwargs["use_cuda"] and torch.cuda.is_available() else "cpu"
    model = sm.new_model(device)

    # train
    print("training")
    sm.train_model(
        model,
        train_dl,
        train_kwargs["epochs"],
        train_kwargs["lr"],
        device,
        log_metric=mlflow.log_metric
    )

    # save
    out_pth = config["model_dir"] / run_id 
    out_pth.mkdir(parents=True, exist_ok=True)
    sm.save_model(model, out_pth / "model.pt")

    mlflow.end_run()


if __name__ == "__main__":
    assert len(sys.argv) == 2, "must provide path to model config file"
    main(sys.argv[1])
