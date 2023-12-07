"""Module for loading and parsing data and configs."""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

def load_config(config_pth: Path) -> dict[str, Any]:
    with open(config_pth, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config["data_dir"] = Path(config["data_dir"])
    config["img_dir"] = config["data_dir"] / "images"
    config["model_dir"] = Path(config["model_dir"])
    config["predict_dir"] = Path(config["predict_dir"])
    return config
    

def load_data() -> pd.DataFrame:
    pass

