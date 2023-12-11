from shiny import App, ui, render, reactive
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import torch
from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

import shipseg.data as sd
import shipseg.model as sm
from shipseg.dataset import ShipDataset
from shipseg.transform import img_transforms

config_pth = "config/baseline.yaml"

# load data
config = sd.load_config(config_pth)

data_dir = config["data_dir"]
img_dir = config["img_dir"]
df = sd.load_data(data_dir)

thin_ids = pd.read_csv("data/thin_ids.csv")
df = df.query(f"ImageId in {thin_ids['ImageId'].values.tolist()}")

full_ds = ShipDataset(df, img_dir)
trans_ds = ShipDataset(df, img_dir, transforms=img_transforms)

# load model
device = "cuda" if config["app_kwargs"]["use_cuda"] and torch.cuda.is_available() else "cpu"
model = sm.load_model(
    config["model_dir"] / config["app_kwargs"]["run_id"] / "model.pt"
).to(device)

# app ui
app_ui = ui.page_fluid(
    ui.panel_title("Ship Segmentation"),
    ui.input_select(
        "img_id",
        "Image:",
        full_ds.ids
    ),
    ui.output_plot("plots")  
)

def server(input, output, session):
    @output
    @render.plot
    def plots():
        img_id = input.img_id()
        ix = full_ds.ids.index(img_id)
        image_, mask_ = full_ds[ix]
        image, mask = trans_ds[ix]
        image, mask = image[None,...].to(device), mask[None,...].to(device)
        with torch.no_grad():
            pred = model(image)
            
        f, ax = plt.subplots(1, 3)
        ax[0].set_title("Image")
        ax[0].imshow(image_.swapaxes(0, 2))
        ax[1].set_title("True Mask")
        ax[1].imshow(mask_.T)
        ax[2].set_title("Predict Mask")
        ax[2].imshow(pred["out"][0,0,...].cpu().numpy().T > 0)
        return f

app = App(app_ui, server)
