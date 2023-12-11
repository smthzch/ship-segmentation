"""This module contains model classes for fitting and predicting models.

"""

import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Callable
import numpy.typing as npt
from tqdm import tqdm
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

def new_model(device: str) -> nn.Module:
    model = lraspp_mobilenet_v3_large(num_classes=1).float().to(device)
    for params in model.backbone.parameters():
        params.requires_grad = False
    return model

def train_model(
    model: nn.Module, 
    dl: DataLoader,
    epochs: int, 
    lr: float, 
    device: str, 
    log_metric: Callable
) -> list[float]:
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(dl)
    )

    loser = nn.BCEWithLogitsLoss()

    # train
    losses = []
    for _ in tqdm(range(epochs)):
        for image, mask in dl:
            image = image.to(device)
            mask = mask[:,None,:,:].to(device)

            pred = model(image)["out"]
            loss = loser(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses += [loss.item()]
            log_metric("train_loss", loss.item())
    return losses

def save_model(model: nn.Module, pth: str):
    model.cpu()
    torch.save(model.state_dict(), pth)

def load_model(pth: str) -> nn.Module:
    model = lraspp_mobilenet_v3_large(num_classes=1).float()
    model.load_state_dict(torch.load(pth))
    model.eval()
    return model
