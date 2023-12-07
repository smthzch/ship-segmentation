"""This module contains model classes for fitting and predicting models.

"""

import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any
import numpy.typing as npt
from tqdm import tqdm

import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import v2
from torch.utils.data import DataLoader

class BaseModel(ABC):
    def __init__(self):
        # init pre-trained model
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(num_classes=hidden_dim)
        self.preprocess = v2.Compose([
            v2.Resize((100, 100), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    @abstractmethod
    def train(self, train_df: pd.DataFrame, data_dir: str, **kwargs):
        """
        # make dataloader
        dataset = BaseDataset(train_df, data_dir, transforms=self.preprocess)
        dl = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=int(os.cpu_count() / 2))

        # fit latent model
        self.model.train()
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for epoch in tqdm(range(epochs)):
            for x, y in dl:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.loss(pred)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if log_metric is not None:
                    log_metric("loss", loss.item())
        """
        raise NotImplementedError

    @abstractmethod
    def predict_img(self, img: torch.FloatTensor) -> dict[str, npt.ArrayLike]:
        """Predict a single image
        
        returns
        -------
            dict of string, array pairs with shapes {"bbox": (4,), "prob": (1,30), "proj": (1,2)}
        """
        raise NotImplementedError

    @abstractmethod
    def predict_df(self, df: pd.DataFrame, img_dir: str) -> dict[str, npt.ArrayLike]:
        """Predict a dataframe of image ids
        
        returns
        -------
            dict of string, array pairs with shapes {"bbox": (N,4), "prob": (N,30), "proj": (N,2)}
        """
        raise NotImplementedError

    @abstractmethod
    def to_cpu(self):
        """Move all models to cpu"""
        raise NotImplementedError

    @abstractmethod
    def to_cuda(self):
        """Move all models to gpu"""
        raise NotImplementedError

