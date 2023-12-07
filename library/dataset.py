"""Module containing datasets to perform specific data loading functions for models.

Datasets inherit from `torch.utils.data.Dataset` and contain specific methods which are used by
dataloaders to fetch batches of data and perform transforms before being fed to a model.
"""

import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.io.image import read_image

from pathlib import Path
from typing import Callable

class BaseDataset(Dataset):
    """Base dataset
    """
    def __init__(self, df: pd.DataFrame, data_dir: str, transforms: Callable=None):
        """
        parameters
        ----------
            df: dataframe of ids
            data_dir: directory to load data
            transforms: function to transform loaded image
        """
        self.df = df
        self.boxes = boxes
        self.data_dir = data_dir if isinstance(img_dir, Path) else Path(data_dir)
        self.transforms = transforms
            
    def __len__(self) -> int:
        return self.df.shape[0]
    
    def __getitem__(self, ix: int) -> tuple[torch.FloatTensor, str]:
        item = self.df.iloc[ix]
        img_id = item["id"]

        img = read_image(str(self.img_dir / (img_id + ".jpg")))

        if self.transforms:
            img = self.transforms(img)

        return img, manufacturer

