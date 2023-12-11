"""Module containing datasets to perform specific data loading functions for models.

Datasets inherit from `torch.utils.data.Dataset` and contain specific methods which are used by
dataloaders to fetch batches of data and perform transforms before being fed to a model.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image  
from sklearn.model_selection import train_test_split

from shipseg.util import rle2mask

class ShipDataset(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame, 
        img_dir: Path, 
        transforms: Optional[Callable]=None, 
        split: Optional[str]=None
    ):
        ids = df["ImageId"].unique().tolist()
        try:
            ids.remove("6384c3e78.jpg") # rm bad image
        except:
            pass
        if split is not None:
            assert split in ["train", "test"], "split must be either ['train', 'test']"
            train, test = train_test_split(ids, random_state=0)
            ids = train if split == "train" else test
        
        self.ids = ids
        self.df = df.query(f"ImageId in {ids}")
        self.transforms = transforms
        self.img_dir = img_dir

        # get image dim
        temp_id = df.iloc[0]["ImageId"]
        self.C, self.H, self.W = read_image(str(self.img_dir / temp_id)).shape
            
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, ix: int) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        image_id = self.ids[ix]
        rles = self.df.query(f"ImageId == '{image_id}'")["EncodedPixels"]
        img = read_image(str(self.img_dir / image_id))
        mask = np.sum([rle2mask(rle, self.H, self.W) for rle in rles], axis=0)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        
        return img, mask

