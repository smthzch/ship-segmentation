"""
This script is to take an initial look at the data to try and understand what we are working with before we start modeling.
"""

#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

# %%
config = ld.load_config()

