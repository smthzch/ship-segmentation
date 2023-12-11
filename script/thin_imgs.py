#%%
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from shipseg.data import load_config, load_data
from shipseg.dataset import ShipDataset
from torch.utils.data import DataLoader


config = load_config("../config/baseline.yaml")
data = load_data(".." / config["data_dir"])
img_dir = ".." / config["img_dir"]

# %%
ds = ShipDataset(data, img_dir)
dl = DataLoader(ds, 64, num_workers=8)

# %%
sizes = np.zeros(len(ds))
for i in tqdm(range(len(ds))):
    img, mask = ds[i]
    sizes[i] = mask.sum()

# %%
sizes = np.zeros(len(ds))
ix = 0
for i, (img, mask) in tqdm(enumerate(dl), total=len(dl)):
    sizes[ix:(ix + mask.shape[0])] = mask.sum(axis=[1,2])
    ix += mask.shape[0]

# %%
np.random.seed(1)
cutoff = np.quantile(sizes, 0.9)
ixs = np.random.choice(
    np.argwhere(sizes > cutoff)[:,0],
    100,
    replace=False
)

# %%
img_ids = [
    ds.ids[i] for i in ixs
]

pd.DataFrame(dict(ImageId=img_ids)).to_csv("../data/thin_ids.csv", index=False)

#%%
img_ids = pd.read_csv("../data/thin_ids.csv")["ImageId"].values.tolist()
#%%
for f in img_dir.iterdir():
    if f.name not in img_ids:
        f.unlink()
# %%
