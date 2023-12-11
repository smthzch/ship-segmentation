#%%
from tqdm import tqdm
from shipseg.data import load_config, load_data
from torchvision.io import read_image

config = load_config("../config/baseline.yaml")
data = load_data(".." / config["data_dir"])
img_dir = ".." / config["img_dir"]

# %%
img_ids = data["ImageId"].unique()
bad_imgs = []
for img_id in tqdm(img_ids):
    try:
        read_image(str(img_dir / img_id))
    except:
        bad_imgs.append(img_id)

# %%
bad_imgs
# %%
