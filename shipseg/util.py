import numpy as np
import pandas as pd
import numpy.typing as npt

from PIL import Image

def imread(fn: str) -> npt.ArrayLike:
    return np.array(Image.open(fn)).astype(np.float32) / 255

def rle2mask(rle: str, h: int, w: int) -> npt.ArrayLike:
    total = h * w
    out = np.zeros(total, dtype=int)
    if isinstance(rle, str):
        rle_list = rle.split(" ")
        starts = [int(x) for x in rle_list[::2]]
        lengths = [int(x) for x in rle_list[1::2]]

        for start, length in zip(starts, lengths):
            out[start:(start + length)] = 1
    return out.reshape((h, w)).T

def id2mask(image_id: str, df: pd.DataFrame, h: int, w: int) -> npt.ArrayLike:
    rles = df.query(f"ImageId == '{image_id}'")["EncodedPixels"]
    mask = np.sum([rle2mask(rle, H, W) for rle in rles], axis=0)
    return mask

def mask2rle(mask: npt.ArrayLike) -> str:
    h, w = mask.shape
    long = mask.T.reshape(h * w)
    diff = np.diff(long)
    nzdiff = np.nonzero(diff)[0]
    starts = nzdiff[::2] + 1
    lengths = np.diff(nzdiff)[::2]
    rle = " ".join([str(s) + " " + str(l) for s, l in zip(starts, lengths)])
    return rle
