import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

import library.data as ld

assert len(sys.argv) == 2
config_pth = sys.argv[1]

# load data
config = d.load_config(config_pth)

data_dir = config["data_dir"]
img_dir = config["img_dir"]
predict_dir = config["predict_dir"]

# load model
with open(config["model_dir"] / config["app_kwargs"]["run_id"] / "model.pkl", "rb") as f:
    model = pickle.load(f)
    model.reinit()

if config["app_kwargs"]["use_cuda"] and torch.cuda.is_available():
    model.to_cuda()
    model.device = "cuda"
else:
    model.to_cpu()
    model.device = "cpu"

# title
st.title(f"")

predict_tab, eval_tab = st.tabs(["Predict", "Evaluate"])

with predict_tab:
    # show data
    f = px.scatter(
        pd.DataFrame(),
        x="x",
        y="y",
        color="group",
    )
    st.plotly_chart(f, theme=None)

with eval_tab:
    st.text("Confusion Matrix")
    cmf = le.confusion_matrix_plot(test_enc, probs, manufacturers)
    st.pyplot(cmf)
