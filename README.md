# Ship Segmentation

This is a template repo for starting new ML projects

- [Setup](#setup)
- [App](#app)
- [Pipelines](#pipelines)
- [Library](#library)
- [EDA](#eda)
- [Next Steps](#next-steps)

## Setup

First clone the repo and download the data. 


*GPU Requirements* if you want to use GPU you must make sure you have your CUDA environment set up properly and you update the `requirements-cuda.txt` to point to the correct index based on your cuda driver.
I have cuda 11.8 so I point to that index.
See [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for details on available cuda installs.

### Docker

A Dockerfile is provided. 

```shell
# data dir must be present before building image
docker build -t container-name .
docker run \
    -d \
    --name container-name \
    --gpus all \
    -p 8501:8501 \
    -p 5000:5000 \
    container-name
docker exec -it container-name bash # will log you into shell, skip to `App` or `Pipelines` section for commands to run apps
```

### Local

Local was run on `python==3.10`.`

```shell
# make env
mkvirtualenv airid
# install reqs
pip install -r requirements.txt
pip install -r requirements-cuda.txt
# install model library (airid)
pip install -e .
```

## App

To view predictions and evaluation on the test set images call:

```shell
streamlit run app/app.py config/baseline.yaml
```

and navigate to the reported url.
The first time running it will likely need to download fasterrcnn weights.

To look at model training runs on MLFlow call:

```shell
mlflow server -h 0.0.0.0
```

and navigate to the reported url.

## Pipelines

**Training**

To train and save a model.
Run info will be logged to the local MLFlow directory.

```shell
python script/train.py config/baseline.yaml
```

**Predicting**

To generate predictions for the test set - these are used in the app.
Be sure to update the `[app_kwargs][run_id]` to the model you want to predict with.
A pretrained model is already specified.

```shell
python script/predict.py config/baseline.yaml
```

## Library

The library, housed in the `library/` directory contains all code for the model and dataset handling.

### Models

New models should be put in `library/model.py` and match the `BaseModel` class signature.
A config will also need to be developed for the new model.
See `config/baseline.yaml` for an example model config.

## EDA

My entrypoint for working on this project is found in the `notebooks` directory. 
Specifically `notebooks/eda.py` is where I took my first look at the data.

The main items of note:


## Next Steps


