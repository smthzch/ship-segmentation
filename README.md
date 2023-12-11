# Ship Segmentation

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

A Dockerfile is provided to run the shiny app.

```shell
# data dir must be present before building image
docker build -t ship-seg .
docker run \
    -d \
    --name ship-seg \
    --gpus all \
    -p 8000:8000 \
    ship-seg
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
shiny run app.py
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
