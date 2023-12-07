FROM python:3.10-slim

RUN apt-get update
RUN apt-get install -y --no-install-recommends git # mlfow uses for tracking

WORKDIR /streamlit-app

COPY requirements.txt ./requirements.txt
COPY requirements-cuda.txt ./requirements-cuda.txt
RUN pip3 install -r requirements.txt
RUN pip3 install -r requirements-cuda.txt

COPY . ./
RUN pip3 install -e .

CMD ["tail", "-f", "/dev/null"]

# docker exec -it container-name bash
# streamlit run app/app.py config/baseline.yaml # to run app
# mlflow server -h 0.0.0.0 # to inspect mlflow runs
# python script/train.py config/baseline.yaml # to train a model
# python script/predict.py config/baseline.yaml # to make predictions (update config run_id with model to use)
