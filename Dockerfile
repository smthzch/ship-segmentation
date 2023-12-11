FROM python:3.10-slim

# this allows this container to be deployed in Google Cloud Run
# remove to run container locally
EXPOSE ${PORT}

RUN apt-get update
RUN apt-get install -y --no-install-recommends git # mlfow uses for tracking

WORKDIR /streamlit-app

COPY requirements.txt ./requirements.txt
COPY requirements-cuda.txt ./requirements-cuda.txt
RUN pip3 install -r requirements.txt
RUN pip3 install -r requirements-cuda.txt

COPY . ./
RUN pip3 install -e .

ENTRYPOINT uvicorn app:app --host 0.0.0.0 --port ${PORT}
