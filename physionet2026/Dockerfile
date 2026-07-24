FROM python:3.10.1-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install system dependencies and C++ compilers needed for LightGBM / XGBoost
RUN apt-get update && apt-get install -y build-essential libgomp1 && rm -rf /var/lib/apt/lists/*

## Include the following line if you have a requirements.txt file.
RUN pip install --no-cache-dir -r requirements.txt