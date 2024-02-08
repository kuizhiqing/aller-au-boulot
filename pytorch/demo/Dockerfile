FROM python:3.8

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN apt-get update

RUN apt-get -y install git vim ripgrep net-tools curl wget

WORKDIR /

COPY demo.py .

