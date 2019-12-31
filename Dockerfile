FROM tensorflow/tensorflow:latest-gpu-py3

ADD ./requirements.txt /requirements.txt

RUN pip3 install -r /requirements.txt

RUN mkdir -p /normative
WORKDIR /normative

