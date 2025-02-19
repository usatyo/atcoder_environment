# Pythonのイメージ
FROM ubuntu:22.04
USER root

ENV TZ=Asia/Tokyo
ENV DEBIAN_FRONTEND=noninteractive

# common
RUN apt-get update && \
    apt-get install -y time tzdata tree git curl

# language
RUN apt-get update && \
    apt-get install -y build-essential gcc-12 g++-12 python3.10 python3-pip pypy3

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 30 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 30 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 30 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 30 && \
    update-alternatives --install /usr/bin/pypy pypy /usr/bin/pypy3 30

# c++ setting
RUN git clone https://github.com/atcoder/ac-library.git /lib/ac-library
ENV CPLUS_INCLUDE_PATH=/lib/ac-library
ENV CXX=g++-12

# python setting
RUN pip install git+https://github.com/not522/ac-library-python
RUN pypy3 -m pip install git+https://github.com/not522/ac-library-python
RUN pip install Pillow

COPY . /work
WORKDIR /work
