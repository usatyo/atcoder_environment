# Pythonのイメージ
FROM ubuntu:22.04
USER root

ENV TZ=Asia/Tokyo
ENV DEBIAN_FRONTEND=noninteractive

# common
RUN apt-get update
RUN apt-get install -y time
RUN apt-get install -y tzdata
RUN apt-get install -y tree
RUN apt-get install -y git
RUN apt-get install -y curl

# c++
RUN apt-get install -y build-essential
RUN apt-get install -y g++-11
RUN git clone https://github.com/atcoder/ac-library.git /lib/ac-library
ENV CPLUS_INCLUDE_PATH=/lib/ac-library
ENV CXX=g++-11

# python
RUN apt-get install -y python3.11 python3-pip pypy3

# 参照先の変更
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 30
RUN update-alternatives --install /usr/bin/pypy pypy /usr/bin/pypy3 30

COPY . /work
WORKDIR /work
