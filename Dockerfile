FROM buildpack-deps:jammy
LABEL org.opencontainers.image.authors="Georg Krispel <georg.krispel@icg.tugraz.at>"

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    gcc \
    git \
    g++ \
    gdb \
    imagemagick \
    libatlas-base-dev \
    libboost-all-dev \
    libblas-dev \
    libeigen3-dev \
    liblapack-dev \
    libprotobuf-dev \
    libopencv-dev \
    python3-opencv \
    libopencv-contrib-dev \
    mlocate \
    unzip \
    zip \
    wget \
    valgrind \
    python3-clang-13 \
    ruby-full\
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* 
