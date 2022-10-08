FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-devel

# conda env in docker ref: https://qiita.com/junkor-1011/items/cd7c0e626aedc335011d

# system update & package install
RUN apt-get clean && \
    apt-get -y update && \
    apt-get install -y --no-install-recommends \
    unzip bzip2 \
    openssl libssl-dev \
    curl wget \
    ca-certificates \
    locales \
    bash \
    sudo \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# set up enviroment for conda
ENV CONDA_DIR=/opt/conda \
    CONDA_TMP_DIR=/tmp/conda \
    SHELL=/bin/bash
# to avoid the error: CondaValueError: prefix already exists: /opt/conda
RUN rm -rf $CONDA_DIR
RUN mkdir -p $CONDA_DIR && \
    mkdir -p $CONDA_TMP_DIR
    
# import yaml
ARG CONDA_YAML="./environment.yaml"
COPY $CONDA_YAML /tmp/conda_packages.yml
WORKDIR /work
COPY . .

# create conda env using miniconda
ARG MINICONDA_VERSION=py37_4.8.3-Linux-x86_64
ARG MINICONDA_MD5=751786b92c00b1aeae3f017b781018df
ENV PATH=${CONDA_DIR}/bin:$PATH
RUN cd /tmp && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}.sh && \
    echo "${MINICONDA_MD5} *Miniconda3-${MINICONDA_VERSION}.sh" | md5sum -c - && \
    /bin/bash Miniconda3-${MINICONDA_VERSION}.sh -f -b -p $CONDA_TMP_DIR && \
    rm Miniconda3-${MINICONDA_VERSION}.sh && \
    $CONDA_TMP_DIR/bin/conda env create -f /tmp/conda_packages.yml -p $CONDA_DIR && \
    rm -rf /home/.cache/* && \
    rm -rf $CONDA_TMP_DIR/*

# to write outputs and share with the host user
RUN chown 1000:1000 /home
