FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV WORKING_DIR=/workspace

RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    build-essential \
    nvidia-cuda-toolkit \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

WORKDIR ${WORKING_DIR}

RUN mkdir -p ${WORKING_DIR}/outputs ${WORKING_DIR}/logs ${WORKING_DIR}/saved_models && \
    chmod -R 775 ${WORKING_DIR}

COPY requirements.txt ${WORKING_DIR}/
RUN pip3 install --no-cache-dir -r requirements.txt

RUN useradd -u 3754 -g 100 -m myuser && \
    chown -R myuser:users ${WORKING_DIR}

USER myuser

# Copy the SeqGAN files
COPY --chown=myuser:users config.py discriminator.py generator.py oracle.py rollout.py pretrain.py test_discriminator.py ${WORKING_DIR}/

ENV PYTHONPATH="${WORKING_DIR}"
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all

# Default command - replace with your preferred entry point
CMD ["python3", "test_discriminator.py"]