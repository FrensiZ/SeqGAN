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

# Create necessary directories and set permissions
RUN mkdir -p ${WORKING_DIR}/outputs ${WORKING_DIR}/logs ${WORKING_DIR}/saved_models/disc_outputs && \
    chmod -R 775 ${WORKING_DIR}

# Copy requirements first for better layer caching
COPY requirements.txt ${WORKING_DIR}/

# Install packages using the requirements file
RUN pip3 install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r requirements.txt

# Create user for better security
RUN useradd -u 3754 -g 100 -m myuser && \
    chown -R myuser:users ${WORKING_DIR}

USER myuser

# Copy the SeqGAN files
COPY --chown=myuser:users discriminator.py generator.py oracle.py disc_train.py parallel_disc_train.py ${WORKING_DIR}/

# Copy the target_params.pkl to the saved_models directory
COPY --chown=myuser:users saved_models/ ${WORKING_DIR}/saved_models/

# Set environment variables
ENV PYTHONPATH="${WORKING_DIR}"
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=2,3,4

# Default command set to run the parallel discriminator testing
CMD ["python3", "parallel_disc_train.py"]
