FROM ubuntu:latest

# Set the working directory inside the container.
WORKDIR /app

# Install necessary system dependencies.
# `wget`, `bzip2`, `ca-certificates` are needed for Miniconda.
# `build-essential`, `g++`, `make`, `libsfml-dev`, `libjsoncpp-dev` are for C++ compilation.
RUN apt-get update && \
    apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    gnupg \
    build-essential \
    g++ \
    make \
    libsfml-dev \
    libjsoncpp-dev && \
    rm -rf /var/lib/apt/lists/*


# Download and install Miniconda.
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Add Miniconda to the system's PATH for all subsequent commands.
# This ensures that `conda` and Python executables are available.
ENV PATH=/opt/conda/bin:$PATH

# Change working directory back to /app
WORKDIR /app

# Install libmamba and set it as the default solver for conda operations to speed up
# the creation of the environment. Also, set channel_priority to strict.
RUN conda install -y libmamba && \
    conda config --set solver libmamba && \
    conda config --set channel_priority strict

# Copy environment.yaml file into the container.
COPY environment.yaml .

# Create the conda environment as specified in environment.yaml.
RUN conda env create -f environment.yaml

# Activate the newly created Conda environment for subsequent commands
ARG CONDA_ENV_NAME=spaceborne
RUN bash -c "source $(conda info --base)/etc/profile.d/conda.sh && conda activate ${CONDA_ENV_NAME}"

# Copy the entire contents of current local directory (Spaceborne code)
# into the `/app` directory inside the container.
COPY . .

# Define the default command to run when the container starts.
CMD ["conda", "run", "--no-capture-output", "-n", "spaceborne", "python", "main.py"]