# Dockerfile used to create the docker

To create/adapt the docker, you need can adapt the following Dockerfile. You need to have a folder containing both the Dockerfile and the kaleidocell folder (KaleidoCell/kaleidocell), as it is going to be copy-pasted to the docker for within-docker installation. You should have following structure before building the docker:

- kaleidocell_env
    - Dockerfile 
    - kaleidocell

Then, you can run following commands to build the docker from scratch:

```python
cd path/to/kaleidocell_env
docker build -t kaleidocell_env .
```

Please find the Dockerfile below.

### Dockerfile

```python 
# Base image with Conda
FROM continuumio/miniconda3

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y \
    sudo \
    git \
    htop \
    less \
    tmux

# Create the Conda environment with Python 3.10
RUN conda create -n kaleidocell python=3.10 -y && \
    echo "conda activate kaleidocell" >> ~/.bashrc

# Install required pip packages inside the environment
RUN /bin/bash -c "source ~/.bashrc && \
    conda activate kaleidocell && \
    pip install \
        anndata==0.11.4 \
        matplotlib==3.10.8 \
        numpy==1.26.4 \
        pandas==2.3.3 \
        plotly==6.5.2 \
        rds2py==0.4.2 \
        scikit-learn==1.7.2 \
        scikit_learn_extra==0.3.0 \
        scipy==1.15.3 \
        seaborn==0.13.2 \
        setuptools>=59.5.0 \
        torch==2.10.0 \
        scanpy==1.11.5 \
        ipykernel==7.2.0 \
        neptune==1.14.0 \
        fastcluster==1.2.6 \
        jupyterlab \
        gseapy \
        statsmodels \
        tqdm \
        "

# Copy kaleidocell source and install it (baked into the conda env)
COPY kaleidocell /opt/kaleidocell
RUN /bin/bash -c "source ~/.bashrc && \
    conda activate kaleidocell && \
    pip install /opt/kaleidocell"

# Register the environment as a Jupyter kernel
RUN /bin/bash -c "source ~/.bashrc && \
    conda activate kaleidocell && \
    python -m ipykernel install --user --name=kaleidocell --display-name 'Python (kaleidocell)'"

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
EXPOSE 8888

CMD ["/bin/bash"]
```