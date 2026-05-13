# Use official R base image
FROM r-base:4.5.1

# Set environment variables to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies needed for R packages
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgit2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    git \
    curl \
    tmux \
    python3-pip \
    python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install notebook jupyterlab

ENV PATH="/opt/venv/bin:$PATH"

# Install pak and other R packages
RUN R -e "install.packages(c( \
      'Seurat', \
      'ggplot2', \
      'UCell', \
      'patchwork', \
      'Matrix', \
      'RcppML', \
      'RColorBrewer', \
      'viridis', \
      'remotes', \
      'rhdf5' \
    ), repos='https://cloud.r-project.org/')"

RUN R -e "remotes::install_github('carmonalab/GeneNMF')"

RUN R -e "install.packages('pak', repos='https://cloud.r-project.org')" && \
    R -e "pak::pak('scverse/anndataR')" 

RUN R -e "install.packages(c('msigdbr'), repos='https://cloud.r-project.org')"

RUN R -e 'if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager"); BiocManager::install("fgsea")'
RUN R -e 'BiocManager::install("UCell")'


# Set working directory
WORKDIR /workspace

RUN R -e "install.packages('IRkernel')"
# Register Jupyter kernel with a custom name
RUN R -e "IRkernel::installspec(user = FALSE, name = 'genenmf', displayname = 'R (GeneNMF)')"

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Default command
CMD ["R"]
