Installation
============

Quick install via pip
---------------------

.. code-block:: bash

   pip install kaleidocell

For GPU acceleration (Linux/HPC), install PyTorch with the appropriate CUDA
wheel **before** installing kaleidocell:

.. code-block:: bash

   # Replace cu124 with your CUDA version (run nvidia-smi to check)
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   pip install kaleidocell

Prerequisites
-------------

`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or Anaconda is
required for Options A–C.  Docker is required for Options D–E.

Option A — Linux / HPC server (GPU, CUDA 12.4)
-----------------------------------------------

.. code-block:: bash

   # Initialize conda (only needed if not already in your shell)
   source "$(conda info --base)/etc/profile.d/conda.sh"

   # Define paths (edit these)
   PROJECT_DIR=/path/to/kaleidocell
   ENV_DIR=$PROJECT_DIR/.environments/kaleidocell_env

   conda create -p "$ENV_DIR" python=3.11 -y
   conda activate "$ENV_DIR"

   # Install GPU-enabled PyTorch — replace cu124 with your CUDA version
   # (run nvidia-smi to check: cu121 = 12.1, cu118 = 11.8, …)
   pip install torch --index-url https://download.pytorch.org/whl/cu124

   cd "$PROJECT_DIR"
   pip install -e .
   pip install ipykernel
   python -m ipykernel install --user --name=kaleidocell_env --display-name "kaleidocell_env"

Option B — Linux / HPC via environment.yml
-------------------------------------------

.. code-block:: bash

   source "$(conda info --base)/etc/profile.d/conda.sh"

   PROJECT_DIR=/path/to/kaleidocell
   ENV_DIR=$PROJECT_DIR/.environments/kaleidocell_env

   cd "$PROJECT_DIR"
   conda env create -p "$ENV_DIR" -f environment.yml
   conda activate "$ENV_DIR"

   python -m ipykernel install --user --name=kaleidocell_env --display-name "kaleidocell_env"

**To reactivate later:**

.. code-block:: bash

   source "$(conda info --base)/etc/profile.d/conda.sh"
   conda activate "$ENV_DIR"

Option C — Mac (Apple Silicon M1/M2/M3)
-----------------------------------------

PyTorch ships with MPS (Metal Performance Shaders) support out of the box —
no separate CUDA wheel is needed.

.. code-block:: bash

   source "$(conda info --base)/etc/profile.d/conda.sh"

   PROJECT_DIR=/path/to/kaleidocell
   ENV_DIR=$PROJECT_DIR/.environments/kaleidocell_env

   conda create -p "$ENV_DIR" python=3.11 -y
   conda activate "$ENV_DIR"

   pip install torch   # MPS support included by default

   cd "$PROJECT_DIR"
   pip install -e .
   pip install ipykernel
   python -m ipykernel install --user --name=kaleidocell_env --display-name "kaleidocell_env"

**To reactivate later:**

.. code-block:: bash

   source "$(conda info --base)/etc/profile.d/conda.sh"
   conda activate "$ENV_DIR"

Option D — Docker (terminal)
-----------------------------

A pre-built image with all dependencies (Python 3.11, PyTorch CUDA 12.4,
kaleidocell) is available on Docker Hub as ``hdsu/kaleidocell_env:latest``.

.. code-block:: bash

   # Pull the image
   docker pull hdsu/kaleidocell_env:latest

   # Run interactively — mount your data directory into /workspace/data
   docker run --gpus all -it --rm \
       -v /path/to/your/data:/workspace/data \
       hdsu/kaleidocell_env:latest

   # Run Jupyter Notebook (open http://localhost:8888 in your browser)
   docker run --gpus all -it --rm \
       -v /path/to/your/data:/workspace/data \
       -p 8888:8888 \
       hdsu/kaleidocell_env:latest \
       jupyter notebook --ip=0.0.0.0 --no-browser --allow-root \
           --notebook-dir=/workspace/data

.. note::
   GPU support requires the
   `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_.
   Omit ``--gpus all`` to run on CPU only.

Option E — Docker (VS Code Dev Container)
------------------------------------------

Dev Containers open the project inside the Docker image directly from VS Code,
with full IntelliSense, debugging, and Jupyter support.

**Setup**

1. Install the
   `Dev Containers extension <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers>`_
   in VS Code.
2. Create a ``.devcontainer/kaleidocell/`` folder at the root of your project and
   place the following ``devcontainer.json`` inside it
   (a copy is provided in ``docker/devcontainer.json``):

.. code-block:: json

   {
       "name": "kaleidocell",
       "image": "hdsu/kaleidocell_env:latest",
       "runArgs": [
           "--name", "kaleidocell",
           "--gpus", "all"
       ],
       "workspaceMount": "source=${localWorkspaceFolder},target=/workspace,type=bind",
       "workspaceFolder": "/workspace",
       "customizations": {
           "vscode": {
               "extensions": [
                   "ms-python.python",
                   "ms-python.vscode-pylance",
                   "ms-toolsai.jupyter",
                   "ms-python.black-formatter"
               ]
           }
       }
   }

3. Open VS Code, press **F1** → ``Dev Containers: Reopen in Container``.

Verify GPU acceleration
-----------------------

.. code-block:: python

   import torch
   print("CUDA:", torch.cuda.is_available())
   print("MPS: ", torch.backends.mps.is_available())
