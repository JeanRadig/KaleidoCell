# kaleidocell

---

## Installation

---

### Option A — Installation using conda

- Prerequisite: [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda

```bash
# Initialize conda (only needed if not already in your shell)
source "$(conda info --base)/etc/profile.d/conda.sh"

# Edit the PROJECT_DIR path such that it points to KaleidoCell/kaleidocell
PROJECT_DIR=/path/to/KaleidoCell/kaleidocell
ENV_DIR=$PROJECT_DIR/.environments/kaleidocell_env

# Create environment in the same folder as your project (remove -p and path if you want it at your conda place)
conda create -p "$ENV_DIR" python=3.11 -y
conda activate "$ENV_DIR"

# Install GPU-enabled PyTorch (example: CUDA 12.4)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install kaleidocell
cd "$PROJECT_DIR"
pip install -e .

# Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name=kaleidocell_env --display-name "kaleidocell_env"
```

> **CUDA version** — replace `cu124` with the tag matching your driver:
> `cu121` (12.1), `cu118` (11.8), etc.  Run `nvidia-smi` to check.

---

### Option B — installation using conda via environment.yml

```bash
# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Edit the PROJECT_DIR path such that it points to KaleidoCell/kaleidocell
PROJECT_DIR=/path/to/KaleidoCell/kaleidocell
ENV_DIR=$PROJECT_DIR/.environments/kaleidocell_env

cd "$PROJECT_DIR"

# Create environment from file
conda env create -p "$ENV_DIR" -f environment.yml
conda activate "$ENV_DIR"

# Jupyter kernel
python -m ipykernel install --user --name=kaleidocell_env --display-name "kaleidocell_env"
```

**To reactivate later:**

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
```

---

### Option C — Mac (Apple Silicon M1/M2/M3)

PyTorch ships with MPS (Metal Performance Shaders) support out of the box — no separate CUDA wheel is needed. Please note that GPU-acceleration on Mac is only twice as fast as CPU support. We do not recommend running KaleidoCell on Mac. Usage of a high-performance computing machine is recommended. This installation has been tested on a MacBook Pro with Apple M2 Pro chip, 10 cores, and 16 GB of memory.

```bash
# Initialize conda (if not already active in your shell)
source "$(conda info --base)/etc/profile.d/conda.sh"

# Edit the PROJECT_DIR path such that it points to KaleidoCell/kaleidocell
PROJECT_DIR=/path/to/KaleidoCell/kaleidocell
ENV_DIR=$PROJECT_DIR/.environments/kaleidocell_env

# Create environment
conda create -p "$ENV_DIR" python=3.11 -y
conda activate "$ENV_DIR"

# Install PyTorch (MPS support is included by default)
pip install torch

# Install kaleidocell
cd "$PROJECT_DIR"
pip install -e .

# Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name=kaleidocell_env --display-name "kaleidocell_env"
```

**To reactivate later:**

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
```

---

### Option D — Docker (terminal)

A pre-built image with all dependencies is available on Docker Hub. Please note that this installation has been tested on Linux x86_64 systems with NVIDIA GPU support (CUDA 12.4), and usage on Mac or personal computers is not recommended.

```bash
# Pull the image
docker pull hdsu/kaleidocell_env:latest

# Run interactively — mount your KaleidoCell folder into /workspace/KaleidoCell
docker run --gpus all -it --rm \
    -v /path/to/KaleidoCell:/workspace/KaleidoCell \
    hdsu/kaleidocell_env:latest

# Run Jupyter Lab and open the quickstart example (open http://localhost:8888 in your browser)
docker run --gpus all -it --rm \
    -v /path/to/KaleidoCell:/workspace/KaleidoCell \
    -p 8888:8888 \
    hdsu/kaleidocell_env:latest \
    jupyter lab --ip=0.0.0.0 --no-browser --allow-root /workspace/KaleidoCell/examples/01_quickstart.ipynb
```

---

### Option E — Docker (VS Code Dev Container)

Dev Containers let you open the project inside the Docker image directly from
VS Code with full IntelliSense, debugging, and Jupyter support. Please note that this installation has been tested on Linux x86_64 systems with NVIDIA GPU support (CUDA 12.4), and usage on Mac or personal computers is not recommended.

1. Install the
   [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
   in VS Code.
2. Create a `.devcontainer/kaleidocell/` folder at the root of your project.
3. Place the following `devcontainer.json` inside it:

```json
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
```

4. Open VS Code, press **F1** → `Dev Containers: Reopen in Container`.

> A ready-to-use `devcontainer.json` is also provided in `docker/devcontainer.json`.

---

## Verify GPU acceleration

```python
import torch
print("CUDA:", torch.cuda.is_available())
print("MPS: ", torch.backends.mps.is_available())
```

---