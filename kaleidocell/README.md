# kaleidocell

---

## Installation

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda

---

### Option A — Installation using conda

```bash
# Initialize conda (only needed if not already in your shell)
source "$(conda info --base)/etc/profile.d/conda.sh"

# Define paths (edit these)
PROJECT_DIR=/path/to/kaleidocell
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

### Option B — Linux / HPC via environment.yml

```bash
# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Define paths
PROJECT_DIR=/path/to/kaleidocell
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

# Define paths (edit these)
PROJECT_DIR=/path/to/kaleidoCell/kaleidocell
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

# Run interactively — mount your data directory into /workspace/data
docker run --gpus all -it --rm \
    -v /path/to/your/data:/workspace/data \
    hdsu/kaleidocell_env:latest

# Run Jupyter Lab and open the quickstart example (open http://localhost:8888 in your browser)
docker run --gpus all -it --rm \
    -v /path/to/your/data:/workspace/data \
    -v /path/to/kaleidocell_v1/examples:/workspace/examples \
    -p 8888:8888 \
    hdsu/kaleidocell_env:latest \
    jupyter lab --ip=0.0.0.0 --no-browser --allow-root /workspace/examples/01_quickstart.ipynb
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

## Quick start

```python
import kaleidocell
import scanpy as sc

adata = sc.read_h5ad("your_data.h5ad")

# 1. Run NMF on every sample
results_nmf, _ = kaleidocell.multi_sample_nmf(
    adata,
    batch_key="sample" # Replace this with the obs of interest!
)

# 2. Derive consensus meta-programs
results_mp = kaleidocell.derive_nmf_metaprograms(results_nmf)

# 3. Score cells
mp_scores = kaleidocell.compute_mp_scores(results_mp, adata)

# 4. Generate HTML report
kaleidocell.get_html(
    results_mp, adata,
    mp_scores=mp_scores,
    obs=["Treatment", "donor"], # Replace this with obs of interest.
    output_path="results/",
)
```

---

## Output files produced by `get_html`

All files are written to the same directory as the HTML file.

| File | Description | Always written |
|------|-------------|:--------------:|
| `results.html` | Self-contained tabbed HTML report | ✓ |
| `genes.csv` | Long-format gene table: `gene`, `mp`, `score` | ✓ |
| `heatmap.pdf` | Cosine-similarity matrix | ✓ |
| `umap_scores.pdf` | MP scores on UMAP | requires `mp_scores` |
| `gsea_{label}.csv` | Significant GSEA terms per GMT file | when GSEA finds results |
| `gsea_{label}_{MP}.pdf` | GSEA bar plot per MP per GMT file | when GSEA finds results |
| `violins_{obs_key}.pdf` | Violin plots per MP per obs key | requires `obs` |

---

## Bundled gene-set files

```python
import kaleidocell
print(kaleidocell.files)   # lists all bundled files with descriptions
```

| File | Content |
|------|---------|
| `h.all.v2026.1.Hs.symbols.gmt` | MSigDB Hallmarks (50 gene sets) |
| `c5.go.bp.v2026.1.Hs.symbols.gmt` | GO Biological Process |
| `c5.go.cc.v2026.1.Hs.symbols.gmt` | GO Cellular Component |
| `c5.go.mf.v2026.1.Hs.symbols.gmt` | GO Molecular Function |
| `c6.all.v2026.1.Hs.symbols.gmt` | Oncogenic signatures |
| `c7.all.v2026.1.Hs.symbols.gmt` | Immunologic signatures |
| `c8.all.v2026.1.Hs.symbols.gmt` | Cell type signatures |
| `c9.all.v2026.1.Hs.symbols.gmt` | Cancer gene sets |
| `hgnc_ensembl_translation.txt` | Ensembl ↔ HGNC symbol table |

Short filenames are resolved automatically — no path needed:

```python
kaleidocell.run_gsea_pipeline(results_mp, from_file=["h.all.v2026.1.Hs.symbols.gmt"])
```

---

## Documentation

Build locally:

```bash
pip install sphinx pydata-sphinx-theme nbsphinx pandoc
cd docs && sphinx-build -b html . _build/html
# open docs/_build/html/index.html
```

---

## License

MIT
