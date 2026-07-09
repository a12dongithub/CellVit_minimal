#!/bin/bash
set -e

echo "============================================================"
echo "   CellViT Cloud Batch Processing Pipeline"
echo "============================================================"

ENV_DIR="$(pwd)/env"

# Initialize conda so 'conda activate' works in the script
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "$ENV_DIR" ]; then
    echo "Creating local conda environment in: $ENV_DIR"
    conda create -p "$ENV_DIR" python=3.10 -y
    conda activate "$ENV_DIR"
    
    echo "Installing PyTorch and CuPy..."
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
    python -m pip install "cupy-cuda12x>=13.0.0"
    
    echo "Installing other dependencies..."
    python -m pip install einops timm shapely ujson tqdm scikit-image scipy matplotlib pandas colorama "numpy<2.0" numba Pillow ray opencv-python-headless openslide-python openslide-bin
else
    echo "Activating existing local conda environment: $ENV_DIR"
    conda activate "$ENV_DIR"
fi

PY=python
echo "Using Python: $PY ($($PY --version 2>&1))"

# Export PYTHONPATH so modules inside cellvit can be found
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Auto-configure LD_LIBRARY_PATH for NVIDIA libraries (Fixes libcudnn_graph.so issue)
export LD_LIBRARY_PATH=$($PY -c "import os, site; print(':'.join([os.path.join(p, 'nvidia', d, 'lib') for p in site.getsitepackages() for d in os.listdir(os.path.join(p, 'nvidia')) if os.path.exists(os.path.join(p, 'nvidia', d, 'lib'))] + [os.environ.get('LD_LIBRARY_PATH', '')]))")

echo ""
echo "Running process_cloud_wsis.py with arguments: $@"
echo ""

$PY process_cloud_wsis.py "$@"

echo "============================================================"
echo "   Batch Processing Complete"
echo "============================================================"
