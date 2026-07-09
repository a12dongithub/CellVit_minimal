#!/bin/bash
set -e

echo "=========================================================="
echo "   Database CuPy Cloud Fixer"
echo "=========================================================="

PY=python3
if [ -f /opt/conda/bin/python ]; then PY=/opt/conda/bin/python; fi
if [ -f /root/miniconda3/bin/python ]; then PY=/root/miniconda3/bin/python; fi

echo "Using Python: $PY"

# 1. Clean Slate (Uninstall potential conflicts)
echo "[1/4] Cleaning environment..."
$PY -m pip uninstall -y cupy cupy-cuda11x cupy-cuda12x cupy-cuda13x torch torchvision torchaudio nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 || true

# 2. Install PyTorch (CU128) & CuPy (CUDA12x)
echo "[2/4] Installing PyTorch (cu128) and CuPy-CUDA12x..."
$PY -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
$PY -m pip install "cupy-cuda12x>=13.0.0"

# 3. Create Verification Script
cat << 'EOF' > verify_cupy.py
import sys
import os
import ctypes

def find_library(name):
    # Try filtering standard paths
    paths = os.environ.get('LD_LIBRARY_PATH', '').split(':')
    paths += ['/usr/lib', '/usr/local/lib', '/usr/lib/x86_64-linux-gnu']
    
    # Also check site-packages for nvidia packages
    import site
    for site_pkg in site.getsitepackages():
        nvidia_path = os.path.join(site_pkg, 'nvidia')
        if os.path.exists(nvidia_path):
            for root, dirs, files in os.walk(nvidia_path):
                if name in files:
                    return os.path.join(root, name)
        
        # Check torch/lib
        torch_path = os.path.join(site_pkg, 'torch', 'lib')
        if os.path.exists(torch_path):
             if name in os.listdir(torch_path):
                 return os.path.join(torch_path, name)
    
    return None

print(f"Python: {sys.version}")
try:
    import torch
    print(f"PyTorch: {torch.__version__} (CUDA: {torch.version.cuda})")
except ImportError:
    print("PyTorch not installed.")

print("\n--- Searching for libnvrtc ---")
lib_name = "libnvrtc.so.12"
found_path = find_library(lib_name)
if found_path:
    print(f"Found {lib_name} at: {found_path}")
    # Attempt to load it explicitly
    try:
        ctypes.CDLL(found_path)
        print("Successfully loaded via ctypes.")
    except Exception as e:
        print(f"Failed to load via ctypes: {e}")
else:
    print(f"Could NOT find {lib_name} in standard locations.")

print("\n--- Testing CuPy ---")
try:
    import cupy as cp
    print(f"CuPy Version: {cp.__version__}")
    
    # Simple operation
    x = cp.array([1, 2, 3])
    y = cp.array([4, 5, 6])
    z = x + y
    print(f"Basic Add Check: {z}")
    
    # Kernel compilation check (Triggers NVRTC)
    print("Testing NVRTC compilation...")
    kernel = cp.ElementwiseKernel(
        'float32 x, float32 y',
        'float32 z',
        'z = x * y',
        'multiply_kernel'
    )
    res = kernel(cp.array([2.], dtype=cp.float32), cp.array([3.], dtype=cp.float32))
    print(f"Kernel Result: {res}")
    print("\nSUCCESS: CuPy is working correctly!")
    
except Exception as e:
    print(f"\nFAILURE: {e}")
    sys.exit(1)
EOF

# 4. Configure Environment & Run
echo "[3/4] Auto-Configuring Paths..."

# Find Lib Path
NVRTC_PATH=$($PY -c "import os, site; 
found = ''
for p in site.getsitepackages():
    targ = os.path.join(p, 'nvidia', 'cuda_nvrtc', 'lib'); 
    if os.path.exists(targ): found = targ; break
if not found:
    for p in site.getsitepackages():
        targ = os.path.join(p, 'torch', 'lib');
        if os.path.exists(targ): found = targ; break

print(found)")

if [ ! -z "$NVRTC_PATH" ]; then
    echo "Files located at: $NVRTC_PATH"
    export LD_LIBRARY_PATH=$NVRTC_PATH:$LD_LIBRARY_PATH
    echo "Updated LD_LIBRARY_PATH"
else
    echo "WARNING: Could not auto-detect NVRTC library path inside Python packages."
fi

echo "[4/4] Verifying Fix..."
$PY verify_cupy.py
