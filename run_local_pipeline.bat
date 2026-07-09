@echo off
REM =========================================================
REM Run CellViT Pipeline Locally (Windows)
REM Assumes you have a Python environment.
REM =========================================================

echo Starting CellViT Local Pipeline...

REM 1. install local package if possible
REM if exist setup.py pip install -e .
REM if exist pyproject.toml pip install -e .

REM 2. Check and Install Dependencies
python -c "import tqdm" >nul 2>&1 || (echo Installing tqdm... & pip install tqdm)
python -c "import ujson" >nul 2>&1 || (echo Installing ujson... & pip install ujson)
python -c "import shapely" >nul 2>&1 || (echo Installing shapely... & pip install shapely)
python -c "import pandas" >nul 2>&1 || (echo Installing pandas... & pip install pandas)
python -c "import scipy" >nul 2>&1 || (echo Installing scipy... & pip install scipy)
python -c "import skimage" >nul 2>&1 || (echo Installing scikit-image... & pip install scikit-image)
python -c "import matplotlib" >nul 2>&1 || (echo Installing matplotlib... & pip install matplotlib)
python -c "import einops" >nul 2>&1 || (echo Installing einops... & pip install einops)
python -c "import timm" >nul 2>&1 || (echo Installing timm... & pip install timm)

REM Check Numba/Numpy compatibility
python -c "import numba" >nul 2>&1 || (
    echo Numba import failed. Downgrading NumPy...
    pip install "numpy<2.3"
)

REM 3. Check for specific parallel/gpu libs
python -c "import cupy" >nul 2>&1 || (
    echo CupY missing. Attempting to install cupy-cuda12x...
    pip install cupy-cuda12x
)

REM 4. Run Pipeline
python run_full_pipeline.py ^
    --input_dir ./data/Tiles ^
    --output_base ./results ^
    --model ./models/CellViT-SAM-H-x40-AMP-001.pth ^
    --batch_size 4 ^
    --min_tumor_cells 2

echo Pipeline finished! Check ./results folder.
pause
