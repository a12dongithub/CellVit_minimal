@echo off
echo Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing PyTorch with CUDA 12.1 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Installing requirements...
pip install -r requirements.txt

echo Installing CuPy for CUDA 12.x...
pip install cupy-cuda12x

echo Installing OpenSlide...
pip install openslide-python

echo Installing Snappy...
pip install python-snappy

echo Setup complete.
