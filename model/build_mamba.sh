#!/bin/bash
source ~/slothc_venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="12.0"
export MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE
echo "=== nvcc ==="; nvcc --version | grep release
echo "=== building causal-conv1d ==="
pip install -v --no-build-isolation causal-conv1d 2>&1 | tail -8
echo "=== building mamba-ssm ==="
pip install -v --no-build-isolation mamba-ssm 2>&1 | tail -8
python3 -c "from mamba_ssm import Mamba2; print(\"MAMBA2_OK\")" 2>&1 | tail -2
