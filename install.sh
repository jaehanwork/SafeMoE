#!/bin/bash

conda env create -f environments.yml -n SafeMoE
conda activate SafeMoE
cd vllm-0.10.0
VLLM_USE_PRECOMPILED=1 pip install -e .
cd ..