# Source this file in a bash shell. Changes apply only to that shell.
source /home/fyf/anaconda3/etc/profile.d/conda.sh
conda activate /var/tmp/hbfsim-yjl/llama2-layer15-env-20261019-v1/env
export CUDA_HOME=/var/tmp/hbfsim-yjl/llama2-layer15-env-20261019-v1/cuda
export CUDAToolkit_ROOT="$CUDA_HOME"
export PATH="/var/tmp/hbfsim-yjl/llama2-layer15-env-20261019-v1/build-tools/cmake/data/bin:$CUDA_HOME/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
export HBFSIM_LLAMA2_CHECKPOINT=/home/yjl/yjl-r760/llama2-7b-hf-chat
