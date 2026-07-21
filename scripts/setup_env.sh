#!/bin/bash
#
# Download miniconda + create sparse_emb, fasttext_env, eval conda envs.
# Fully non-interactive (auto-accepts all prompts).
#
# Usage:
#   bash scripts/setup_env.sh
#

set -e

MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
INSTALL_DIR="$HOME/miniconda3"

# 1. Download and install miniconda
if [ ! -d "$INSTALL_DIR" ]; then
    echo "=== Downloading miniconda ==="
    wget -q "$MINICONDA_URL" -O /tmp/miniconda.sh
    echo "=== Installing miniconda to $INSTALL_DIR ==="
    bash /tmp/miniconda.sh -b -p "$INSTALL_DIR"
    rm /tmp/miniconda.sh
    echo "=== Miniconda installed ==="
else
    echo "=== Miniconda already installed at $INSTALL_DIR ==="
fi

# 2. Init conda for current shell
eval "$($INSTALL_DIR/bin/conda shell.bash hook)"
conda init bash --quiet 2>/dev/null || true

# 3. Accept conda terms (suppress future prompts)
conda config --set auto_activate_base false
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# 4. Create sparse_emb env
if conda env list | grep -q "sparse_emb"; then
    echo "=== sparse_emb env already exists ==="
else
    echo "=== Creating sparse_emb env ==="
    conda create -n sparse_emb python=3.11 -y
fi

# 5. Install dependencies
echo "=== Installing dependencies in sparse_emb ==="
conda run -n sparse_emb pip install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

conda run -n sparse_emb pip install \
    transformers==5.9.0 datasets==4.8.5 accelerate==1.13.0 \
    scipy wandb pyarrow huggingface_hub openai

# 6. Create fasttext_env
if conda env list | grep -q "fasttext_env"; then
    echo "=== fasttext_env already exists ==="
else
    echo "=== Creating fasttext_env ==="
    conda create -n fasttext_env python=3.11 -y
fi

# 7. Install fasttext from source + numpy<2
echo "=== Installing fasttext from source in fasttext_env ==="
FASTTEXT_DIR="/tmp/fasttext_build"
if [ ! -d "$FASTTEXT_DIR" ]; then
    git clone https://github.com/facebookresearch/fastText.git "$FASTTEXT_DIR"
fi
conda run -n fasttext_env pip install "$FASTTEXT_DIR"
conda run -n fasttext_env pip install "numpy<2"
rm -rf "$FASTTEXT_DIR"

# 8. Create eval env
if conda env list | grep -q "eval"; then
    echo "=== eval env already exists ==="
else
    echo "=== Creating eval env ==="
    conda create -n eval python=3.11 -y
fi

# 9. Install lm-eval-harness from source
echo "=== Installing lm-eval-harness in eval env ==="
LM_EVAL_DIR="/tmp/lm_eval_build"
if [ ! -d "$LM_EVAL_DIR" ]; then
    git clone https://github.com/EleutherAI/lm-evaluation-harness.git "$LM_EVAL_DIR"
fi
conda run -n eval pip install "$LM_EVAL_DIR[hf,vllm,api]"
rm -rf "$LM_EVAL_DIR"

# 10. Verify
echo ""
echo "=== Verifying sparse_emb ==="
sleep 5
conda activate sparse_emb
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count(), 'Version:', torch.version.cuda)"
conda deactivate
echo "=== Done ==="
