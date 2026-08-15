#!/usr/bin/env bash
# Fresh-pod setup — every fix accumulated over the project's pod rebuilds:
#   - extract arg is --dst
#   - torch cu126 pair FORCE-REINSTALLED in BOTH venvs AFTER all other installs
#     (unsloth/lm-eval pull cu130 wheels that fail on older drivers)
#   - lm_eval installed via uv (uv venvs have no pip module)
#   - training-format chat template installed into merged dirs by the chain
# Run ON the pod after rsync. Needs HF_TOKEN exported.
set -euo pipefail
: "${HF_TOKEN:?export HF_TOKEN first}"

cd ~/skazna
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
command -v tmux >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq tmux)

[ -d .venv ] || uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -q \
  "transformers==5.14.1" torch peft accelerate sentencepiece protobuf \
  datasets python-dotenv "lm_eval>=0.4.8"

[ -d .venv-train ] || uv venv .venv-train --python 3.11
uv pip install --python .venv-train/bin/python -q \
  unsloth trl peft accelerate bitsandbytes sentencepiece protobuf python-dotenv

# cu126 pair LAST, in BOTH venvs — nothing may touch torch after this
for V in .venv .venv-train; do
  uv pip install --python $V/bin/python -q --reinstall \
    "torch==2.13.0" torchvision --index-url https://download.pytorch.org/whl/cu126
  $V/bin/python - << EOF
import torch
assert torch.cuda.is_available(), "$V: CUDA NOT AVAILABLE"
print("$V ok:", torch.__version__)
EOF
done

if [ ! -d gemma4-12b-text ]; then
  .venv/bin/python - << 'EOF'
from huggingface_hub import snapshot_download
import os
snapshot_download("google/gemma-4-12B", local_dir="gemma4-12b-full",
                  token=os.environ["HF_TOKEN"])
EOF
  .venv/bin/python train/extract_text_model.py \
    --src gemma4-12b-full --dst gemma4-12b-text
fi
echo "POD READY"
