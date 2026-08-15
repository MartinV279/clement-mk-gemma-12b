#!/usr/bin/env bash
# Rsync the repo to a rented pod. Usage: remote/sync.sh user@host [ssh_port]
# Excludes secrets (.env NEVER leaves this machine as a file — CLAUDE.md rule 6),
# data dumps, checkpoints, and local venvs.
set -euo pipefail

POD="${1:?usage: remote/sync.sh user@host [ssh_port]}"
PORT="${2:-22}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

rsync -avz --progress -e "ssh -p ${PORT}" \
  --exclude '.git/' \
  --exclude '.env' \
  --exclude '.venv/' --exclude '.venv-train/' \
  --exclude 'data/raw/' --exclude 'data/processed/' --exclude 'data/mixture/' --exclude 'data/downloads/' --exclude 'data/models/' \
  --exclude 'checkpoints/' --exclude 'outputs/' --exclude 'models/' \
  --exclude 'wandb/' --exclude 'runs/' \
  --exclude 'eval/results/' --exclude 'eval/generations/' \
  --exclude 'release/' \
  --exclude 'rag/index/' --exclude 'rag/' \
  --exclude 'llama.cpp/' --exclude '*.gguf' \
  --exclude 'unsloth_compiled_cache/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude 'logs/' --exclude '*.log' \
  "${REPO_ROOT}/" "${POD}:~/skazna/"

echo "Synced. On the pod: cd ~/skazna && bash remote/pod_setup.sh"
