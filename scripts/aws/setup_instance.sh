#!/usr/bin/env bash
# One-time setup on a fresh EC2 GPU instance (Deep Learning AMI, Ubuntu 22.04).
#   bash setup_instance.sh            (clones into ~/WaveScore, branch dev)
# Re-running is safe: it pulls the repo and skips anything already in place.
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/Jtrevena14/WaveScore.git}
BRANCH=${BRANCH:-dev}
REPO_DIR=${REPO_DIR:-$HOME/WaveScore}
CHECKPOINT_URL="https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"

# Data folder on the EBS root volume (persists across stop/start).
sudo mkdir -p /data
sudo chown "$USER":"$USER" /data

# Swap: a safety net for the 16 GB of RAM on g4dn.xlarge while the patch arrays are built.
if ! swapon --show | grep -q /swapfile; then
    sudo fallocate -l 8G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    grep -q /swapfile /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" pull --ff-only origin "$BRANCH"
else
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

set +u  # the conda/venv activate scripts read unset variables
source scripts/aws/activate_env.sh
set -u
pip install -r requirements-train.txt

mkdir -p models
if [ ! -s models/piano_note_pedal.pth ]; then
    curl -L --fail -o models/piano_note_pedal.pth "$CHECKPOINT_URL"
fi

nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); assert torch.cuda.is_available()"
free -h
df -h / /opt/dlami/nvme 2>/dev/null || df -h /
echo "Setup done. Repo: $REPO_DIR"
