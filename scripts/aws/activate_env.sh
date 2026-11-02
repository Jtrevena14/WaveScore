# Sourced by the other scripts: activates the Deep Learning AMI's PyTorch environment.
# Newer DLAMIs ship a venv at /opt/pytorch; older ones a conda env called "pytorch".
if [ -f /opt/pytorch/bin/activate ]; then
    source /opt/pytorch/bin/activate
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate pytorch
fi
