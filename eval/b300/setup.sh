#!/usr/bin/env bash
# One-time setup on a fresh B300 VM. Assumes Ubuntu with the NVIDIA driver installed.
set -euo pipefail
cd "$(dirname "$0")/../.."

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv .venv-eval --python 3.12
# --torch-backend=auto picks the PyTorch CUDA build that matches the driver.
uv pip install --python .venv-eval --torch-backend=auto -e . -r eval/requirements.txt

.venv-eval/bin/python - <<'PY'
import torch

major, minor = torch.cuda.get_device_capability(0)
archs = torch.cuda.get_arch_list()
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, GPU sm_{major}{minor}, built for {archs}")
if not any(a.startswith(f"sm_{major}") for a in archs):
    raise SystemExit(
        f"This PyTorch build has no kernels for sm_{major}{minor}. Use the official vLLM "
        "container instead (see eval/README.md)."
    )
PY
echo "Setup done. Next: eval/b300/run.sh"
