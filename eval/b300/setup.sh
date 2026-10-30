#!/usr/bin/env bash
# One-time setup on a fresh B300 VM. Assumes Ubuntu 24.04 with the NVIDIA driver installed.
# Run it as your own user, not with sudo: it asks for sudo itself where it needs it.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ "$(id -u)" = 0 ]; then
  echo "Run this as your normal user, not root: files it creates must stay yours."
  exit 1
fi
# A checkout touched by an earlier sudo run leaves root-owned files that later runs
# cannot write (eval-results, .venv-eval, the haystack copy).
if [ -n "$(find . -maxdepth 2 ! -user "$(id -un)" -print -quit 2>/dev/null)" ]; then
  echo "Some files here are not owned by $(id -un). Fix that, then rerun:"
  echo "  sudo chown -R $(id -un): $PWD"
  exit 1
fi

# System packages: Python headers and gcc (Triton compiles a helper with them), docker
# with compose for the Postgres + pgvector store, tmux so an SSH drop does not kill a run.
PKGS=()
dpkg -s python3.12-dev >/dev/null 2>&1 || PKGS+=(python3.12-dev)
command -v gcc >/dev/null || PKGS+=(build-essential)
command -v tmux >/dev/null || PKGS+=(tmux)
command -v git >/dev/null || PKGS+=(git)
command -v curl >/dev/null || PKGS+=(curl)
command -v docker >/dev/null || PKGS+=(docker.io)
docker compose version >/dev/null 2>&1 || sudo docker compose version >/dev/null 2>&1 \
  || PKGS+=(docker-compose-v2)
if [ "${#PKGS[@]}" -gt 0 ]; then
  echo "Installing ${PKGS[*]} (needs sudo) ..."
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${PKGS[@]}"
fi
sudo systemctl enable --now docker >/dev/null 2>&1 || true

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv .venv-eval --python 3.12
# Triton (under vLLM) compiles a C helper at startup and needs gcc and Python.h for that.
PY_INCLUDE="$(.venv-eval/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
if [ ! -f "$PY_INCLUDE/Python.h" ] || ! command -v gcc >/dev/null; then
  echo "Missing Python headers ($PY_INCLUDE/Python.h) or gcc. Install them, then rerun:"
  echo "  sudo apt-get install -y python3.12-dev build-essential"
  exit 1
fi
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
echo "Setup done. Next, inside tmux: eval/b300/qa.sh --qa-budget 8000 (or eval/b300/run.sh)"
