#!/usr/bin/env bash
# Make sure the Postgres + pgvector store that the codebase Q&A benchmark ingests into
# is reachable. Uses CODEBRAIN_DATABASE_URL if it answers, else starts docker-compose's db.
set -euo pipefail
cd "$(dirname "$0")/../.."

export CODEBRAIN_DATABASE_URL="${CODEBRAIN_DATABASE_URL:-postgresql://codebrain:codebrain@localhost:5432/codebrain}"
PY="${PY:-.venv-eval/bin/python}"

reachable() {
  "$PY" -c 'import os, psycopg; psycopg.connect(os.environ["CODEBRAIN_DATABASE_URL"], connect_timeout=3).close()' \
    2>/dev/null
}

if ! reachable; then
  if ! command -v docker >/dev/null; then
    echo "No database at $CODEBRAIN_DATABASE_URL and no docker to start one."
    echo "Install docker, or point CODEBRAIN_DATABASE_URL at a Postgres 16 with pgvector."
    exit 1
  fi
  # Without docker group membership the socket is root-only; use sudo for docker then.
  DOCKER=(docker)
  if ! docker info >/dev/null 2>&1; then
    if sudo docker info >/dev/null; then
      DOCKER=(sudo docker)
      echo "Using sudo for docker (add yourself to the docker group to skip this:"
      echo "  sudo usermod -aG docker \$USER, then log in again)"
    else
      echo "Cannot reach the docker daemon, even with sudo. Is it running? (sudo systemctl start docker)"
      exit 1
    fi
  fi
  echo "Starting Postgres + pgvector (docker compose service db) ..."
  "${DOCKER[@]}" compose up -d db
  for _ in $(seq 60); do reachable && break; sleep 2; done
  reachable || { echo "Postgres did not come up:"; "${DOCKER[@]}" compose logs --tail 30 db; exit 1; }
fi
"$(dirname "$PY")/codebrain" migrate >/dev/null
echo "Store ready at $CODEBRAIN_DATABASE_URL"
