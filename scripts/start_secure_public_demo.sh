#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is required for a public demo tunnel."
  exit 1
fi

if [[ ! -f .env ]]; then
  echo "Create a .env file first. Public sharing requires APP_ACCESS_PASSWORD."
  exit 1
fi

set -a
source .env
set +a

if [[ -z "${APP_ACCESS_PASSWORD:-}" || "${APP_ACCESS_PASSWORD}" == "change-this-before-public-sharing" ]]; then
  echo "Refusing to start a public demo without a real APP_ACCESS_PASSWORD in .env."
  exit 1
fi

cleanup() {
  if [[ -n "${APP_PID:-}" ]] && kill -0 "${APP_PID}" >/dev/null 2>&1; then
    kill "${APP_PID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

streamlit run app/streamlit_app.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --server.headless true \
  --server.enableCORS true \
  --server.enableXsrfProtection true \
  >/tmp/rag-streamlit.log 2>&1 &
APP_PID=$!

sleep 3

echo "Starting password-protected public demo tunnel."
echo "Stop this process when the demo is over."
cloudflared tunnel --url http://127.0.0.1:8501
