#!/usr/bin/env bash

set -euo pipefail

echo "Ollama runtime:"
ollama ps
echo
echo "GPU status:"
nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader
echo
echo "GPU processes:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
