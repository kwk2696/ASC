#!/usr/bin/env bash
# Run the VirtualHome Unity simulator headless. Clients connect to http://<host>:$PORT,
# which is what --env_ip / --env_port in scripts/asc/train.py expect (default 8080).
PORT=${PORT:-8080}
VH_EXEC=${VH_EXEC:-$HOME/virtualhome/linux_exec.v2.3.0.x86_64}
xvfb-run --auto-servernum --server-args="-screen 0 640x480x24" \
    "$VH_EXEC" -batchmode -http-port="$PORT"
