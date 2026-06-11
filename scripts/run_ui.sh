#!/usr/bin/env bash
# scripts/run_ui.sh — clone (or update) deep-agents-ui and start the Next.js dev server.
#
# Companion to `langgraph dev` (which serves the travel-planner agent on
# http://127.0.0.1:2024). This script handles the UI half:
#
#   1. Clones https://github.com/langchain-ai/deep-agents-ui into ./ui/
#      (gitignored). If it already exists, runs `git pull` to refresh.
#   2. Installs Node deps with yarn.
#   3. Starts `yarn dev` (listens on http://localhost:3000).
#
# Usage:
#   # Terminal 1
#   uv run langgraph dev
#
#   # Terminal 2
#   ./scripts/run_ui.sh
#
# Then open http://localhost:3000 and click Settings to enter:
#     Deployment URL: http://127.0.0.1:2024
#     Assistant ID:   travel_planner

set -euo pipefail

REPO_URL="https://github.com/langchain-ai/deep-agents-ui.git"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
UI_DIR="$PROJECT_ROOT/ui"

# --- prerequisite checks -----------------------------------------------------
require() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: '$1' is required but not installed." >&2
        echo "Install it and retry. (See README §5 for setup.)" >&2
        exit 1
    fi
}

require git
require node
require yarn

# --- clone or update ---------------------------------------------------------
if [[ ! -d "$UI_DIR/.git" ]]; then
    echo ">>> Cloning deep-agents-ui into $UI_DIR ..."
    git clone "$REPO_URL" "$UI_DIR"
else
    echo ">>> Updating existing deep-agents-ui clone at $UI_DIR ..."
    git -C "$UI_DIR" pull --ff-only || {
        echo "WARN: 'git pull' failed; continuing with local copy." >&2
    }
fi

# --- install + run -----------------------------------------------------------
cd "$UI_DIR"

echo ">>> Installing UI dependencies (yarn install) ..."
yarn install --frozen-lockfile || yarn install

cat <<EOF

============================================================
 deep-agents-ui will start on http://localhost:3000

 In the UI, click Settings and enter:
   Deployment URL: http://127.0.0.1:2024
   Assistant ID:   travel_planner

 Make sure 'uv run langgraph dev' is running in another shell.
============================================================
EOF

exec yarn dev
