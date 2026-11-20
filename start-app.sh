#!/bin/bash
# Start the Zotero Chat web app (macOS / Linux).
#
#   ./start-app.sh              -> http://localhost:8502
#   PORT=8600 ./start-app.sh    -> another port
#
# The app only listens on localhost, so it is not reachable from other computers.
# On Windows, run instead:  venv\Scripts\streamlit run app.py --server.port 8502
cd "$(dirname "$0")"
PORT="${PORT:-8502}"

if [ ! -x venv/bin/streamlit ]; then
  echo "No virtual environment found. Set it up first (see README.md, 'Setup')."; exit 1
fi
if ! curl -s -m 3 "http://${OLLAMA_HOST:-localhost}:${OLLAMA_PORT:-11434}/api/tags" >/dev/null; then
  echo "Ollama isn't running. Start it (macOS: 'brew services start ollama' or open the Ollama app)."; exit 1
fi
exec venv/bin/streamlit run app.py --server.port "$PORT" --server.address localhost \
  --server.headless true --browser.gatherUsageStats false --client.toolbarMode minimal
