"""
launch.py — Desktop launcher for the Automotive Decision Intelligence Platform

HOW IT WORKS
------------
1. Starts Streamlit as a background subprocess (same as `streamlit run app.py`)
2. Waits until the local server is ready (polls localhost:8501)
3. Opens the app in a native desktop window using pywebview
   — No browser needed, looks and feels like a real desktop app
   — The Streamlit app code is completely unchanged

REQUIREMENTS
------------
    pip install pywebview

USAGE
-----
    python launch.py          # normal launch
    python launch.py --port 8502   # use a different port if 8501 is busy

"""

import sys
import time
import socket
import argparse
import subprocess
import threading
import os

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8501)
args = parser.parse_args()
PORT = args.port

# ── App metadata ──────────────────────────────────────────────────────────────
APP_TITLE  = "Automotive Decision Intelligence Platform"
APP_URL    = f"http://localhost:{PORT}"
ENTRY_POINT = os.path.join(os.path.dirname(__file__), "app.py")
ICON_PATH   = os.path.join(os.path.dirname(__file__), "assets", "icon.ico")

# ── Step 1: Start Streamlit subprocess ───────────────────────────────────────
print(f"[launcher] Starting Streamlit on port {PORT}…")
streamlit_proc = subprocess.Popen(
    [
        sys.executable, "-m", "streamlit", "run", ENTRY_POINT,
        "--server.port", str(PORT),
        "--server.headless", "true",       # don't open browser automatically
        "--server.runOnSave", "false",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "dark",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

# ── Step 2: Wait for server to be ready ──────────────────────────────────────
def _server_ready(port, timeout=30):
    """Poll localhost:port until it accepts connections or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.4)
    return False

print("[launcher] Waiting for Streamlit server…")
if not _server_ready(PORT):
    print(f"[launcher] ERROR: Server did not start within 30 s on port {PORT}")
    streamlit_proc.terminate()
    sys.exit(1)

print(f"[launcher] Server ready at {APP_URL}")

# ── Step 3: Open in native desktop window ─────────────────────────────────────
try:
    import webview
except ImportError:
    print(
        "\n[launcher] pywebview is not installed.\n"
        "Install it with:  pip install pywebview\n"
        f"Then re-run:      python launch.py\n\n"
        f"Alternatively, open your browser at {APP_URL}\n"
    )
    # Fallback: just open in the default browser
    import webbrowser
    webbrowser.open(APP_URL)
    try:
        streamlit_proc.wait()
    except KeyboardInterrupt:
        pass
    streamlit_proc.terminate()
    sys.exit(0)

def _on_closed():
    """Called when the desktop window is closed — shuts down Streamlit too."""
    print("[launcher] Window closed — shutting down Streamlit…")
    streamlit_proc.terminate()

print("[launcher] Opening desktop window…")
window = webview.create_window(
    title=APP_TITLE,
    url=APP_URL,
    width=1400,
    height=900,
    min_size=(1024, 700),
    resizable=True,
)

# Set window icon (works on Windows and Linux; macOS uses .icns)
try:
    webview.settings["ICON"] = ICON_PATH
except Exception:
    pass
window.events.closed += _on_closed

# Start the webview event loop (blocks until window is closed)
webview.start()

# Clean up if webview exits without triggering the event
streamlit_proc.terminate()
