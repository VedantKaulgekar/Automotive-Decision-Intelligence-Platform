"""
launch.py — Desktop launcher for the Automotive Decision Intelligence Platform

HOW IT WORKS
------------
1. Starts Streamlit as a background subprocess (same as `streamlit run app.py`)
2. Waits until the local server is ready (polls localhost:8501)
3. Opens the app in a native desktop window using pywebview

FIXES
-----
- PYTHONUTF8=1 forces UTF-8 I/O on Windows — prevents UnicodeEncodeError
  from emoji characters in print() statements (e.g. ✔ ❌ in rag_engine.py)
- PDF/file downloads are intercepted and handed to the system browser
  because pywebview's embedded WebView2 does not handle blob:// download
  URLs the same way a real browser does. Any <a download> click or
  Streamlit st.download_button is caught by a JS MutationObserver and
  opened via webbrowser.open() in the user's default browser instead.

REQUIREMENTS
------------
    pip install pywebview

USAGE
-----
    python launch.py               # normal launch
    python launch.py --port 8502   # use a different port if 8501 is busy
"""

import sys
import time
import socket
import argparse
import subprocess
import webbrowser
import os

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8501)
args = parser.parse_args()
PORT = args.port

# ── App metadata ───────────────────────────────────────────────────────────────
APP_TITLE   = "Automotive Decision Intelligence Platform"
APP_URL     = f"http://localhost:{PORT}"
ENTRY_POINT = os.path.join(os.path.dirname(__file__), "app.py")
ICON_PATH   = os.path.join(os.path.dirname(__file__), "assets", "icon.ico")

# ── Step 1: Start Streamlit subprocess ────────────────────────────────────────
print(f"[launcher] Starting Streamlit on port {PORT}...")

# PYTHONUTF8=1 — forces UTF-8 for all Python I/O in the subprocess on Windows.
# Without this, Windows defaults to cp1252 which cannot encode emoji characters
# (e.g. the checkmark and cross in rag_engine.py print statements), causing
# UnicodeEncodeError at module import time.
launch_env = os.environ.copy()
launch_env["PYTHONUTF8"]        = "1"
launch_env["PYTHONIOENCODING"]  = "utf-8"

streamlit_proc = subprocess.Popen(
    [
        sys.executable, "-m", "streamlit", "run", ENTRY_POINT,
        "--server.port",              str(PORT),
        "--server.headless",          "true",
        "--server.runOnSave",         "false",
        "--browser.gatherUsageStats", "false",
        "--theme.base",               "dark",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    env=launch_env,
)

# ── Step 2: Wait for server to be ready ───────────────────────────────────────
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

print("[launcher] Waiting for Streamlit server...")
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
    webbrowser.open(APP_URL)
    try:
        streamlit_proc.wait()
    except KeyboardInterrupt:
        pass
    streamlit_proc.terminate()
    sys.exit(0)


# ── Download handler ──────────────────────────────────────────────────────────
# pywebview's embedded WebView2 / WebKit does not handle blob:// download URLs
# the same way a real browser does. Streamlit's st.download_button generates
# a hidden <a href="..." download> element and programmatically clicks it.
# WebView2 intercepts this and either does nothing or crashes silently.
#
# Fix: expose a Python function to JS via the webview API bridge, then inject
# a MutationObserver that watches for any <a download> element being added or
# clicked. When detected, we extract the href and pass it to Python, which
# opens it in the user's real default browser where downloads work normally.

class DownloadAPI:
    """Exposed to JavaScript as window.pywebview.api"""

    def open_url(self, url):
        """Open a URL in the system default browser (handles downloads)."""
        if url and (url.startswith("http://") or url.startswith("blob:")):
            # For blob: URLs convert to the Streamlit download endpoint
            # For http:// URLs (Streamlit media/download endpoints) open directly
            target = url if url.startswith("http://") else APP_URL
            webbrowser.open(target)

    def open_download(self, path):
        """Open a Streamlit /_stcore/... download URL in the real browser."""
        if path:
            full_url = APP_URL + path if path.startswith("/") else path
            webbrowser.open(full_url)


# JavaScript injected after every page load.
# Intercepts Streamlit download buttons and routes them through the Python API.
DOWNLOAD_JS = """
(function() {
    // Intercept any existing or future <a download> elements
    function interceptDownloads() {
        document.querySelectorAll('a[download], a[href*="/_stcore/"]').forEach(function(a) {
            if (a._intercepted) return;
            a._intercepted = true;
            a.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                var href = a.getAttribute('href') || '';
                if (window.pywebview && window.pywebview.api) {
                    window.pywebview.api.open_download(href);
                }
            }, true);
        });
    }

    // Run on load
    interceptDownloads();

    // Watch for Streamlit dynamically adding download buttons
    var observer = new MutationObserver(function(mutations) {
        interceptDownloads();
    });
    observer.observe(document.body, { childList: true, subtree: true });
})();
"""

def _on_loaded(window):
    """Inject download interceptor after every page navigation."""
    try:
        window.evaluate_js(DOWNLOAD_JS)
    except Exception:
        pass

def _on_closed():
    """Shut down Streamlit when the window closes."""
    print("[launcher] Window closed -- shutting down Streamlit...")
    streamlit_proc.terminate()


# ── Create and start the window ───────────────────────────────────────────────
print("[launcher] Opening desktop window...")

api    = DownloadAPI()
window = webview.create_window(
    title=APP_TITLE,
    url=APP_URL,
    width=1400,
    height=900,
    min_size=(1024, 700),
    resizable=True,
    js_api=api,                    # expose DownloadAPI as window.pywebview.api
)

# Set window icon
try:
    webview.settings["ICON"] = ICON_PATH
except Exception:
    pass

window.events.closed  += _on_closed
window.events.loaded  += lambda: _on_loaded(window)

# Start the webview event loop (blocks until window is closed)
webview.start()

# Clean up if webview exits without triggering the closed event
streamlit_proc.terminate()