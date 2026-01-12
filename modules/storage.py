import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

PATHS = {
    "gov_laws": os.path.join(STORAGE_DIR, "gov_laws"),
    "user_docs": os.path.join(STORAGE_DIR, "user_docs"),
    "training_data": os.path.join(STORAGE_DIR, "training_data"),
    "scenarios": os.path.join(STORAGE_DIR, "scenarios"),
    "reports": os.path.join(STORAGE_DIR, "reports"),
}

def ensure_dirs():
    for p in PATHS.values():
        os.makedirs(p, exist_ok=True)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def load_json(path):
    if not os.path.exists(path): return None
    with open(path) as f:
        return json.load(f)
