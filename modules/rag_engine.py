import os
import pickle
import faiss
import numpy as np
import threading
from pypdf import PdfReader

try:
    from thefuzz import fuzz
except ImportError:
    from fuzzywuzzy import fuzz

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(__file__))
STORAGE_DIR   = os.path.join(BASE_DIR, "storage")
GOV_LAWS_DIR  = os.path.join(STORAGE_DIR, "gov_laws")
USER_DOCS_DIR = os.path.join(STORAGE_DIR, "user_docs")
RAG_INDEX_DIR = os.path.join(STORAGE_DIR, "rag_index")
INDEX_PATH    = os.path.join(RAG_INDEX_DIR, "faiss.index")
META_PATH     = os.path.join(RAG_INDEX_DIR, "meta.pkl")

os.makedirs(RAG_INDEX_DIR,  exist_ok=True)
os.makedirs(GOV_LAWS_DIR,   exist_ok=True)
os.makedirs(USER_DOCS_DIR,  exist_ok=True)

# Global lock — prevents concurrent index rebuilds corrupting the index file
_INDEX_LOCK = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY INDEX CACHE
# Avoids reading the FAISS index file from disk on every query.
# Invalidated (set to None) whenever rebuild_full_index() completes.
# ─────────────────────────────────────────────────────────────────────────────
_index_cache = None   # faiss.Index object
_meta_cache  = None   # dict {"chunks": [...], "sources": [...], "vectors": np.ndarray}

def _invalidate_cache():
    global _index_cache, _meta_cache
    _index_cache = None
    _meta_cache  = None

def _load_index():
    """
    Load FAISS index and metadata from disk into memory cache.
    Subsequent calls return the cached objects without disk I/O.
    Also stores the pre-built normalised vectors in the cache so the
    filename-filtered search path can use them without re-embedding.
    """
    global _index_cache, _meta_cache
    if _index_cache is not None and _meta_cache is not None:
        return _index_cache, _meta_cache

    if not kb_exists():
        return None, None

    _index_cache = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        _meta_cache = pickle.load(f)

    return _index_cache, _meta_cache


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("✔ Embedding model loaded")
except Exception as e:
    print("❌ Embedding load failed:", e)
    embedder = None


# ─────────────────────────────────────────────────────────────────────────────
# PDF TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def load_pdf_text(path):
    try:
        reader = PdfReader(path)
        pages  = [(p.extract_text() or "") for p in reader.pages]
        return "\n".join(pages)
    except:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# INDEX REBUILD
# ─────────────────────────────────────────────────────────────────────────────
def rebuild_full_index():
    """
    Embed all PDF chunks and build the FAISS index.
    Also stores the normalised vectors in meta so query_rag can use them
    directly for filtered searches without re-embedding.
    """
    print("🔄 Rebuilding full RAG index...")

    if not _INDEX_LOCK.acquire(blocking=False):
        print("⚠ Index rebuild already in progress — skipping.")
        return False

    try:
        chunks  = []
        sources = []

        def _chunk_text(text, min_len=40):
            return [p.strip() for p in text.split("\n") if len(p.strip()) >= min_len]

        for directory in [GOV_LAWS_DIR, USER_DOCS_DIR]:
            for f in sorted(os.listdir(directory)):
                if f.endswith(".pdf"):
                    fp  = os.path.join(directory, f)
                    txt = load_pdf_text(fp)
                    if txt.strip():
                        doc_chunks = _chunk_text(txt)
                        chunks.extend(doc_chunks)
                        sources.extend([f] * len(doc_chunks))

        if not chunks or embedder is None:
            return False

        # Embed all chunks once at build time
        vectors = embedder.encode(chunks, show_progress_bar=True, batch_size=64)

        # Normalise — inner product on unit vectors = cosine similarity
        norms        = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors_norm = (vectors / np.where(norms == 0, 1, norms)).astype("float32")

        # FAISS index
        index = faiss.IndexFlatIP(vectors_norm.shape[1])
        index.add(vectors_norm)
        faiss.write_index(index, INDEX_PATH)

        # Save chunks, sources AND pre-built vectors so filtered search
        # never needs to re-embed chunks at query time
        with open(META_PATH, "wb") as f:
            pickle.dump({
                "chunks":   chunks,
                "sources":  sources,
                "vectors":  vectors_norm,   # ← stored so filtered path can reuse them
            }, f)

        print(f"✔ Index built: {len(chunks)} chunks from {len(set(sources))} documents.")

        # Invalidate in-memory cache so next query loads fresh data
        _invalidate_cache()
        return True

    finally:
        _INDEX_LOCK.release()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def kb_exists():
    return os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)


def list_index_documents():
    docs = []
    for directory in [GOV_LAWS_DIR, USER_DOCS_DIR]:
        for f in os.listdir(directory):
            if f.endswith(".pdf"):
                docs.append(os.path.join(directory, f))
    return docs


def delete_from_index(path):
    if os.path.exists(path):
        os.remove(path)
    rebuild_full_index()


# ─────────────────────────────────────────────────────────────────────────────
# QUERY RAG
# ─────────────────────────────────────────────────────────────────────────────
def query_rag(query, k=5, threshold=0.45):
    if not kb_exists() or embedder is None:
        return "NO_INDEX", []

    index, meta = _load_index()
    if index is None or meta is None:
        return "NO_INDEX", []

    # Auto-migrate old index format (whole-document vectors, no chunks key)
    if "texts" in meta and "chunks" not in meta:
        print("⚠ Old index format detected — rebuilding...")
        ok = rebuild_full_index()
        if not ok:
            return "NO_INDEX", []
        index, meta = _load_index()

    chunks   = meta.get("chunks",  [])
    sources  = meta.get("sources", [])
    vectors  = meta.get("vectors", None)   # pre-built normalised vectors

    if not chunks:
        return "NO_INDEX", []

    # Embed and normalise the query vector (needed by both paths)
    q_vec      = embedder.encode([query])
    q_vec_norm = (q_vec / np.linalg.norm(q_vec)).astype("float32")

    # ── Filename-filtered path ────────────────────────────────────────────────
    # If the query mentions a document name, restrict search to that document.
    # Uses pre-built vectors from meta — NO re-embedding at query time.
    q_lower       = query.lower()
    unique_sources = list(set(sources))
    doc_hits      = [
        src for src in unique_sources
        if (src.lower().replace(".pdf", "") in q_lower
            or fuzz.partial_ratio(src.lower().replace(".pdf", ""), q_lower) > 80)
    ]

    if doc_hits and vectors is not None:
        # Get indices of chunks belonging to matched documents
        filtered_idx = [i for i, s in enumerate(sources) if s in doc_hits]

        if filtered_idx:
            f_vecs   = vectors[filtered_idx]           # reuse pre-built vectors
            scores   = np.dot(f_vecs, q_vec_norm.T).flatten()

            # Apply per-result threshold (same logic as full FAISS path)
            results = []
            for rank in scores.argsort()[::-1]:
                if float(scores[rank]) < threshold:
                    break                              # sorted descending — safe to stop
                results.append({
                    "source":    sources[filtered_idx[rank]],
                    "paragraph": chunks[filtered_idx[rank]],
                    "score":     float(scores[rank]),
                })
                if len(results) >= k:
                    break

            if not results:
                return "NO_RELEVANT", []
            return "OK", results

    # ── Full FAISS search (default path) ──────────────────────────────────────
    scores_faiss, indices = index.search(q_vec_norm, k * 2)
    scores_faiss = scores_faiss[0]
    indices      = indices[0]

    results = []
    for score, idx in zip(scores_faiss, indices):
        if idx < 0 or idx >= len(chunks):
            continue
        if float(score) < threshold:
            continue
        results.append({
            "source":    sources[idx],
            "paragraph": chunks[idx],
            "score":     float(score),
        })
        if len(results) >= k:
            break

    if not results:
        return "NO_RELEVANT", []

    return "OK", results