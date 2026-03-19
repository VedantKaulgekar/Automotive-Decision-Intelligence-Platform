import os
import pickle
import faiss
import numpy as np
import threading
from pypdf import PdfReader
try:
    from thefuzz import fuzz
except ImportError:
    from fuzzywuzzy import fuzz  # fallback for older installs

# -------------------------------------------------
# PATH SETUP
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

GOV_LAWS_DIR = os.path.join(STORAGE_DIR, "gov_laws")
USER_DOCS_DIR = os.path.join(STORAGE_DIR, "user_docs")

RAG_INDEX_DIR = os.path.join(STORAGE_DIR, "rag_index")
INDEX_PATH = os.path.join(RAG_INDEX_DIR, "faiss.index")
META_PATH = os.path.join(RAG_INDEX_DIR, "meta.pkl")

os.makedirs(RAG_INDEX_DIR, exist_ok=True)
os.makedirs(GOV_LAWS_DIR, exist_ok=True)
os.makedirs(USER_DOCS_DIR, exist_ok=True)

# Global lock — prevents concurrent index rebuilds corrupting the index file
_INDEX_LOCK = threading.Lock()

# -------------------------------------------------
# LOAD EMBEDDING MODEL
# -------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("✔ Embedding model loaded")
except Exception as e:
    print("❌ Embedding load failed:", e)
    embedder = None


# -------------------------------------------------
# PDF TEXT EXTRACTION
# -------------------------------------------------
def load_pdf_text(path):
    try:
        reader = PdfReader(path)
        pages = [(p.extract_text() or "") for p in reader.pages]
        return "\n".join(pages)
    except:
        return ""


# -------------------------------------------------
# FULL REBUILD INDEX — chunks stored at build time
# -------------------------------------------------
def rebuild_full_index():
    print("🔄 Rebuilding full RAG index...")

    # Prevent concurrent rebuilds from corrupting the index file
    if not _INDEX_LOCK.acquire(blocking=False):
        print("⚠ Index rebuild already in progress — skipping duplicate request.")
        return False

    try:
        chunks = []   # text of each chunk
        sources = []  # source filename for each chunk

        def _chunk_text(text, min_len=40):
            """Split document text into paragraph-level chunks."""
            return [p.strip() for p in text.split("\n") if len(p.strip()) >= min_len]

        # Load GOV PDFs
        for f in sorted(os.listdir(GOV_LAWS_DIR)):
            if f.endswith(".pdf"):
                fp  = os.path.join(GOV_LAWS_DIR, f)
                txt = load_pdf_text(fp)
                if txt.strip():
                    doc_chunks = _chunk_text(txt)
                    chunks.extend(doc_chunks)
                    sources.extend([f] * len(doc_chunks))

        # Load USER PDFs
        for f in sorted(os.listdir(USER_DOCS_DIR)):
            if f.endswith(".pdf"):
                fp  = os.path.join(USER_DOCS_DIR, f)
                txt = load_pdf_text(fp)
                if txt.strip():
                    doc_chunks = _chunk_text(txt)
                    chunks.extend(doc_chunks)
                    sources.extend([f] * len(doc_chunks))

        if not chunks or embedder is None:
            return False

        # Embed all chunks at build time
        vectors = embedder.encode(chunks, show_progress_bar=True, batch_size=64)

        # Normalise for cosine similarity via inner product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors_norm = vectors / np.where(norms == 0, 1, norms)

        # Build FAISS index (inner product on normalised vecs = cosine similarity)
        index = faiss.IndexFlatIP(vectors_norm.shape[1])
        index.add(vectors_norm)

        faiss.write_index(index, INDEX_PATH)

        # Save chunk texts and sources (one entry per chunk)
        with open(META_PATH, "wb") as f:
            pickle.dump({"chunks": chunks, "sources": sources}, f)

        print(f"✔ Index built: {len(chunks)} chunks from {len(set(sources))} documents.")
        return True

    finally:
        _INDEX_LOCK.release()


# -------------------------------------------------
# CHECK IF KB EXISTS
# -------------------------------------------------
def kb_exists():
    return os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)


# -------------------------------------------------
# LIST ALL DOCUMENTS
# -------------------------------------------------
def list_index_documents():
    docs = []

    for f in os.listdir(GOV_LAWS_DIR):
        if f.endswith(".pdf"):
            docs.append(os.path.join(GOV_LAWS_DIR, f))

    for f in os.listdir(USER_DOCS_DIR):
        if f.endswith(".pdf"):
            docs.append(os.path.join(USER_DOCS_DIR, f))

    return docs


# -------------------------------------------------
# SIMPLE DELETE (file + rebuild index)
# -------------------------------------------------
def delete_from_index(path):
    if os.path.exists(path):
        os.remove(path)

    # Always rebuild safely
    rebuild_full_index()


# -------------------------------------------------
# QUERY RAG
# -------------------------------------------------
def query_rag(query, k=5, threshold=0.45):
    if not kb_exists() or embedder is None:
        return "NO_INDEX", []

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    # Detect old whole-document index format and rebuild automatically
    if "texts" in meta and "chunks" not in meta:
        print("⚠ Old index format detected — rebuilding with chunk-level vectors...")
        ok = rebuild_full_index()
        if not ok:
            return "NO_INDEX", []
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)

    chunks  = meta.get("chunks",  [])
    sources = meta.get("sources", [])

    if not chunks:
        return "NO_INDEX", []

    index = faiss.read_index(INDEX_PATH)

    # -----------------------------
    # OPTIONAL FILE-NAME FILTERING
    # -----------------------------
    q = query.lower()
    unique_sources = list(set(sources))
    doc_hits = []

    for src in unique_sources:
        clean = src.lower().replace(".pdf", "")
        if clean in q or fuzz.partial_ratio(clean, q) > 80:
            doc_hits.append(src)

    if doc_hits:
        filtered = [(c, s) for c, s in zip(chunks, sources) if s in doc_hits]
        if filtered:
            f_chunks, f_sources = zip(*filtered)
            f_vecs = embedder.encode(list(f_chunks))
            norms  = np.linalg.norm(f_vecs, axis=1, keepdims=True)
            f_vecs_norm = f_vecs / np.where(norms == 0, 1, norms)

            q_vec      = embedder.encode([query])
            q_vec_norm = q_vec / np.linalg.norm(q_vec)

            scores = np.dot(f_vecs_norm, q_vec_norm.T).flatten()

            if float(max(scores)) < threshold:
                return "NO_RELEVANT", []

            top = scores.argsort()[::-1][:k]
            return "OK", [
                {"source": f_sources[i], "paragraph": f_chunks[i], "score": float(scores[i])}
                for i in top
            ]

    # -----------------------------
    # FULL INDEX SEARCH via FAISS
    # -----------------------------
    q_vec      = embedder.encode([query])
    q_vec_norm = (q_vec / np.linalg.norm(q_vec)).astype("float32")

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
            "score":     float(score)
        })
        if len(results) >= k:
            break

    if not results:
        return "NO_RELEVANT", []

    return "OK", results