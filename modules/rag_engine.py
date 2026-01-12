import os
import pickle
import faiss
import numpy as np
from pypdf import PdfReader
from fuzzywuzzy import fuzz

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
# FULL REBUILD INDEX (simple + safe)
# -------------------------------------------------
def rebuild_full_index():
    print("🔄 Rebuilding full RAG index...")

    docs = []
    names = []

    # Load GOV PDFs
    for f in os.listdir(GOV_LAWS_DIR):
        if f.endswith(".pdf"):
            fp = os.path.join(GOV_LAWS_DIR, f)
            txt = load_pdf_text(fp)
            if txt.strip():
                docs.append(txt)
                names.append(f)

    # Load USER PDFs
    for f in os.listdir(USER_DOCS_DIR):
        if f.endswith(".pdf"):
            fp = os.path.join(USER_DOCS_DIR, f)
            txt = load_pdf_text(fp)
            if txt.strip():
                docs.append(txt)
                names.append(f)

    if not docs or embedder is None:
        return False

    # Embed
    vectors = embedder.encode(docs, show_progress_bar=True)

    # Build FAISS index
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)

    # Save metadata
    with open(META_PATH, "wb") as f:
        pickle.dump({"texts": docs, "names": names}, f)

    return True


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

    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    texts = meta["texts"]
    names = meta["names"]

    # -----------------------------
    # SPLIT INTO PARAGRAPHS
    # -----------------------------
    paras = []
    sources = []

    for text, src in zip(texts, names):
        # Large chunks work better than tiny ones
        ps = [p.strip() for p in text.split("\n") if len(p.strip()) > 40]
        paras.extend(ps)
        sources.extend([src] * len(ps))

    if not paras:
        return "NO_INDEX", []

    # -----------------------------
    # OPTIONAL FILE-NAME FILTERING
    # -----------------------------
    q = query.lower()
    doc_hits = []

    for src in set(names):
        clean = src.lower().replace(".pdf", "")
        if clean in q or fuzz.partial_ratio(clean, q) > 80:
            doc_hits.append(src)

    # Filter paragraphs ONLY if filename matched
    if doc_hits:
        filtered = [(p, s) for p, s in zip(paras, sources) if s in doc_hits]
        if filtered:
            paras, sources = zip(*filtered)
            paras = list(paras)
            sources = list(sources)

    # -----------------------------
    # EMBEDDINGS
    # -----------------------------
    para_vecs = embedder.encode(paras)
    query_vec = embedder.encode([query])

    # Cosine similarity (better than dot product)
    para_vecs_norm = para_vecs / np.linalg.norm(para_vecs, axis=1, keepdims=True)
    query_vec_norm = query_vec / np.linalg.norm(query_vec)

    scores = np.dot(para_vecs_norm, query_vec_norm.T).flatten()

    # -----------------------------
    # RELEVANCE CHECK
    # -----------------------------
    if max(scores) < threshold:
        return "NO_RELEVANT", []

    # -----------------------------
    # TOP-K MATCHES
    # -----------------------------
    top = scores.argsort()[::-1][:k]

    results = []
    for i in top:
        results.append({
            "source": sources[i],
            "paragraph": paras[i],
            "score": float(scores[i])
        })

    return "OK", results
