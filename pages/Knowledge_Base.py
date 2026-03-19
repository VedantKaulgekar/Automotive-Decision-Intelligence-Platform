import streamlit as st
import os
import base64
from modules.rag_engine import (
    rebuild_full_index,
    kb_exists,
    list_index_documents,
    delete_from_index,
)

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
STORAGE_GOV = "storage/gov_laws"
STORAGE_USER = "storage/user_docs"

os.makedirs(STORAGE_GOV, exist_ok=True)
os.makedirs(STORAGE_USER, exist_ok=True)


# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Knowledge Base", layout="wide")
st.title("📘 Knowledge Base Manager")


# -------------------------------------------------------
# CATEGORY DETECTION
# -------------------------------------------------------
def detect_category(filename):
    name = filename.lower()

    if any(w in name for w in ["policy", "regulation", "gov"]):
        return "Government Policy"
    if any(w in name for w in ["safety", "compliance"]):
        return "Safety & Standards"
    if "report" in name:
        return "Technical Report"
    if "manual" in name:
        return "User Manual"

    return "General Document"


# -------------------------------------------------------
# PDF VIEWER
# -------------------------------------------------------
def show_pdf(path):
    try:
        with open(path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")

        st.markdown(
            f"""
            <iframe src="data:application/pdf;base64,{base64_pdf}" 
                    width="100%" height="600px" 
                    style="border:none;">
            </iframe>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        st.error("⚠ Preview not available (scanned or corrupted PDF).")


# -------------------------------------------------------
# 📄 PDF UPLOAD (NO INDEXING HERE)
# -------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload PDFs:",
    type=["pdf"],
    accept_multiple_files=True,
)

uploaded_paths = []

if uploaded_files:
    st.subheader("📥 Uploading Files...")

    skipped = []
    saved   = []

    for file in uploaded_files:
        save_path = os.path.join(STORAGE_USER, file.name)

        # Deduplication check
        if os.path.exists(save_path):
            skipped.append(file.name)
            continue

        with open(save_path, "wb") as out:
            out.write(file.read())
        uploaded_paths.append(save_path)
        saved.append(file.name)

    if saved:
        st.success(f"✔ Uploaded {len(saved)} file(s): {', '.join(saved)}")

        # Auto-rebuild index after every successful upload
        with st.spinner("Rebuilding knowledge base index..."):
            ok = rebuild_full_index()
        if ok:
            st.success("✔ Index rebuilt automatically.")
        else:
            st.error("❌ Index rebuild failed — check that the PDFs contain extractable text.")

    if skipped:
        st.warning(
            f"⚠️ {len(skipped)} file(s) skipped — already exist in the knowledge base: "
            f"{', '.join(skipped)}"
        )


# -------------------------------------------------------
# 📚 SHOW ALL DOCUMENTS (WITH DELETE)
# -------------------------------------------------------
st.subheader("📂 Stored Documents")

docs = list_index_documents()  # returns full paths

if not docs:
    st.info("No documents stored yet.")
else:
    for path in docs:
        file = os.path.basename(path)
        category = detect_category(file)

        with st.expander(f"📘 {file}  |  🏷 {category}", expanded=False):

            colA, colB = st.columns([4, 1])

            # PDF Preview
            with colA:
                show_pdf(path)

            # Delete Button (NO INDEXING)
            with colB:
                st.write("### 🗑 Delete File")
                if st.button(f"Delete {file}", key=f"del_{file}"):
                    delete_from_index(path)
                    st.success(f"🗑 Deleted {file} and rebuilt index.")
                    st.rerun()


# -------------------------------------------------------
# 🔄 MANUAL INDEX REBUILD ONLY
# -------------------------------------------------------
st.subheader("🔧 Manage Index")

if st.button("🔄 Rebuild Knowledge Base Index"):
    with st.spinner("Rebuilding index..."):
        ok = rebuild_full_index()

    if ok:
        st.success("✔ Full index rebuilt!")
    else:
        st.error("❌ Failed to rebuild — no valid PDF text found.")


# -------------------------------------------------------
# INDEX STATUS
# -------------------------------------------------------
if kb_exists():
    st.info("✔ Knowledge Base index exists and is ready.")
else:
    st.warning("⚠ No index found. Please rebuild the index.")