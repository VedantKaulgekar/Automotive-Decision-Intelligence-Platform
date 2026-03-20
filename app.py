import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Automotive Decision Intelligence Platform",
    layout="wide",
    page_icon="🚗"
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* ── Hero ── */
.hero-box {
    padding: 52px 40px 44px 40px;
    border-radius: 18px;
    background: linear-gradient(135deg, #0d1117 0%, #1c2333 60%, #1a2744 100%);
    border: 1px solid #2a3550;
    margin-bottom: 36px;
    text-align: center;
}
.hero-title {
    font-size: 44px;
    font-weight: 900;
    color: #ffffff;
    margin: 0 0 10px 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 18px;
    color: #8b9bb4;
    margin: 0 0 28px 0;
    line-height: 1.6;
}
.hero-badge {
    display: inline-block;
    background: rgba(78,140,255,0.15);
    border: 1px solid rgba(78,140,255,0.35);
    color: #4e8cff;
    border-radius: 20px;
    padding: 5px 16px;
    font-size: 13px;
    font-weight: 600;
    margin: 0 5px 8px 5px;
}

/* ── Section header ── */
.section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4e8cff;
    margin-bottom: 16px;
}

/* ── KPI pill row ── */
.kpi-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 36px;
}
.kpi-pill {
    background: #111820;
    border: 1px solid #252d3d;
    border-radius: 10px;
    padding: 14px 22px;
    flex: 1;
    min-width: 140px;
    text-align: center;
}
.kpi-pill .kpi-icon { font-size: 22px; margin-bottom: 5px; }
.kpi-pill .kpi-name { font-size: 13px; font-weight: 700; color: #fff; }
.kpi-pill .kpi-task { font-size: 11px; color: #5a6a80; margin-top: 2px; }

/* ── Feature cards ── */
.feature-card {
    background: #0d1117;
    padding: 24px;
    border-radius: 14px;
    border: 1px solid #1e2535;
    color: #fff;
    height: 100%;
    min-height: 185px;
}
.feature-card:hover {
    border-color: #4e8cff;
    background: #111820;
}
.feature-card .fc-icon { font-size: 28px; margin-bottom: 10px; }
.feature-card h4 { margin: 0 0 8px 0; font-size: 15px; color: #fff; font-weight: 700; }
.feature-card p  { margin: 0; font-size: 13px; color: #6b7a90; line-height: 1.55; }
.feature-card .fc-tag {
    display: inline-block;
    margin-top: 12px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #4e8cff;
    background: rgba(78,140,255,0.1);
    border-radius: 4px;
    padding: 2px 8px;
}

/* ── Pipeline steps ── */
.pipeline-row {
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 32px;
    flex-wrap: wrap;
}
.pipeline-step {
    background: #0d1117;
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 14px 18px;
    flex: 1;
    min-width: 120px;
    text-align: center;
}
.pipeline-step .ps-num  { font-size: 10px; color: #4e8cff; font-weight: 700; letter-spacing: 1px; }
.pipeline-step .ps-name { font-size: 13px; color: #fff; font-weight: 600; margin-top: 4px; }
.pipeline-step .ps-desc { font-size: 11px; color: #5a6a80; margin-top: 3px; }
.pipeline-arrow { color: #2a3550; font-size: 20px; padding: 0 6px; flex-shrink: 0; }

/* ── Tech stack ── */
.tech-pill {
    display: inline-block;
    background: #111820;
    border: 1px solid #1e2535;
    border-radius: 6px;
    padding: 5px 13px;
    font-size: 12px;
    color: #8b9bb4;
    margin: 3px;
    font-weight: 500;
}

/* ── CTA ── */
.cta-card {
    background: linear-gradient(135deg, #0d1117, #131c2e);
    border: 1px solid #2a3550;
    border-radius: 14px;
    padding: 28px 24px;
    text-align: center;
    height: 100%;
}
.cta-card .cta-icon  { font-size: 32px; margin-bottom: 10px; }
.cta-card h4 { margin: 0 0 6px 0; font-size: 16px; color: #fff; font-weight: 700; }
.cta-card p  { margin: 0 0 16px 0; font-size: 13px; color: #6b7a90; }
.cta-card a  {
    display: inline-block;
    background: #4e8cff;
    color: #fff !important;
    text-decoration: none !important;
    border-radius: 8px;
    padding: 8px 22px;
    font-size: 13px;
    font-weight: 600;
}
.cta-card a:hover { background: #3a7af0; }

/* ── Status banner ── */
.status-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 28px;
}
.status-item {
    display: flex;
    align-items: center;
    gap: 7px;
    background: #0d1117;
    border: 1px solid #1e2535;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 12px;
    color: #8b9bb4;
}
.status-dot-green { width:8px; height:8px; border-radius:50%; background:#27ae60; flex-shrink:0; }
.status-dot-amber { width:8px; height:8px; border-radius:50%; background:#f0a500; flex-shrink:0; }
.status-dot-grey  { width:8px; height:8px; border-radius:50%; background:#3a4455; flex-shrink:0; }

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATUS  (check which models are trained)
# ─────────────────────────────────────────────────────────────────────────────
energy_trained     = "energy_model"     in st.session_state
efficiency_trained = "efficiency_model" in st.session_state
emission_trained   = "emission_model"   in st.session_state
maintenance_trained= "maintenance_model"in st.session_state
all_trained        = all([energy_trained, efficiency_trained,
                          emission_trained, maintenance_trained])

rag_index_exists   = os.path.exists(os.path.join("storage", "rag_index", "faiss.index"))


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
    <h1 class="hero-title">🚗 Automotive Decision Intelligence</h1>
    <p class="hero-subtitle">
        End-to-end AI platform for automotive factory sustainability —<br>
        AutoML model training, Monte Carlo uncertainty analysis,
        and RAG-powered regulatory Q&amp;A.
    </p>
    <span class="hero-badge">AutoML</span>
    <span class="hero-badge">Monte Carlo Simulation</span>
    <span class="hero-badge">RAG · FAISS</span>
    <span class="hero-badge">Data Visualization</span>
    <span class="hero-badge">PDF Reporting</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATUS BANNER
# ─────────────────────────────────────────────────────────────────────────────
def _dot(trained):
    return '<span class="status-dot-green"></span>' if trained \
           else '<span class="status-dot-amber"></span>'

st.markdown('<p class="section-label">Session Status</p>', unsafe_allow_html=True)
st.markdown(f"""
<div class="status-row">
    <div class="status-item">{_dot(energy_trained)}      Energy Model</div>
    <div class="status-item">{_dot(efficiency_trained)}  Efficiency Classifier</div>
    <div class="status-item">{_dot(emission_trained)}    Emission Classifier</div>
    <div class="status-item">{_dot(maintenance_trained)} Maintenance Model</div>
    <div class="status-item">{_dot(rag_index_exists)}    RAG Knowledge Base</div>
</div>
""", unsafe_allow_html=True)

if not all_trained:
    st.info(
        "⚠️ **Models not yet trained.** Go to **Model Training Studio**, upload the "
        "CSVs from `storage/training_data/`, and train all 4 models before using the Dashboard.",
        icon=None
    )


# ─────────────────────────────────────────────────────────────────────────────
# KPI TARGETS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Target KPIs</p>', unsafe_allow_html=True)
st.markdown("""
<div class="kpi-row">
    <div class="kpi-pill">
        <div class="kpi-icon">⚡</div>
        <div class="kpi-name">Energy</div>
        <div class="kpi-task">Regression · R²</div>
    </div>
    <div class="kpi-pill">
        <div class="kpi-icon">📈</div>
        <div class="kpi-name">Efficiency</div>
        <div class="kpi-task">3-Class · Accuracy</div>
    </div>
    <div class="kpi-pill">
        <div class="kpi-icon">🌍</div>
        <div class="kpi-name">Emissions</div>
        <div class="kpi-task">3-Class · Accuracy</div>
    </div>
    <div class="kpi-pill">
        <div class="kpi-icon">🛠</div>
        <div class="kpi-name">Maintenance</div>
        <div class="kpi-task">3-Class · Accuracy</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PLATFORM CAPABILITIES  (4 cards — no optimization card)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Platform Capabilities</p>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="feature-card">
        <div class="fc-icon">🛠</div>
        <h4>AutoML Training Studio</h4>
        <p>Up to 6 candidate models trained in parallel — RandomForest, ExtraTrees,
           Ridge, XGBoost, LightGBM, MLP. Live scoreboard updates
           as each finishes. Best model selected by 5-fold CV.</p>
        <span class="fc-tag">ThreadPoolExecutor · RandomizedSearchCV</span>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature-card">
        <div class="fc-icon">📊</div>
        <h4>Intelligence Dashboard</h4>
        <p>Slider-driven predictions for all 4 KPIs with cost (₹/kWh) and CO₂ estimates.
           Monte Carlo simulation propagates sensor uncertainty through the model.
           Sensitivity and tornado charts show which inputs matter most.</p>
        <span class="fc-tag">Monte Carlo · Plotly · PDF Export</span>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature-card">
        <div class="fc-icon">💬</div>
        <h4>RAG Regulatory Assistant</h4>
        <p>Ask natural language questions about automotive policy documents.
           Answers grounded strictly in the knowledge base — cites the exact
           document and paragraph. Works offline with Ollama or via Groq cloud.</p>
        <span class="fc-tag">FAISS · all-MiniLM-L6-v2 · LLaMA 3</span>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="feature-card">
        <div class="fc-icon">📚</div>
        <h4>Knowledge Base</h4>
        <p>Upload PDF policy documents and rebuild the vector index instantly.
           Pre-loaded with Automotive Sustainability Principles v4.0, EO 14057,
           and India's National Automotive Policy Draft v2.</p>
        <span class="fc-tag">pypdf · Paragraph Chunking · Auto-Index</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ML PIPELINE OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-label">ML Pipeline</p>', unsafe_allow_html=True)
st.markdown("""
<div class="pipeline-row">
    <div class="pipeline-step">
        <div class="ps-num">01</div>
        <div class="ps-name">Upload CSV</div>
        <div class="ps-desc">CSV or Excel, column validation</div>
    </div>
    <div class="pipeline-arrow">›</div>
    <div class="pipeline-step">
        <div class="ps-num">02</div>
        <div class="ps-name">Preprocess</div>
        <div class="ps-desc">Median impute · Winsorise · RobustScale</div>
    </div>
    <div class="pipeline-arrow">›</div>
    <div class="pipeline-step">
        <div class="ps-num">03</div>
        <div class="ps-name">Feature Engineering</div>
        <div class="ps-desc">thermal_load · wear_index · effective_power</div>
    </div>
    <div class="pipeline-arrow">›</div>
    <div class="pipeline-step">
        <div class="ps-num">04</div>
        <div class="ps-name">80 / 20 Split</div>
        <div class="ps-desc">Stratified · test set held out</div>
    </div>
    <div class="pipeline-arrow">›</div>
    <div class="pipeline-step">
        <div class="ps-num">05</div>
        <div class="ps-name">Parallel AutoML</div>
        <div class="ps-desc">6 candidates · 5-fold CV · n_jobs=-1</div>
    </div>
    <div class="pipeline-arrow">›</div>
    <div class="pipeline-step">
        <div class="ps-num">06</div>
        <div class="ps-name">Evaluate</div>
        <div class="ps-desc">Test R² / Accuracy · Confusion Matrix</div>
    </div>
    <div class="pipeline-arrow">›</div>
    <div class="pipeline-step">
        <div class="ps-num">07</div>
        <div class="ps-name">PDF Report</div>
        <div class="ps-desc">Metrics · Charts · Feature Importance</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TECH STACK
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Tech Stack</p>', unsafe_allow_html=True)
st.markdown("""
<div style="margin-bottom: 28px;">
    <span class="tech-pill">Streamlit</span>
    <span class="tech-pill">scikit-learn</span>
    <span class="tech-pill">XGBoost</span>
    <span class="tech-pill">LightGBM</span>
    <span class="tech-pill">FAISS</span>
    <span class="tech-pill">sentence-transformers</span>
    <span class="tech-pill">Groq / Ollama</span>
    <span class="tech-pill">Plotly</span>
    <span class="tech-pill">ReportLab</span>
    <span class="tech-pill">scipy</span>
    <span class="tech-pill">pandas · numpy</span>
    <span class="tech-pill">pywebview</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CTA — WHERE TO GO NEXT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Where to Go Next</p>', unsafe_allow_html=True)

n1, n2, n3, n4 = st.columns(4)

with n1:
    st.markdown("""
    <div class="cta-card">
        <div class="cta-icon">🛠</div>
        <h4>Model Training Studio</h4>
        <p>Upload datasets and train all 4 models. Start here on first run.</p>
        <a href="./Model_Training_Studio">Open →</a>
    </div>
    """, unsafe_allow_html=True)

with n2:
    st.markdown("""
    <div class="cta-card">
        <div class="cta-icon">📊</div>
        <h4>Intelligence Dashboard</h4>
        <p>Predictions, Monte Carlo analysis, and PDF reports for all KPIs.</p>
        <a href="./Dashboard">Open →</a>
    </div>
    """, unsafe_allow_html=True)

with n3:
    st.markdown("""
    <div class="cta-card">
        <div class="cta-icon">💬</div>
        <h4>RAG Assistant</h4>
        <p>Ask questions about automotive policy and compliance documents.</p>
        <a href="./RAG_Assistant">Open →</a>
    </div>
    """, unsafe_allow_html=True)

with n4:
    st.markdown("""
    <div class="cta-card">
        <div class="cta-icon">📚</div>
        <h4>Knowledge Base</h4>
        <p>Manage and upload PDF documents for the RAG assistant.</p>
        <a href="./Knowledge_Base">Open →</a>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "Automotive Decision Intelligence Platform — "
    "Academic demonstration of AutoML, Monte Carlo simulation, and RAG for factory sustainability."
)