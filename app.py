import streamlit as st
import os

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Automotive Decision Intelligence Platform",
    layout="wide",
    page_icon="🚗"
)

# ---------------------------------------------------------
# STYLES (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>

.hero-title {
    font-size: 42px;
    font-weight: 900;
    color: #ffffff;
    text-align: center;
    padding-top: 20px;
}

.hero-subtitle {
    font-size: 20px;
    color: #e0e0e0;
    text-align: center;
    margin-top: -10px;
}

.hero-box {
    padding: 40px;
    border-radius: 16px;
    background: linear-gradient(135deg, #1c1f26 0%, #2f3542 100%);
    margin-bottom: 40px;
}

.feature-card {
    background: #111418;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #2c2f33;
    text-align: left;
    color: #fff;
    height: 220px;
}

.feature-card:hover {
    border-color: #4e8cff;
    background: #151a20;
    transition: 0.25s ease;
}

.big-icon {
    font-size: 36px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# FIRST TIME WARNING
# ---------------------------------------------------------
if not os.path.exists("storage/training_data") or len(os.listdir("storage/training_data")) == 0:
    st.warning("🚨 No training data found. Redirecting to Model Training Studio…")
    st.markdown("[➡ Go to Model Training Studio](./Model_Training_Studio)")


# ---------------------------------------------------------
# HERO SECTION
# ---------------------------------------------------------
st.markdown("""
<div class="hero-box">
    <h1 class="hero-title">🚗 Automotive Decision Intelligence Platform</h1>
    <p class="hero-subtitle">
        AI-powered analytics, optimization & regulatory compliance for modern automotive factories.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# FEATURE GRID
# ---------------------------------------------------------
st.write("### 🔥 Platform Capabilities")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="small-icon">📘</div>
        <h4>RAG Knowledge Engine</h4>
        <p>Search across government automotive policies, OEM guidelines, and technical PDFs using semantic RAG.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="small-icon">⚙️</div>
        <h4>AutoML Training Studio</h4>
        <p>Automated feature engineering, model selection, training, metrics, and PDF reporting for all production KPIs.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="small-icon">📊</div>
        <h4>Predictive Dashboards</h4>
        <p>Real-time predictions for Energy, Efficiency, Emissions & Maintenance using optimized ML pipelines.</p>
    </div>
    """, unsafe_allow_html=True)


col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    <div class="feature-card">
        <div class="small-icon">🚀</div>
        <h4>Optimization Engine</h4>
        <p>Fast random search for minimization of energy, emissions & maintenance risk.</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="feature-card">
        <div class="small-icon">🔍</div>
        <h4>What-If Analysis</h4>
        <p>Run Monte-Carlo variability tests & scenario analysis with interactive sliders.</p>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
    <div class="feature-card" style="height">
        <div class="small-icon">📄</div>
        <h4>Automated PDF Reports</h4>
        <p>Generate professional reports including predictions, plots, optimization insights & compliance citations.</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# CALL TO ACTION
# ---------------------------------------------------------
st.write("")
st.write("### ➡ Where do you want to go next?")

cta_col1, cta_col2, cta_col3 = st.columns(3)

with cta_col1:
    st.markdown("#### 📘 RAG Assistant")
    st.markdown("[Open →](./RAG_Assistant)")

with cta_col2:
    st.markdown("#### ⚙️ Model Training Studio")
    st.markdown("[Open →](./Model_Training_Studio)")

with cta_col3:
    st.markdown("#### 📊 Intelligence Dashboard")
    st.markdown("[Open →](./Dashboard)")



