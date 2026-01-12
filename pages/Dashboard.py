import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import joblib
import os

# Model Wrappers
from modules.model_wrappers import (
    EnergyModel, EfficiencyModel, EmissionModel, MaintenanceModel
)

# Optimizers
from modules.optimizer import optimize_with_model, pareto_optimize

# Business logic
from modules.cost_model import predict_cost
from modules.grid_model import estimate_grid_emission
from modules.uncertainity_model import monte_carlo_energy

# Storage
from modules.storage import PATHS, ensure_dirs

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

warnings = __import__("warnings")
warnings.filterwarnings("ignore")
os.makedirs("/mount/data/models", exist_ok=True)

MODEL_DIR = "/mount/data/models"


# ============================================================================================
# PAGE SETUP
# ============================================================================================
st.set_page_config(page_title="Automotive Intelligence Dashboard", layout="wide")
ensure_dirs()
pdfmetrics.registerFont(TTFont("DejaVu", "assets/fonts/DejaVuSans.ttf"))

st.title("🚗 Automotive Intelligence Dashboard")
st.caption("AI-driven operational intelligence for energy, efficiency, emissions & maintenance.")


# ============================================================================================
# HELPERS
# ============================================================================================
def safe_fig(text="No data"):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)
    ax.axis("off")
    return fig


def fig_to_img(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return buf


def pdf_export(title, figs, inputs, optim_results):
    buf = BytesIO()
    styles = getSampleStyleSheet()
    styles["Normal"].fontName = "DejaVu"
    styles["Heading1"].fontName = "DejaVu"
    styles["Heading2"].fontName = "DejaVu"

    doc = SimpleDocTemplate(buf, pagesize=A4)
    story = []

    story.append(Paragraph(title, styles["Heading1"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>User Inputs</b>", styles["Heading2"]))
    for k, v in inputs.items():
        story.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    story.append(Spacer(1, 14))
    story.append(Paragraph("<b>Optimization Results</b>", styles["Heading2"]))
    for k, v in optim_results.items():
        story.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    story.append(Spacer(1, 14))
    story.append(Paragraph("<b>Visual Analytics</b>", styles["Heading2"]))

    for fig in figs:
        img = fig_to_img(fig)
        story.append(Image(img, width=380, height=240))
        story.append(Spacer(1, 15))

    doc.build(story)
    buf.seek(0)
    return buf


# ============================================================================================
# LOAD MODELS
# ============================================================================================
energy_model = EnergyModel()
eff_model = EfficiencyModel()
emiss_model = EmissionModel()
maint_model = MaintenanceModel()

# Use session_state to store optimization results
if "optim" not in st.session_state:
    st.session_state.optim = {
        "energy": None,
        "eff": None,
        "em": None,
        "maint": None
    }


# ============================================================================================
# TABS
# ============================================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Energy", "📈 Efficiency", "🌍 Emissions", "🛠 Maintenance"
])

import os
import streamlit as st

# Check if models exist in session_state
missing = []
if "energy_model" not in st.session_state:
    missing.append("Energy Model")
if "efficiency_model" not in st.session_state:
    missing.append("Efficiency Model")
if "emission_model" not in st.session_state:
    missing.append("Emission Model")
if "maintenance_model" not in st.session_state:
    missing.append("Maintenance Model")

if missing:
    st.warning(
        "⚠ The following models are missing:\n\n" +
        "\n".join(f"- **{m}**" for m in missing)
    )
    st.info("➡ Please go to **Model Training Studio** and train the models.")
    st.stop()

# ====================================================================================================
# TAB 1 - ENERGY
# ====================================================================================================
with tab1:

    st.header("⚡ Energy Model Analysis")

    col1, col2 = st.columns(2)
    with col1:
        en_load = st.slider("Production Load (%)", 30, 100, 80, key="en_load") / 100
        en_cycle = st.slider("Cycle Time (s)", 10, 90, 40, key="en_cycle")
    with col2:
        en_temp = st.slider("Machine Temperature (°C)", 20, 120, 60, key="en_temp")
        en_speed = st.slider("Axis Speed (m/s)", 0.2, 3.0, 1.2, key="en_speed")

    energy = energy_model.predict(en_load, en_cycle, en_temp, en_speed)

    if energy:
        cost = predict_cost(energy)
        co2 = estimate_grid_emission(energy)

        c1, c2, c3 = st.columns(3)
        c1.metric("Energy Consumption", f"{energy:.2f} kWh")
        c2.metric("Cost Estimate", f"₹{cost:.2f}")
        c3.metric("CO₂ Emission", f"{co2:.2f} kg")

    st.divider()
    st.subheader("🚀 What-If Analysis")

    # WHAT-IF MONTE CARLO
    if energy:
        mc = monte_carlo_energy(energy)

        # Variability
        fig_mc, ax = plt.subplots(figsize=(6, 3))
        ax.plot(mc)
        ax.set_title("Monte Carlo Variability")
        st.pyplot(fig_mc)

        # Histogram
        fig_hist, ax2 = plt.subplots(figsize=(6, 3))
        ax2.hist(mc, bins=30)
        ax2.set_title("Distribution of Energy Outcomes")
        st.pyplot(fig_hist)
    else:
        fig_mc = fig_hist = safe_fig()

    st.divider()

    # -------------------------------------------------------------
    # OPTIMIZATION BUTTONS
    # -------------------------------------------------------------
    st.subheader("🚀 Optimization")

    if st.button("Run Grid Optimization (Min Energy)"):
        st.session_state.optim["energy"] = optimize_with_model(
            energy_model, "min", n_samples=15
        )

    best_energy = st.session_state.optim["energy"]

    if best_energy:
        st.success(
            f"**Optimal Settings →** Load={best_energy['load']:.2f}, "
            f"Cycle={best_energy['cycle']:.1f}, Temp={best_energy['temp']:.1f}, "
            f"Speed={best_energy['speed']:.2f}  \n\n"
            f"📉 Min Energy: **{best_energy['score']:.2f} kWh**"
        )

    # Training heatmap
    try:
        df = st.session_state.get("energy_raw_data")

        fig_corr, ax3 = plt.subplots(figsize=(6, 4))
        sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=ax3)
        ax3.set_title("Training Correlation Map")
        st.pyplot(fig_corr)

        # Scatterplot
        fig_scatter, ax4 = plt.subplots(figsize=(6, 4))
        ax4.scatter(df["production_load"], df["energy"], s=5)
        ax4.set_xlabel("Load")
        ax4.set_ylabel("Energy")
        ax4.set_title("Load vs Energy")
        st.pyplot(fig_scatter)

    except:
        fig_corr = fig_scatter = safe_fig()

    # PDF Export
    pdf = pdf_export(
        "Energy Model Report",
        [fig_mc, fig_hist, fig_corr, fig_scatter],
        {
            "load": en_load,
            "cycle": en_cycle,
            "temp": en_temp,
            "speed": en_speed
        },
        best_energy or {}
    )

    st.download_button("📄 Download Energy Report", pdf, file_name="energy_report.pdf", key="tab1")


# ====================================================================================================
# TAB 2 - EFFICIENCY
# ====================================================================================================
with tab2:

    st.header("📈 Efficiency Classifier")

    col1, col2 = st.columns(2)
    with col1:
        ef_load = st.slider("Load (%)", 30, 100, 80, key="ef_load") / 100
        ef_cycle = st.slider("Cycle Time", 10, 90, 40, key="ef_cycle")
    with col2:
        ef_temp = st.slider("Temperature", 20, 120, 60, key="ef_temp")
        ef_speed = st.slider("Axis Speed", 0.2, 3.0, 1.2, key="ef_speed")

    pred_eff = eff_model.predict(ef_load, ef_cycle, ef_temp, ef_speed)
    st.metric("Predicted Class", pred_eff)

    st.divider()
    st.subheader("🚀 Optimization")

    if st.button("Optimize Efficiency (Grid)", key="eff_grid"):
        st.session_state.optim["eff"] = optimize_with_model(eff_model, "max", 15)

    best_eff = st.session_state.optim["eff"]

    if best_eff:
        st.success(
            f"**Optimal Settings →** Load={best_energy['load']:.2f}, "
            f"Cycle={best_energy['cycle']:.1f}, Temp={best_energy['temp']:.1f}, "
            f"Speed={best_energy['speed']:.2f}  \n\n"
            f"⭐ Best Efficiency Score: **{best_eff['raw_prediction']}**"
        )
    # Visuals
    try:
        df = st.session_state.get("efficiency_raw_data")

        fig_eff, ax = plt.subplots(figsize=(6, 3))
        sns.countplot(x=df["efficiency_class"], ax=ax)
        st.pyplot(fig_eff)

        fig_sc, ax2 = plt.subplots(figsize=(6, 3))
        ax2.scatter(df["production_load"], df["cycle_time"], s=5)
        ax2.set_title("Load vs Cycle Time")
        st.pyplot(fig_sc)

    except:
        fig_eff = fig_sc = safe_fig()

    pdf = pdf_export("Efficiency Report", [fig_eff, fig_sc],
                     {"load": ef_load, "cycle": ef_cycle, "temp": ef_temp, "speed": ef_speed},
                     best_eff or {}
                     )

    st.download_button("📄 Download PDF", pdf, file_name="efficiency_report.pdf", key="tab2")


# ====================================================================================================
# TAB 3 - EMISSIONS
# ====================================================================================================
with tab3:

    st.header("🌍 Emission Classifier")

    col1, col2 = st.columns(2)
    with col1:
        em_load = st.slider("Load (%)", 30, 100, 80, key="em_load") / 100
        em_cycle = st.slider("Cycle Time", 10, 90, 40, key="em_cycle")
    with col2:
        em_temp = st.slider("Temperature", 20, 120, 60, key="em_temp")
        em_speed = st.slider("Axis Speed", 0.2, 3.0, 1.2, key="em_speed")

    pred_em = emiss_model.predict(em_load, em_cycle, em_temp, em_speed)
    st.metric("Emission Class", pred_em)

    st.divider()
    st.subheader("🚀 Optimization")

    if st.button("Minimize Emissions (Grid)", key="em_grid"):
        st.session_state.optim["em"] = optimize_with_model(emiss_model, "min", 15)

    best_em = st.session_state.optim["em"]

    if best_energy:
        st.success(
            f"**Optimal Settings →** Load={best_energy['load']:.2f}, "
            f"Cycle={best_energy['cycle']:.1f}, Temp={best_energy['temp']:.1f}, "
            f"Speed={best_energy['speed']:.2f}  \n\n"
            f"🌍 Lowest Emission Score: **{best_em['raw_prediction']}**"
        )

    # Visuals
    try:
        df = st.session_state.get("emission_raw_data")

        fig_em, ax = plt.subplots(figsize=(6, 3))
        sns.countplot(x=df["emission_class"], ax=ax)
        st.pyplot(fig_em)

        fig_sc, ax2 = plt.subplots(figsize=(6, 3))
        ax2.scatter(df["production_load"], df["machine_temperature"], s=5)
        st.pyplot(fig_sc)

    except:
        fig_em = fig_sc = safe_fig()

    pdf = pdf_export("Emission Report",
                     [fig_em, fig_sc],
                     {"load": em_load, "cycle": em_cycle, "temp": em_temp, "speed": em_speed},
                     best_em or {})

    st.download_button("📄 Download PDF", pdf, file_name="emission_report.pdf", key="tab3")


# ====================================================================================================
# TAB 4 - MAINTENANCE
# ====================================================================================================
with tab4:

    st.header("🛠 Maintenance Predictor")

    col1, col2 = st.columns(2)
    with col1:
        m_load = st.slider("Load (%)", 30, 100, 80, key="m_load") / 100
        m_cycle = st.slider("Cycle Time", 10, 90, 40, key="m_cycle")
    with col2:
        m_temp = st.slider("Temperature", 20, 120, 60, key="m_temp")
        m_speed = st.slider("Speed", 0.2, 3.0, 1.2, key="m_speed")

    pred_m = maint_model.predict(m_load, m_cycle, m_temp, m_speed)
    st.metric("Maintenance Risk", pred_m)

    st.divider()
    st.subheader("🚀 Optimization")
    
    if st.button("Minimize Risk (Grid)", key="m_grid"):
        st.session_state.optim["maint"] = optimize_with_model(maint_model, "min", 15)

    best_m = st.session_state.optim["maint"]

    if best_energy:
        st.success(
            f"**Optimal Settings →** Load={best_energy['load']:.2f}, "
            f"Cycle={best_energy['cycle']:.1f}, Temp={best_energy['temp']:.1f}, "
            f"Speed={best_energy['speed']:.2f}  \n\n"
            f"🛠 Minimum Risk Level: **{best_m['raw_prediction']}**"
        )

    # Visuals
    try:
        df = st.session_state.get("maintenance_raw_data")

        fig_mt, ax = plt.subplots(figsize=(6, 3))
        sns.countplot(x=df["maintenance_class"], ax=ax)
        st.pyplot(fig_mt)

        fig_sc, ax2 = plt.subplots(figsize=(6, 3))
        ax2.scatter(df["vibration_level"], df["maintenance_class"].astype(str).astype("category").cat.codes)
        st.pyplot(fig_sc)

    except:
        fig_mt = fig_sc = safe_fig()

    pdf = pdf_export("Maintenance Report",
                     [fig_mt, fig_sc],
                     {"load": m_load, "cycle": m_cycle, "temp": m_temp, "speed": m_speed},
                     best_m or {})

    st.download_button("📄 Download PDF", pdf, file_name="maintenance_report.pdf", key="tab4")
