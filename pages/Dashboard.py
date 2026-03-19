import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
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

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings = __import__("warnings")
warnings.filterwarnings("ignore")


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
    import datetime
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import HRFlowable, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

    buf = BytesIO()

    # ── Styles ────────────────────────────────────────────────
    base_font = "DejaVu"

    cover_title_style = ParagraphStyle("CoverTitle", fontName=base_font,
        fontSize=22, leading=28, alignment=TA_CENTER, spaceAfter=6)
    cover_sub_style   = ParagraphStyle("CoverSub", fontName=base_font,
        fontSize=11, leading=14, alignment=TA_CENTER, textColor=colors.HexColor("#555555"), spaceAfter=4)
    section_style     = ParagraphStyle("Section", fontName=base_font,
        fontSize=13, leading=16, textColor=colors.HexColor("#1a1a2e"), spaceAfter=6, spaceBefore=14)
    body_style        = ParagraphStyle("Body", fontName=base_font,
        fontSize=10, leading=14, spaceAfter=3)
    caption_style     = ParagraphStyle("Caption", fontName=base_font,
        fontSize=9, leading=12, textColor=colors.HexColor("#444444"),
        alignment=TA_CENTER, spaceAfter=10, spaceBefore=4)
    kv_key_style      = ParagraphStyle("KVKey", fontName=base_font,
        fontSize=10, leading=13, textColor=colors.HexColor("#333333"))
    kv_val_style      = ParagraphStyle("KVVal", fontName=base_font,
        fontSize=10, leading=13, textColor=colors.HexColor("#1a1a2e"))

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=18*mm, bottomMargin=18*mm
    )
    story = []

    # ── Cover header ──────────────────────────────────────────
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph("Automotive Decision Intelligence Platform", cover_sub_style))
    story.append(Paragraph(title, cover_title_style))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}",
        cover_sub_style
    ))
    story.append(Spacer(1, 4*mm))
    story.append(HRFlowable(width="100%", thickness=1.5,
                             color=colors.HexColor("#4e8cff"), spaceAfter=8*mm))

    # ── Per-report intro text ─────────────────────────────────
    intros = {
        "Energy Model Report": (
            "This report summarises the AI-predicted energy consumption for the configured "
            "operating parameters, together with Monte Carlo variability analysis and training-data "
            "insights. Use the optimization results to identify settings that minimise energy draw."
        ),
        "Efficiency Report": (
            "This report presents the predicted efficiency class for the given operating conditions, "
            "supported by training-data visualizations. Efficiency is classified as Low, Medium, or High "
            "based on production load, cycle time, machine temperature, and axis speed."
        ),
        "Emission Report": (
            "This report covers the predicted emission class for the configured parameters, "
            "with supporting data visualizations from the training set. Reducing production load "
            "and machine temperature are the primary levers for moving from High to Low emission class."
        ),
        "Maintenance Report": (
            "This report details the predicted maintenance risk level for the given operating conditions. "
            "Elevated vibration and tool wear are the strongest predictors of High risk. "
            "Use the optimization results to identify parameter settings that minimise maintenance risk."
        ),
    }
    intro_text = intros.get(title,
        "AI-driven operational intelligence for the automotive factory floor.")
    story.append(Paragraph(intro_text, body_style))
    story.append(Spacer(1, 6*mm))

    # ── Section 1 — Operating Parameters ─────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#cccccc"), spaceAfter=4))
    story.append(Paragraph("1. Operating Parameters", section_style))
    story.append(Paragraph(
        "The following parameter values were set by the user and used as model inputs:",
        body_style
    ))
    story.append(Spacer(1, 2*mm))

    input_labels = {
        "load":  ("Production Load",       "fraction (0–1)"),
        "cycle": ("Cycle Time",            "seconds"),
        "temp":  ("Machine Temperature",   "°C"),
        "speed": ("Axis Speed",            "m/s"),
    }
    if inputs:
        table_data = [["Parameter", "Value", "Unit"]]
        for k, v in inputs.items():
            label, unit = input_labels.get(k, (k.replace("_", " ").title(), ""))
            table_data.append([
                Paragraph(label, kv_key_style),
                Paragraph(f"{v:.3f}" if isinstance(v, float) else str(v), kv_val_style),
                Paragraph(unit, kv_key_style),
            ])
        t = Table(table_data, colWidths=[90*mm, 45*mm, 35*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#4e8cff")),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",    (0, 0), (-1, 0),  base_font),
            ("FONTSIZE",    (0, 0), (-1, 0),  10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#f5f7ff"), colors.white]),
            ("GRID",        (0, 0), (-1, -1),  0.4, colors.HexColor("#dddddd")),
            ("LEFTPADDING", (0, 0), (-1, -1),  6),
            ("RIGHTPADDING",(0, 0), (-1, -1),  6),
            ("TOPPADDING",  (0, 0), (-1, -1),  4),
            ("BOTTOMPADDING",(0,0), (-1, -1),  4),
        ]))
        story.append(t)
    story.append(Spacer(1, 6*mm))

    # ── Section 2 — Optimization Results ─────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#cccccc"), spaceAfter=4))
    story.append(Paragraph("2. Optimization Results", section_style))

    optim_descriptions = {
        "Energy Model Report": (
            "A random-search optimizer sampled 1,500 parameter combinations and identified the "
            "configuration that minimises predicted energy consumption. The optimal settings below "
            "represent the lowest-energy operating point found."
        ),
        "Efficiency Report": (
            "The optimizer searched for parameter combinations that maximise the predicted efficiency "
            "class score. Settings below represent the highest-efficiency operating point found."
        ),
        "Emission Report": (
            "The optimizer searched for the parameter combination that minimises the predicted "
            "emission class score. Settings below represent the lowest-emission operating point found."
        ),
        "Maintenance Report": (
            "The optimizer searched for the parameter combination that minimises predicted "
            "maintenance risk. Settings below represent the lowest-risk operating point found."
        ),
    }
    story.append(Paragraph(
        optim_descriptions.get(title, "Optimizer search results:"), body_style
    ))
    story.append(Spacer(1, 2*mm))

    if optim_results:
        optim_labels = {
            "load":           ("Optimal Production Load",  "fraction"),
            "cycle":          ("Optimal Cycle Time",       "seconds"),
            "temp":           ("Optimal Temperature",      "°C"),
            "speed":          ("Optimal Axis Speed",       "m/s"),
            "score":          ("Predicted Score",          "model units"),
            "raw_prediction": ("Predicted Class",         ""),
        }
        opt_data = [["Parameter", "Value", "Unit"]]
        for k, v in optim_results.items():
            label, unit = optim_labels.get(k, (k.replace("_", " ").title(), ""))
            opt_data.append([
                Paragraph(label, kv_key_style),
                Paragraph(f"{v:.3f}" if isinstance(v, float) else str(v), kv_val_style),
                Paragraph(unit, kv_key_style),
            ])
        t2 = Table(opt_data, colWidths=[90*mm, 45*mm, 35*mm])
        t2.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#27ae60")),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",    (0, 0), (-1, 0),  base_font),
            ("FONTSIZE",    (0, 0), (-1, 0),  10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#f0fff4"), colors.white]),
            ("GRID",        (0, 0), (-1, -1),  0.4, colors.HexColor("#dddddd")),
            ("LEFTPADDING", (0, 0), (-1, -1),  6),
            ("RIGHTPADDING",(0, 0), (-1, -1),  6),
            ("TOPPADDING",  (0, 0), (-1, -1),  4),
            ("BOTTOMPADDING",(0,0), (-1, -1),  4),
        ]))
        story.append(t2)
    else:
        story.append(Paragraph(
            "No optimization was run for this report. Use the dashboard optimization "
            "button and re-download to include results.", body_style
        ))
    story.append(Spacer(1, 6*mm))

    # ── Section 3 — Visual Analytics ─────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#cccccc"), spaceAfter=4))
    story.append(Paragraph("3. Visual Analytics", section_style))
    story.append(Paragraph(
        "The charts below were generated from the model's training data and the current "
        "what-if simulation. Each chart is accompanied by an interpretive caption.",
        body_style
    ))
    story.append(Spacer(1, 3*mm))

    # Per-report chart captions — in the same order figs are passed
    chart_captions = {
        "Energy Model Report": [
            ("Monte Carlo Energy Variability",
             "Each line represents one simulation run with slight parameter noise. "
             "The dashed orange line marks the mean predicted energy. The shaded band "
             "shows the 5th–95th percentile range, capturing 90% of likely outcomes."),
            ("Distribution of Simulated Energy Outcomes",
             "A frequency histogram of all Monte Carlo outcomes. The P5 and P95 markers "
             "define the realistic worst- and best-case energy range under normal variation. "
             "A narrow, left-shifted distribution indicates stable, low-energy operation."),
            ("Feature Correlation Map",
             "Pearson correlation coefficients between all training features. "
             "Strong positive correlations (dark blue) indicate features that rise together; "
             "strong negative correlations (dark red) indicate inverse relationships. "
             "High correlation with the energy target highlights the most influential features."),
            ("Production Load vs Energy Consumption",
             "Each point is one training sample. The upward trend confirms that higher "
             "production load is a primary driver of energy consumption. Scatter around "
             "the trend reflects the influence of other variables such as temperature and speed."),
        ],
        "Efficiency Report": [
            ("Efficiency Class Distribution",
             "The count of training samples in each efficiency class (Low / Medium / High). "
             "Class imbalance here can affect model performance — a balanced distribution "
             "leads to more reliable predictions across all classes."),
            ("Load vs Cycle Time by Efficiency Class",
             "Each point is one training sample, coloured by its efficiency class. "
             "Clusters indicate the operating regions that define each tier. "
             "Points where classes overlap represent conditions the model treats as ambiguous."),
        ],
        "Emission Report": [
            ("Emission Class Distribution",
             "The count of training samples in each emission class (Low / Medium / High). "
             "A high proportion of High-emission samples suggests the fleet frequently "
             "operates under conditions that drive elevated emissions."),
            ("Load vs Temperature by Emission Class",
             "Each point is a training sample coloured by emission class. "
             "A diagonal pattern (top-right = High) confirms that high load combined "
             "with high temperature is the dominant driver of elevated emissions."),
        ],
        "Maintenance Report": [
            ("Maintenance Risk Distribution",
             "Count of training samples per risk class (Low / Medium / High). "
             "A large High-risk proportion indicates that the equipment frequently "
             "operates near or beyond safe maintenance thresholds."),
            ("Vibration vs Tool Wear by Maintenance Risk",
             "Each point is a training sample. High-risk samples tend to cluster "
             "in the upper-right region, confirming that elevated vibration combined "
             "with high tool wear is the strongest predictor of imminent maintenance need."),
        ],
    }

    captions = chart_captions.get(title, [])

    for i, fig in enumerate(figs):
        img = fig_to_img(fig)
        story.append(Image(img, width=155*mm, height=90*mm))
        if i < len(captions):
            chart_title, chart_caption = captions[i]
            story.append(Paragraph(
                f"Figure {i+1}: {chart_title}", caption_style
            ))
            story.append(Paragraph(chart_caption, caption_style))
        story.append(Spacer(1, 4*mm))

    # ── Footer note ───────────────────────────────────────────
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#cccccc"), spaceAfter=4))
    story.append(Paragraph(
        "This report was auto-generated by the Automotive Decision Intelligence Platform. "
        "Predictions are based on trained ML models and should be validated against "
        "live sensor data before operational decisions are made.",
        ParagraphStyle("Footer", fontName=base_font, fontSize=8,
                       textColor=colors.HexColor("#888888"), alignment=TA_CENTER)
    ))

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
# SIDEBAR — CONFIGURABLE PARAMETERS
# ============================================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    st.caption("Adjust these to match your factory's local rates.")

    price_rate = st.number_input(
        "Electricity Price Rate (₹/kWh)",
        min_value=1.0, max_value=50.0, value=10.0, step=0.5,
        help="Your local electricity tariff. Default: ₹10/kWh."
    )
    co2_factor = st.number_input(
        "Grid CO₂ Factor (kg CO₂/kWh)",
        min_value=0.1, max_value=2.0, value=0.85, step=0.05,
        help="CO₂ emitted per kWh from your regional grid. "
             "India average ~0.82, coal-heavy ~1.0, renewables ~0.1."
    )
    st.divider()
    st.caption("🔒 Settings apply to the current session only.")


# ============================================================================================
# TABS
# ============================================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Energy", "📈 Efficiency", "🌍 Emissions", "🛠 Maintenance"
])

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
        cost = predict_cost(energy, price_rate=price_rate)
        co2  = estimate_grid_emission(energy, co2_factor=co2_factor)

        c1, c2, c3 = st.columns(3)
        c1.metric("Energy Consumption", f"{energy:.2f} kWh")
        c2.metric("Cost Estimate", f"₹{cost:.2f}")
        c3.metric("CO₂ Emission", f"{co2:.2f} kg")

    st.divider()
    st.subheader("🚀 What-If Analysis")

    # WHAT-IF MONTE CARLO
    if energy:
        mc = monte_carlo_energy(energy)
        mc_arr = np.array(mc)
        mc_mean = float(np.mean(mc_arr))
        mc_p5   = float(np.percentile(mc_arr, 5))
        mc_p95  = float(np.percentile(mc_arr, 95))

        wif_col1, wif_col2 = st.columns(2)

        # --- Variability line chart ---
        with wif_col1:
            fig_mc_plotly = go.Figure()
            fig_mc_plotly.add_trace(go.Scatter(
                y=mc_arr,
                mode="lines",
                line=dict(color="#4e8cff", width=1.2),
                name="Simulated Energy",
                hovertemplate="Simulation #%{x}<br>Energy: %{y:.3f} kWh<extra></extra>"
            ))
            fig_mc_plotly.add_hline(
                y=mc_mean, line_dash="dash", line_color="#f0a500",
                annotation_text=f"Mean: {mc_mean:.2f} kWh",
                annotation_position="top right"
            )
            fig_mc_plotly.add_hrect(
                y0=mc_p5, y1=mc_p95,
                fillcolor="rgba(78,140,255,0.08)", line_width=0,
                annotation_text="90% confidence band",
                annotation_position="top left"
            )
            fig_mc_plotly.update_layout(
                title=dict(text="Monte Carlo Energy Variability", font=dict(size=14)),
                xaxis_title="Simulation Run",
                yaxis_title="Energy Consumption (kWh)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=50, b=40),
                height=320,
            )
            st.plotly_chart(fig_mc_plotly, use_container_width=True)

        # --- Distribution histogram ---
        with wif_col2:
            fig_hist_plotly = go.Figure()
            fig_hist_plotly.add_trace(go.Histogram(
                x=mc_arr,
                nbinsx=30,
                marker_color="#4e8cff",
                opacity=0.8,
                name="Frequency",
                hovertemplate="Energy: %{x:.3f} kWh<br>Count: %{y}<extra></extra>"
            ))
            fig_hist_plotly.add_vline(
                x=mc_mean, line_dash="dash", line_color="#f0a500",
                annotation_text=f"Mean {mc_mean:.2f} kWh", annotation_position="top right"
            )
            fig_hist_plotly.add_vline(
                x=mc_p5, line_dash="dot", line_color="#e05c5c",
                annotation_text=f"P5 {mc_p5:.2f}", annotation_position="top left"
            )
            fig_hist_plotly.add_vline(
                x=mc_p95, line_dash="dot", line_color="#e05c5c",
                annotation_text=f"P95 {mc_p95:.2f}", annotation_position="top right"
            )
            fig_hist_plotly.update_layout(
                title=dict(text="Distribution of Simulated Energy Outcomes", font=dict(size=14)),
                xaxis_title="Energy Consumption (kWh)",
                yaxis_title="Number of Simulations",
                margin=dict(t=50, b=40),
                height=320,
            )
            st.plotly_chart(fig_hist_plotly, use_container_width=True)

        # keep matplotlib figures for PDF (invisible to UI)
        fig_mc, ax = plt.subplots(figsize=(6, 3))
        ax.plot(mc_arr); ax.set_title("Monte Carlo Variability")
        ax.set_xlabel("Simulation Run"); ax.set_ylabel("Energy (kWh)")
        plt.tight_layout(); plt.close(fig_mc)

        fig_hist, ax2 = plt.subplots(figsize=(6, 3))
        ax2.hist(mc_arr, bins=30); ax2.set_title("Distribution of Energy Outcomes")
        ax2.set_xlabel("Energy (kWh)"); ax2.set_ylabel("Frequency")
        plt.tight_layout(); plt.close(fig_hist)
    else:
        fig_mc = fig_hist = safe_fig()

    st.divider()

    # -------------------------------------------------------------
    # OPTIMIZATION BUTTONS
    # -------------------------------------------------------------
    st.subheader("🚀 Optimization")

    if st.button("Run Grid Optimization (Min Energy)"):
        st.session_state.optim["energy"] = optimize_with_model(
            energy_model, "min"
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

        corr = df.corr(numeric_only=True)

        corr_col1, corr_col2 = st.columns(2)

        with corr_col1:
            fig_corr_plotly = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.columns.tolist(),
                colorscale="RdBu",
                zmid=0,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}",
                hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
                colorbar=dict(title="r", thickness=12)
            ))
            fig_corr_plotly.update_layout(
                title=dict(
                    text="Feature Correlation Map<br><sup>How strongly each variable moves together (−1 to +1)</sup>",
                    font=dict(size=14)
                ),
                xaxis=dict(tickangle=-35),
                margin=dict(t=70, b=80, l=120),
                height=400,
            )
            st.plotly_chart(fig_corr_plotly, use_container_width=True)

        with corr_col2:
            fig_scatter_plotly = px.scatter(
                df,
                x="production_load",
                y="energy",
                color="energy",
                color_continuous_scale="Blues",
                opacity=0.7,
                labels={
                    "production_load": "Production Load (fraction)",
                    "energy": "Energy Consumption (kWh)"
                },
                title="Production Load vs Energy Consumption<br><sup>Higher load generally drives higher energy use</sup>",
                hover_data={col: True for col in df.columns if col in
                            ["cycle_time", "machine_temperature", "axis_speed"]}
            )
            fig_scatter_plotly.update_traces(marker=dict(size=5))
            fig_scatter_plotly.update_layout(
                coloraxis_colorbar=dict(title="kWh", thickness=12),
                margin=dict(t=70, b=40),
                height=400,
            )
            st.plotly_chart(fig_scatter_plotly, use_container_width=True)

        # Row 2: box plot of each numeric feature + energy distribution
        box_col1, box_col2 = st.columns(2)

        with box_col1:
            numeric_cols = [c for c in df.columns if c != "energy"]
            df_melted = df[numeric_cols].melt(var_name="Feature", value_name="Value")
            fig_box = px.box(
                df_melted,
                x="Feature",
                y="Value",
                color="Feature",
                points=False,
                title="Feature Value Distributions<br><sup>Spread and outliers for each input variable in the training set</sup>",
                labels={"Value": "Feature Value", "Feature": ""}
            )
            fig_box.update_layout(
                showlegend=False,
                xaxis_tickangle=-30,
                margin=dict(t=70, b=80),
                height=380,
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with box_col2:
            fig_en_hist = px.histogram(
                df,
                x="energy",
                nbins=35,
                color_discrete_sequence=["#4e8cff"],
                marginal="rug",
                title="Energy Output Distribution<br><sup>Frequency of different energy consumption levels across all training samples</sup>",
                labels={"energy": "Energy Consumption (kWh)", "count": "Frequency"}
            )
            fig_en_hist.update_layout(
                margin=dict(t=70, b=40),
                height=380,
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig_en_hist, use_container_width=True)

        # keep matplotlib figures for PDF
        fig_corr, ax3 = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, cmap="coolwarm", ax=ax3)
        ax3.set_title("Training Correlation Map")
        plt.tight_layout(); plt.close(fig_corr)

        fig_scatter, ax4 = plt.subplots(figsize=(6, 4))
        ax4.scatter(df["production_load"], df["energy"], s=5)
        ax4.set_xlabel("Production Load"); ax4.set_ylabel("Energy (kWh)")
        ax4.set_title("Load vs Energy")
        plt.tight_layout(); plt.close(fig_scatter)

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
        st.session_state.optim["eff"] = optimize_with_model(eff_model, "max")

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

        color_map = {"Low": "#e05c5c", "Medium": "#f0a500", "High": "#4caf73"}
        class_order = ["Low", "Medium", "High"]

        eff_col1, eff_col2 = st.columns(2)

        with eff_col1:
            class_counts = df["efficiency_class"].value_counts().reindex(class_order).reset_index()
            class_counts.columns = ["Efficiency Class", "Count"]
            fig_eff_plotly = px.bar(
                class_counts,
                x="Efficiency Class",
                y="Count",
                color="Efficiency Class",
                color_discrete_map=color_map,
                text="Count",
                title="Efficiency Class Distribution<br><sup>How often each efficiency tier appears in training data</sup>",
                labels={"Count": "Number of Samples", "Efficiency Class": "Efficiency Class"},
                category_orders={"Efficiency Class": class_order}
            )
            fig_eff_plotly.update_traces(textposition="outside",
                hovertemplate="<b>%{x}</b><br>Samples: %{y}<extra></extra>")
            fig_eff_plotly.update_layout(showlegend=False, margin=dict(t=70, b=40), height=360,
                yaxis_title="Number of Training Samples")
            st.plotly_chart(fig_eff_plotly, use_container_width=True)

        with eff_col2:
            # Violin: production_load distribution per efficiency class
            fig_violin = px.violin(
                df,
                x="efficiency_class",
                y="production_load",
                color="efficiency_class",
                color_discrete_map=color_map,
                box=True,
                points="outliers",
                category_orders={"efficiency_class": class_order},
                title="Production Load Distribution by Efficiency Class<br><sup>Load range that each efficiency tier typically operates at</sup>",
                labels={"efficiency_class": "Efficiency Class", "production_load": "Production Load (fraction)"}
            )
            fig_violin.update_layout(showlegend=False, margin=dict(t=70, b=40), height=360)
            st.plotly_chart(fig_violin, use_container_width=True)

        eff_col3, eff_col4 = st.columns(2)

        with eff_col3:
            # Strip plot: cycle_time per class — avoids overlap by using jitter
            fig_strip = px.strip(
                df,
                x="efficiency_class",
                y="cycle_time",
                color="efficiency_class",
                color_discrete_map=color_map,
                stripmode="overlay",
                category_orders={"efficiency_class": class_order},
                title="Cycle Time Spread by Efficiency Class<br><sup>Each dot is one training sample — see the range per class</sup>",
                labels={"efficiency_class": "Efficiency Class", "cycle_time": "Cycle Time (s)"}
            )
            fig_strip.update_traces(jitter=0.4, marker=dict(size=4, opacity=0.5))
            fig_strip.update_layout(showlegend=False, margin=dict(t=70, b=40), height=360)
            st.plotly_chart(fig_strip, use_container_width=True)

        with eff_col4:
            # Box: machine temperature grouped by class
            fig_box2 = px.box(
                df,
                x="efficiency_class",
                y="machine_temperature",
                color="efficiency_class",
                color_discrete_map=color_map,
                points="outliers",
                category_orders={"efficiency_class": class_order},
                title="Machine Temperature by Efficiency Class<br><sup>Temperature ranges associated with each efficiency level</sup>",
                labels={"efficiency_class": "Efficiency Class", "machine_temperature": "Machine Temperature (°C)"}
            )
            fig_box2.update_layout(showlegend=False, margin=dict(t=70, b=40), height=360)
            st.plotly_chart(fig_box2, use_container_width=True)

        # keep matplotlib figures for PDF
        fig_eff, ax = plt.subplots(figsize=(6, 3))
        sns.countplot(x=df["efficiency_class"], ax=ax, order=class_order)
        ax.set_title("Efficiency Class Distribution"); ax.set_xlabel("Efficiency Class"); ax.set_ylabel("Count")
        plt.tight_layout(); plt.close(fig_eff)

        fig_sc, ax2 = plt.subplots(figsize=(6, 3))
        ax2.scatter(df["production_load"], df["cycle_time"], s=5)
        ax2.set_xlabel("Production Load"); ax2.set_ylabel("Cycle Time (s)")
        ax2.set_title("Load vs Cycle Time")
        plt.tight_layout(); plt.close(fig_sc)

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
        st.session_state.optim["em"] = optimize_with_model(emiss_model, "min")

    best_em = st.session_state.optim["em"]

    if best_em:
        st.success(
            f"**Optimal Settings →** Load={best_energy['load']:.2f}, "
            f"Cycle={best_energy['cycle']:.1f}, Temp={best_energy['temp']:.1f}, "
            f"Speed={best_energy['speed']:.2f}  \n\n"
            f"🌍 Lowest Emission Score: **{best_em['raw_prediction']}**"
        )

    # Visuals
    try:
        df = st.session_state.get("emission_raw_data")

        em_color_map = {"Low": "#4caf73", "Medium": "#f0a500", "High": "#e05c5c"}
        em_order = ["Low", "Medium", "High"]

        em_col1, em_col2 = st.columns(2)

        with em_col1:
            em_counts = df["emission_class"].value_counts().reindex(em_order).reset_index()
            em_counts.columns = ["Emission Class", "Count"]
            fig_em_plotly = px.bar(
                em_counts,
                x="Emission Class",
                y="Count",
                color="Emission Class",
                color_discrete_map=em_color_map,
                text="Count",
                title="Emission Class Distribution<br><sup>Proportion of Low / Medium / High emission events in training data</sup>",
                labels={"Count": "Number of Samples", "Emission Class": "Emission Class"},
                category_orders={"Emission Class": em_order}
            )
            fig_em_plotly.update_traces(textposition="outside",
                hovertemplate="<b>%{x}</b> emissions<br>Samples: %{y}<extra></extra>")
            fig_em_plotly.update_layout(showlegend=False, margin=dict(t=70, b=40), height=360,
                yaxis_title="Number of Training Samples")
            st.plotly_chart(fig_em_plotly, use_container_width=True)

        with em_col2:
            # True 2D scatter: load (continuous X) vs temperature (continuous Y), coloured by class
            fig_sc_plotly = px.scatter(
                df,
                x="production_load",
                y="machine_temperature",
                color="emission_class",
                color_discrete_map=em_color_map,
                opacity=0.6,
                size_max=6,
                labels={
                    "production_load": "Production Load (fraction)",
                    "machine_temperature": "Machine Temperature (°C)",
                    "emission_class": "Emission Class"
                },
                title="Load vs Temperature by Emission Class<br><sup>High load + high temperature correlates strongly with elevated emissions</sup>",
                category_orders={"emission_class": em_order}
            )
            fig_sc_plotly.update_traces(marker=dict(size=5))
            fig_sc_plotly.update_layout(margin=dict(t=70, b=80), height=380,
                legend=dict(title="Emission Class", orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5))
            st.plotly_chart(fig_sc_plotly, use_container_width=True)

        em_col3, em_col4 = st.columns(2)

        with em_col3:
            # Violin: machine temperature per emission class
            fig_em_violin = px.violin(
                df,
                x="emission_class",
                y="machine_temperature",
                color="emission_class",
                color_discrete_map=em_color_map,
                box=True,
                points="outliers",
                category_orders={"emission_class": em_order},
                title="Machine Temperature Distribution by Emission Class<br><sup>Temperature spread within each emission tier</sup>",
                labels={"emission_class": "Emission Class", "machine_temperature": "Machine Temperature (°C)"}
            )
            fig_em_violin.update_layout(showlegend=False, margin=dict(t=70, b=40), height=360)
            st.plotly_chart(fig_em_violin, use_container_width=True)

        with em_col4:
            # Strip plot: cycle time per emission class — avoids single-line collapse
            fig_em_strip = px.strip(
                df,
                x="emission_class",
                y="cycle_time",
                color="emission_class",
                color_discrete_map=em_color_map,
                stripmode="overlay",
                category_orders={"emission_class": em_order},
                title="Cycle Time Spread by Emission Class<br><sup>Longer cycles tend to accumulate more emissions — each dot is one sample</sup>",
                labels={"emission_class": "Emission Class", "cycle_time": "Cycle Time (s)"}
            )
            fig_em_strip.update_traces(jitter=0.4, marker=dict(size=4, opacity=0.45))
            fig_em_strip.update_layout(showlegend=False, margin=dict(t=70, b=40), height=360)
            st.plotly_chart(fig_em_strip, use_container_width=True)

        # keep matplotlib figures for PDF
        fig_em, ax = plt.subplots(figsize=(6, 3))
        sns.countplot(x=df["emission_class"], ax=ax, order=em_order)
        ax.set_title("Emission Class Distribution"); ax.set_xlabel("Emission Class"); ax.set_ylabel("Count")
        plt.tight_layout(); plt.close(fig_em)

        fig_sc, ax2 = plt.subplots(figsize=(6, 3))
        ax2.scatter(df["production_load"], df["machine_temperature"], s=5)
        ax2.set_xlabel("Production Load"); ax2.set_ylabel("Machine Temperature (°C)")
        ax2.set_title("Load vs Temperature by Emission Class")
        plt.tight_layout(); plt.close(fig_sc)

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
        st.session_state.optim["maint"] = optimize_with_model(maint_model, "min")

    best_m = st.session_state.optim["maint"]

    if best_m:
        st.success(
            f"**Optimal Settings →** Load={best_energy['load']:.2f}, "
            f"Cycle={best_energy['cycle']:.1f}, Temp={best_energy['temp']:.1f}, "
            f"Speed={best_energy['speed']:.2f}  \n\n"
            f"🛠 Minimum Risk Level: **{best_m['raw_prediction']}**"
        )

    # Visuals
    try:
        df = st.session_state.get("maintenance_raw_data")

        mt_color_map = {"Low": "#4caf73", "Medium": "#f0a500", "High": "#e05c5c"}
        mt_order = ["Low", "Medium", "High"]

        mt_col1, mt_col2 = st.columns(2)

        with mt_col1:
            mt_counts = df["maintenance_class"].value_counts().reindex(mt_order).reset_index()
            mt_counts.columns = ["Maintenance Risk", "Count"]
            fig_mt_plotly = px.bar(
                mt_counts,
                x="Maintenance Risk",
                y="Count",
                color="Maintenance Risk",
                color_discrete_map=mt_color_map,
                text="Count",
                title="Maintenance Risk Distribution<br><sup>Breakdown of Low / Medium / High risk events in training data</sup>",
                labels={"Count": "Number of Samples", "Maintenance Risk": "Risk Level"},
                category_orders={"Maintenance Risk": mt_order}
            )
            fig_mt_plotly.update_traces(
                textposition="outside",
                hovertemplate="<b>%{x}</b> risk<br>Samples: %{y}<extra></extra>"
            )
            fig_mt_plotly.update_layout(showlegend=False, margin=dict(t=70, b=40), height=360,
                yaxis_title="Number of Training Samples")
            st.plotly_chart(fig_mt_plotly, use_container_width=True)

        with mt_col2:
            # True 2D scatter: vibration (X) vs tool_wear (Y) — both continuous, coloured by class
            fig_sc_plotly = px.scatter(
                df,
                x="vibration_level",
                y="tool_wear",
                color="maintenance_class",
                color_discrete_map=mt_color_map,
                opacity=0.6,
                labels={
                    "vibration_level": "Vibration Level",
                    "tool_wear": "Tool Wear (fraction)",
                    "maintenance_class": "Risk Level"
                },
                title="Vibration vs Tool Wear by Maintenance Risk<br><sup>High vibration combined with high tool wear strongly predicts maintenance need</sup>",
                category_orders={"maintenance_class": mt_order}
            )
            fig_sc_plotly.update_traces(marker=dict(size=5))
            fig_sc_plotly.update_layout(margin=dict(t=70, b=80), height=380,
                legend=dict(title="Risk Level", orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5))
            st.plotly_chart(fig_sc_plotly, use_container_width=True)

        mt_col3, mt_col4 = st.columns(2)

        with mt_col3:
            # Violin: vibration level per maintenance class — shows spread, not a flat line
            fig_mt_violin = px.violin(
                df,
                x="maintenance_class",
                y="vibration_level",
                color="maintenance_class",
                color_discrete_map=mt_color_map,
                box=True,
                points="outliers",
                category_orders={"maintenance_class": mt_order},
                title="Vibration Level Distribution by Risk Class<br><sup>Higher risk classes exhibit a wider and higher vibration range</sup>",
                labels={"maintenance_class": "Risk Level", "vibration_level": "Vibration Level"}
            )
            fig_mt_violin.update_layout(showlegend=False, margin=dict(t=70, b=40), height=360)
            st.plotly_chart(fig_mt_violin, use_container_width=True)

        with mt_col4:
            # Box: oil_quality per maintenance class
            fig_mt_box = px.box(
                df,
                x="maintenance_class",
                y="oil_quality",
                color="maintenance_class",
                color_discrete_map=mt_color_map,
                points="outliers",
                category_orders={"maintenance_class": mt_order},
                title="Oil Quality by Maintenance Risk<br><sup>Degraded oil quality is a leading indicator of high maintenance risk</sup>",
                labels={"maintenance_class": "Risk Level", "oil_quality": "Oil Quality Score"}
            )
            fig_mt_box.update_layout(showlegend=False, margin=dict(t=70, b=40), height=360)
            st.plotly_chart(fig_mt_box, use_container_width=True)

        # keep matplotlib figures for PDF
        fig_mt, ax = plt.subplots(figsize=(6, 3))
        sns.countplot(x=df["maintenance_class"], ax=ax, order=mt_order)
        ax.set_title("Maintenance Risk Distribution")
        ax.set_xlabel("Risk Level"); ax.set_ylabel("Count")
        plt.tight_layout(); plt.close(fig_mt)

        fig_sc, ax2 = plt.subplots(figsize=(6, 3))
        ax2.scatter(df["vibration_level"], df["tool_wear"], s=5)
        ax2.set_xlabel("Vibration Level"); ax2.set_ylabel("Tool Wear")
        ax2.set_title("Vibration vs Tool Wear")
        plt.tight_layout(); plt.close(fig_sc)

    except:
        fig_mt = fig_sc = safe_fig()

    pdf = pdf_export("Maintenance Report",
                     [fig_mt, fig_sc],
                     {"load": m_load, "cycle": m_cycle, "temp": m_temp, "speed": m_speed},
                     best_m or {})

    st.download_button("📄 Download PDF", pdf, file_name="maintenance_report.pdf", key="tab4")