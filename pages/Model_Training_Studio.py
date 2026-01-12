import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from io import BytesIO

# ML
from sklearn.metrics import (
    r2_score, mean_absolute_error, accuracy_score, classification_report,
    confusion_matrix
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier

# Viz
import matplotlib.pyplot as plt
import seaborn as sns

# Modules
from modules.preprocessing import load_dataset, preprocess
from modules.feature_engineering import add_automotive_features
from modules.storage import ensure_dirs, PATHS

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

os.makedirs("/mount/data/models", exist_ok=True)

# ------------------------------------------------------------
# SETUP
# ------------------------------------------------------------
st.set_page_config(page_title="Model Training Studio", layout="wide")
ensure_dirs()

pdfmetrics.registerFont(TTFont("DejaVu", "assets/fonts/DejaVuSans.ttf"))
st.title("🛠 Automotive Sustainability – Model Training Studio")


# ------------------------------------------------------------
# TARGET DETECTOR
# ------------------------------------------------------------
def find_target(df):
    for t in ["energy", "efficiency_class", "emission_class", "maintenance_class"]:
        if t in df.columns:
            return t
    return None


# ------------------------------------------------------------
# SAVE METADATA FOR DASHBOARD MODELS
# ------------------------------------------------------------
def save_metadata(feature_order, df, model_name):
    metadata = {
        "feature_order": feature_order,
        "feature_means": df.mean(numeric_only=True).to_dict()
    }

    joblib.dump(metadata, f"/mount/data/models/{model_name}_metadata.pkl")


# ------------------------------------------------------------
# AUTO ML — ENERGY REGRESSION
# ------------------------------------------------------------
def train_energy(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=350),
        "GradientBoost": GradientBoostingRegressor(),
        "Linear": LinearRegression()
    }

    best_score = -999
    best_model = None
    best_name = ""

    for name, model in models.items():
        model.fit(X, y)
        pred = model.predict(X)
        score = r2_score(y, pred)

        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    metrics = {
        "best_model": best_name,
        "r2": float(best_score),
        "mae": float(mean_absolute_error(y, best_model.predict(X)))
    }

    # Save
    joblib.dump(best_model, "/mount/data/models/energy_model.pkl")
    save_metadata(list(X.columns), X, "energy")

    return best_model, metrics


# ------------------------------------------------------------
# AUTO ML — CLASSIFIERS
# ------------------------------------------------------------
def train_classifier(df, target, model_path, name):
    X = df.drop(columns=[target])
    y = df[target]

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=250),
        "LogisticRegression": LogisticRegression(max_iter=500),
        "MLPClassifier": MLPClassifier(hidden_layer_sizes=(64,), max_iter=500)
    }

    best_acc = -1
    best_model = None
    best_name = ""

    for name_, model in models.items():
        try:
            model.fit(X, y)
            pred = model.predict(X)
            acc = accuracy_score(y, pred)
            if acc > best_acc:
                best_acc = acc
                best_name = name_
                best_model = model
        except:
            pass

    metrics = {
        "best_model": best_name,
        "accuracy": float(best_acc),
        "report": classification_report(y, best_model.predict(X), output_dict=True)
    }

    joblib.dump(best_model, model_path)
    save_metadata(list(X.columns), X, name)

    return best_model, metrics


# ------------------------------------------------------------
# VISUALIZATION HELPERS
# ------------------------------------------------------------
def feature_importance_plot(model, X):
    fig, ax = plt.subplots(figsize=(5, 3))

    if hasattr(model, "feature_importances_"):
        sns.barplot(x=model.feature_importances_, y=X.columns, ax=ax)
        ax.set_title("Feature Importance")
    else:
        ax.text(0.5, 0.5, "Feature importance not available", ha="center")

    return fig


def confusion_matrix_plot(model, X, y):
    fig, ax = plt.subplots(figsize=(5, 3))
    pred = model.predict(X)
    cm = confusion_matrix(y, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    return fig


# ------------------------------------------------------------
# PDF GENERATOR
# ------------------------------------------------------------
def export_pdf(title, metrics, figs):
    buf = BytesIO()
    styles = getSampleStyleSheet()
    styles["Normal"].fontName = "DejaVu"
    styles["Heading1"].fontName = "DejaVu"
    doc = SimpleDocTemplate(buf, pagesize=A4)

    story = []
    story.append(Paragraph(title, styles["Heading1"]))
    story.append(Spacer(1, 15))

    for k, v in metrics.items():
        story.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    story.append(Spacer(1, 20))

    for fig in figs:
        img = BytesIO()
        fig.savefig(img, format="png", bbox_inches="tight")
        img.seek(0)
        story.append(Image(img, width=380, height=230))
        story.append(Spacer(1, 12))

    doc.build(story)
    buf.seek(0)

    return buf


# ------------------------------------------------------------
# TAB HANDLER
# ------------------------------------------------------------
def handle_tab(upload_key, model_title, train_fn, model_path, model_name):

    uploaded = st.file_uploader(
        f"Upload dataset for **{model_title}**",
        type=["csv", "xlsx"],
        key=upload_key
    )

    if not uploaded:
        return

    df = load_dataset(uploaded)
    st.write("### Raw Data Preview", df.head())

    # 1. Clean + Scale
    cleaned, scaled, imputer, scaler = preprocess(df)

    # 2. Add Feature Engineering
    engineered = add_automotive_features(cleaned)
    st.write("### Engineered Features", engineered.head())

    # 3. Detect target
    target = find_target(df)
    if not target:
        st.error("❌ Could not detect target column automatically.")
        return

    # 4. Train
    if st.button(f"🚀 Train {model_title}"):

        with st.spinner("Auto-selecting best model…"):
            model, metrics = train_fn(engineered, target)

        st.success(f"{model_title} trained successfully!")
        st.json(metrics)

        # 5. Visuals
        figs = []

        X = engineered.drop(columns=[target])
        y = engineered[target]

        if target != "energy":
            figs.append(confusion_matrix_plot(model, X, y))

        figs.append(feature_importance_plot(model, X))

        # 6. PDF
        pdf = export_pdf(model_title, metrics, figs)

        st.download_button(
            f"📄 Download {model_title} Report",
            data=pdf,
            file_name=f"{model_name}_report.pdf",
            mime="application/pdf"
        )


# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Energy Model",
    "📈 Efficiency Classifier",
    "🌍 Emission Classifier",
    "🛠 Maintenance DL"
])

with tab1:
    handle_tab(
        upload_key="energy_data",
        model_title="Energy Prediction Model",
        train_fn=lambda df, t: train_energy(df, t),
        model_path="/mount/data/models/energy_model.pkl",
        model_name="energy"
    )

with tab2:
    handle_tab(
        upload_key="efficiency_data",
        model_title="Efficiency Classifier",
        train_fn=lambda df, t: train_classifier(df, t, "/mount/data/models/efficiency_model.pkl", "efficiency"),
        model_path="/mount/data/models/efficiency_model.pkl",
        model_name="efficiency"
    )

with tab3:
    handle_tab(
        upload_key="emission_data",
        model_title="Emission Classifier",
        train_fn=lambda df, t: train_classifier(df, t, "/mount/data/models/emission_model.pkl", "emission"),
        model_path="/mount/data/models/emission_model.pkl",
        model_name="emission"
    )

with tab4:
    handle_tab(
        upload_key="maintenance_data",
        model_title="Maintenance DL Model",
        train_fn=lambda df, t: train_classifier(df, t, "/mount/data/models/maintenance_model.pkl", "maintenance"),
        model_path="/mount/data/models/maintenance_dl.pkl",
        model_name="maintenance"
    )
