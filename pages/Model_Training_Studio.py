import streamlit as st
import pandas as pd
import numpy as np
import os
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
# AUTO ML — ENERGY REGRESSION
# ------------------------------------------------------------
def train_energy(df, target):
    from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV

    st.session_state[f"energy_raw_data"] = df.copy()
    X = df.drop(columns=[target])
    y = df[target]

    # 80/20 split — test set is held out entirely from model selection
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Candidate models with hyperparameter search spaces
    candidates = {
        "RandomForest": (
            RandomForestRegressor(random_state=42),
            {
                "n_estimators":    [100, 200, 300],
                "max_depth":       [6, 8, 10, None],
                "min_samples_leaf":[5, 10, 20],
                "max_features":    [0.5, 0.6, 0.8],
            }
        ),
        "GradientBoost": (
            GradientBoostingRegressor(random_state=42),
            {
                "n_estimators":    [100, 200],
                "max_depth":       [3, 4, 5],
                "learning_rate":   [0.03, 0.05, 0.1],
                "subsample":       [0.7, 0.8, 1.0],
                "min_samples_leaf":[5, 10],
            }
        ),
        "Linear": (LinearRegression(), {}),
    }

    best_score = -999
    best_model = None
    best_name  = ""

    for name, (model, param_dist) in candidates.items():
        if param_dist:
            search = RandomizedSearchCV(
                model, param_dist,
                n_iter=12, cv=5, scoring="r2",
                random_state=42, n_jobs=-1
            )
            search.fit(X_train, y_train)
            cv_mean     = float(search.best_score_)
            tuned_model = search.best_estimator_
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
            cv_mean   = float(cv_scores.mean())
            model.fit(X_train, y_train)
            tuned_model = model

        if cv_mean > best_score:
            best_score = cv_mean
            best_name  = name
            best_model = tuned_model

    train_r2 = float(r2_score(y_train, best_model.predict(X_train)))
    test_r2  = float(r2_score(y_test,  best_model.predict(X_test)))
    test_mae = float(mean_absolute_error(y_test, best_model.predict(X_test)))

    metrics = {
        "best_model": best_name,
        "r2":         best_score,
        "train_r2":   train_r2,
        "test_r2":    test_r2,
        "mae":        test_mae,
    }

    st.session_state["energy_model"]    = best_model
    st.session_state["energy_metadata"] = {"feature_order": list(X.columns)}

    return best_model, metrics


# ------------------------------------------------------------
# AUTO ML — CLASSIFIERS
# ------------------------------------------------------------
def train_classifier(df, target, name):
    from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV

    st.session_state[f"{name}_raw_data"] = df.copy()
    X = df.drop(columns=[target])
    y = df[target]

    # 80/20 stratified split — test set held out entirely from model selection
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Candidate models with hyperparameter search spaces
    candidates = {
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators":    [100, 200, 300],
                "max_depth":       [6, 8, 10, None],
                "min_samples_leaf":[10, 15, 25],
                "max_features":    [0.5, 0.6, 0.8],
            }
        ),
        "LogisticRegression": (
            LogisticRegression(max_iter=500, random_state=42),
            {
                "C": [0.1, 0.3, 0.5, 1.0, 2.0],
            }
        ),
        "MLPClassifier": (
            MLPClassifier(max_iter=500, random_state=42),
            {
                "hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
                "alpha":              [0.001, 0.01, 0.05],
                "learning_rate_init": [0.001, 0.005],
            }
        ),
    }

    best_acc   = -1
    best_model = None
    best_name  = ""

    for name_, (model, param_dist) in candidates.items():
        try:
            if param_dist:
                search = RandomizedSearchCV(
                    model, param_dist,
                    n_iter=10, cv=5, scoring="accuracy",
                    random_state=42, n_jobs=-1
                )
                search.fit(X_train, y_train)
                cv_mean     = float(search.best_score_)
                tuned_model = search.best_estimator_
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
                cv_mean   = float(cv_scores.mean())
                model.fit(X_train, y_train)
                tuned_model = model

            if cv_mean > best_acc:
                best_acc   = cv_mean
                best_name  = name_
                best_model = tuned_model
        except:
            pass

    train_acc = float(accuracy_score(y_train, best_model.predict(X_train)))
    test_acc  = float(accuracy_score(y_test,  best_model.predict(X_test)))

    metrics = {
        "best_model":     best_name,
        "accuracy":       best_acc,
        "train_accuracy": train_acc,
        "test_accuracy":  test_acc,
        "report":         classification_report(y_test, best_model.predict(X_test), output_dict=True)
    }

    st.session_state[f"{name}_model"]    = best_model
    st.session_state[f"{name}_metadata"] = {"feature_order": list(X.columns)}

    return best_model, metrics


# ------------------------------------------------------------
# VISUALIZATION HELPERS
# ------------------------------------------------------------
def feature_importance_plot(model, X, y=None):
    from sklearn.inspection import permutation_importance

    fig, ax = plt.subplots(figsize=(6, 4))

    if hasattr(model, "feature_importances_"):
        # Tree-based models: Gini / impurity importance
        importances = model.feature_importances_
        sorted_idx  = np.argsort(importances)
        sns.barplot(x=importances[sorted_idx], y=np.array(X.columns)[sorted_idx], ax=ax)
        ax.set_title("Feature Importance (Gini)")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")

    elif hasattr(model, "coef_"):
        # Linear / Logistic Regression: use absolute coefficients
        coef = model.coef_
        if coef.ndim > 1:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef.flatten())
        sorted_idx = np.argsort(importances)
        sns.barplot(x=importances[sorted_idx], y=np.array(X.columns)[sorted_idx], ax=ax)
        ax.set_title("Feature Coefficients (Absolute Value)")
        ax.set_xlabel("|Coefficient|")
        ax.set_ylabel("Feature")

    elif y is not None:
        # MLP or other black-box: fall back to permutation importance
        try:
            result      = permutation_importance(model, X, y, n_repeats=10,
                                                 random_state=42, n_jobs=-1)
            importances = result.importances_mean
            sorted_idx  = np.argsort(importances)
            sns.barplot(x=importances[sorted_idx], y=np.array(X.columns)[sorted_idx], ax=ax)
            ax.set_title("Feature Importance (Permutation)")
            ax.set_xlabel("Mean Accuracy Drop")
            ax.set_ylabel("Feature")
        except Exception as e:
            ax.text(0.5, 0.5, f"Could not compute importance:\n{e}",
                    ha="center", va="center", fontsize=10, wrap=True)
            ax.axis("off")

    else:
        ax.text(0.5, 0.5, "Feature importance not available\nfor this model type",
                ha="center", va="center", fontsize=11)
        ax.axis("off")

    plt.tight_layout()
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
    import datetime
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import HRFlowable, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

    buf = BytesIO()
    base_font = "DejaVu"

    # ── Styles ────────────────────────────────────────────────
    cover_title_style = ParagraphStyle("CoverTitle", fontName=base_font,
        fontSize=22, leading=28, alignment=TA_CENTER, spaceAfter=6)
    cover_sub_style   = ParagraphStyle("CoverSub", fontName=base_font,
        fontSize=11, leading=14, alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"), spaceAfter=4)
    section_style     = ParagraphStyle("Section", fontName=base_font,
        fontSize=13, leading=16, textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=6, spaceBefore=14)
    body_style        = ParagraphStyle("Body", fontName=base_font,
        fontSize=10, leading=14, spaceAfter=3)
    caption_style     = ParagraphStyle("Caption", fontName=base_font,
        fontSize=9, leading=12, textColor=colors.HexColor("#444444"),
        alignment=TA_CENTER, spaceAfter=10, spaceBefore=4)
    kv_key_style      = ParagraphStyle("KVKey", fontName=base_font,
        fontSize=10, leading=13, textColor=colors.HexColor("#333333"))
    kv_val_style      = ParagraphStyle("KVVal", fontName=base_font,
        fontSize=10, leading=13, textColor=colors.HexColor("#1a1a2e"))
    footer_style      = ParagraphStyle("Footer", fontName=base_font,
        fontSize=8, textColor=colors.HexColor("#888888"), alignment=TA_CENTER)

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
                             color=colors.HexColor("#4e8cff"), spaceAfter=6*mm))

    # ── Per-model intro ───────────────────────────────────────
    intros = {
        "Energy Prediction Model": (
            "This report documents the results of AutoML training for the Energy Prediction model. "
            "Three regression algorithms were evaluated — Random Forest, Gradient Boosting, and "
            "Linear Regression — and the best-performing model was automatically selected based on "
            "R² score. The report includes performance metrics, feature importance, and training "
            "data visualizations to support model interpretation."
        ),
        "Efficiency Classifier": (
            "This report documents AutoML training for the Efficiency Classifier, which predicts "
            "whether a machine operates at Low, Medium, or High efficiency. Three classifiers were "
            "evaluated — Random Forest, Logistic Regression, and MLP — and the best was selected "
            "by accuracy. The confusion matrix and feature importance below reveal how well the "
            "model distinguishes between efficiency tiers."
        ),
        "Emission Classifier": (
            "This report documents AutoML training for the Emission Classifier, which predicts "
            "the emission class (Low / Medium / High) for a given set of operating conditions. "
            "The confusion matrix shows per-class prediction accuracy, while feature importance "
            "identifies which operating variables most strongly drive emission class."
        ),
        "Maintenance Model": (
            "This report documents AutoML training for the Maintenance Risk classifier, which "
            "predicts whether maintenance risk is Low, Medium, or High. Understanding which "
            "features drive risk classification helps maintenance teams prioritise inspections "
            "and prevent unplanned downtime."
        ),
    }
    story.append(Paragraph(
        intros.get(title, "AutoML training report for the automotive intelligence platform."),
        body_style
    ))
    story.append(Spacer(1, 6*mm))

    # ── Section 1 — Model Performance ────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#cccccc"), spaceAfter=4))
    story.append(Paragraph("1. Model Performance Metrics", section_style))

    is_regression = "r2" in metrics

    if is_regression:
        story.append(Paragraph(
            "The energy model is evaluated as a regression task. Model selection used 5-fold CV on "
            "the training set (80% of data). Final metrics are reported on the held-out test set (20%) "
            "which was never seen during training or model selection.",
            body_style
        ))
        story.append(Spacer(1, 3*mm))

        metric_data = [
            ["Metric", "Value", "Interpretation"],
            [Paragraph("Best Algorithm Selected", kv_key_style),
             Paragraph(str(metrics.get("best_model", "—")), kv_val_style),
             Paragraph("Algorithm with highest CV R² on training set", kv_key_style)],
            [Paragraph("CV R² (Train Set)", kv_key_style),
             Paragraph(f"{metrics.get('r2', 0):.4f}", kv_val_style),
             Paragraph("5-fold CV R² — used for model selection", kv_key_style)],
            [Paragraph("Train R²", kv_key_style),
             Paragraph(f"{metrics.get('train_r2', 0):.4f}", kv_val_style),
             Paragraph("R² on training data — compare with Test R² for overfitting check", kv_key_style)],
            [Paragraph("Test R² (Held-Out)", kv_key_style),
             Paragraph(f"{metrics.get('test_r2', 0):.4f}", kv_val_style),
             Paragraph("Unbiased generalisation estimate on 20% held-out test set", kv_key_style)],
            [Paragraph("Test MAE (Held-Out)", kv_key_style),
             Paragraph(f"{metrics.get('mae', 0):.4f} kWh", kv_val_style),
             Paragraph("Mean Absolute Error on test set in kWh; lower is better", kv_key_style)],
        ]
    else:
        story.append(Paragraph(
            "The classifier uses 5-fold CV on the training set (80% of data) for model selection. "
            "Final accuracy and the classification report are computed on the held-out test set (20%) "
            "which was never seen during training or model selection.",
            body_style
        ))
        story.append(Spacer(1, 3*mm))

        report = metrics.get("report", {})
        class_rows = {k: v for k, v in report.items()
                      if k not in ("accuracy", "macro avg", "weighted avg")}

        metric_data = [["Metric", "Value", "Interpretation"]]
        metric_data.append([
            Paragraph("Best Algorithm Selected", kv_key_style),
            Paragraph(str(metrics.get("best_model", "—")), kv_val_style),
            Paragraph("Algorithm with highest CV accuracy on training set", kv_key_style)
        ])
        metric_data.append([
            Paragraph("CV Accuracy (Train Set)", kv_key_style),
            Paragraph(f"{metrics.get('accuracy', 0)*100:.1f}%", kv_val_style),
            Paragraph("5-fold CV accuracy — used for model selection", kv_key_style)
        ])
        metric_data.append([
            Paragraph("Train Accuracy", kv_key_style),
            Paragraph(f"{metrics.get('train_accuracy', 0)*100:.1f}%", kv_val_style),
            Paragraph("Accuracy on training data — compare with Test Accuracy for overfitting check", kv_key_style)
        ])
        metric_data.append([
            Paragraph("Test Accuracy (Held-Out)", kv_key_style),
            Paragraph(f"{metrics.get('test_accuracy', 0)*100:.1f}%", kv_val_style),
            Paragraph("Unbiased generalisation estimate on 20% held-out test set", kv_key_style)
        ])

    # Render summary metrics table
    t = Table(metric_data, colWidths=[65*mm, 40*mm, 65*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  colors.HexColor("#4e8cff")),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",       (0, 0), (-1, 0),  base_font),
        ("FONTSIZE",       (0, 0), (-1, 0),  10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f5f7ff"), colors.white]),
        ("GRID",           (0, 0), (-1, -1),  0.4, colors.HexColor("#dddddd")),
        ("LEFTPADDING",    (0, 0), (-1, -1),  6),
        ("RIGHTPADDING",   (0, 0), (-1, -1),  6),
        ("TOPPADDING",     (0, 0), (-1, -1),  4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1),  4),
    ]))
    story.append(t)
    story.append(Spacer(1, 4*mm))

    # Per-class breakdown table for classifiers
    if not is_regression and class_rows:
        story.append(Paragraph("Per-Class Breakdown", section_style))
        story.append(Paragraph(
            "The table below shows precision, recall, F1-score, and support for each predicted class.",
            body_style
        ))
        story.append(Spacer(1, 2*mm))

        class_data = [["Class", "Precision", "Recall", "F1-Score", "Support"]]
        for cls, vals in sorted(class_rows.items()):
            class_data.append([
                Paragraph(cls, kv_key_style),
                Paragraph(f"{vals['precision']:.2%}", kv_val_style),
                Paragraph(f"{vals['recall']:.2%}", kv_val_style),
                Paragraph(f"{vals['f1-score']:.2%}", kv_val_style),
                Paragraph(str(int(vals['support'])), kv_val_style),
            ])

        # Macro / weighted avg rows
        for avg_key in ("macro avg", "weighted avg"):
            if avg_key in report:
                v = report[avg_key]
                class_data.append([
                    Paragraph(avg_key.title(), kv_key_style),
                    Paragraph(f"{v['precision']:.2%}", kv_val_style),
                    Paragraph(f"{v['recall']:.2%}", kv_val_style),
                    Paragraph(f"{v['f1-score']:.2%}", kv_val_style),
                    Paragraph("—", kv_key_style),
                ])

        t2 = Table(class_data, colWidths=[40*mm, 30*mm, 30*mm, 30*mm, 25*mm])
        t2.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0),  colors.HexColor("#27ae60")),
            ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",       (0, 0), (-1, 0),  base_font),
            ("FONTSIZE",       (0, 0), (-1, 0),  10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -3),
             [colors.HexColor("#f0fff4"), colors.white]),
            ("BACKGROUND",     (0, -2), (-1, -1), colors.HexColor("#e8e8e8")),
            ("GRID",           (0, 0), (-1, -1),  0.4, colors.HexColor("#dddddd")),
            ("LEFTPADDING",    (0, 0), (-1, -1),  6),
            ("RIGHTPADDING",   (0, 0), (-1, -1),  6),
            ("TOPPADDING",     (0, 0), (-1, -1),  4),
            ("BOTTOMPADDING",  (0, 0), (-1, -1),  4),
        ]))
        story.append(t2)
        story.append(Spacer(1, 6*mm))

    # ── Section 2 — Visual Analytics ─────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#cccccc"), spaceAfter=4))
    story.append(Paragraph("2. Visual Analytics", section_style))
    story.append(Paragraph(
        "The charts below were produced during training. They support model interpretation "
        "and help identify which features most influence the model's predictions.",
        body_style
    ))
    story.append(Spacer(1, 3*mm))

    # Chart metadata keyed by title — order matches: confusion matrix first, then feature importance
    chart_info = {
        "Energy Prediction Model": [
            ("Feature Importance",
             "Relative importance of each input feature as determined by the best-selected model. "
             "Features with higher importance have a stronger influence on predicted energy consumption. "
             "Focus operational improvements on the top-ranked features for maximum impact."),
        ],
        "Efficiency Classifier": [
            ("Confusion Matrix",
             "Each cell shows how many samples of the actual class (rows) were predicted as each class (columns). "
             "A perfect model has all counts on the diagonal. Off-diagonal values indicate misclassifications "
             "— review these to understand where the model struggles between adjacent efficiency tiers."),
            ("Feature Importance",
             "Relative importance of each input feature for predicting efficiency class. "
             "Features ranked highest are the strongest discriminators between Low, Medium, and High efficiency."),
        ],
        "Emission Classifier": [
            ("Confusion Matrix",
             "Shows predicted vs actual emission class counts. Diagonal cells are correct predictions; "
             "off-diagonal cells are misclassifications. High off-diagonal counts between adjacent classes "
             "(e.g. Low vs Medium) indicate borderline operating conditions."),
            ("Feature Importance",
             "Relative importance of each input feature for predicting emission class. "
             "The top features represent the primary operational levers for reducing emissions."),
        ],
        "Maintenance Model": [
            ("Confusion Matrix",
             "Shows predicted vs actual maintenance risk counts. Pay particular attention to "
             "High-risk samples misclassified as Low-risk — these false negatives represent "
             "the greatest operational hazard."),
            ("Feature Importance",
             "Relative importance of each input feature for predicting maintenance risk. "
             "High-importance features are the most reliable early-warning indicators and "
             "should be prioritised in sensor monitoring and inspection schedules."),
        ],
    }

    captions = chart_info.get(title, [])
    # If classifier, there are 2 figs (confusion matrix + feature importance);
    # if regression (energy), only 1 fig (feature importance) — align captions accordingly
    if not is_regression and len(captions) < len(figs):
        captions = [("Chart", "Training visualization.")] * len(figs)

    for i, fig in enumerate(figs):
        img = BytesIO()
        fig.savefig(img, format="png", dpi=150, bbox_inches="tight")
        img.seek(0)
        story.append(Image(img, width=140*mm, height=85*mm))
        if i < len(captions):
            chart_title, chart_caption = captions[i]
            story.append(Paragraph(f"Figure {i+1}: {chart_title}", caption_style))
            story.append(Paragraph(chart_caption, caption_style))
        story.append(Spacer(1, 4*mm))

    # ── Footer ────────────────────────────────────────────────
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#cccccc"), spaceAfter=4))
    story.append(Paragraph(
        "This report was auto-generated by the Automotive Decision Intelligence Platform. "
        "Model metrics are computed on training data. For production use, evaluate on a "
        "held-out test set to obtain unbiased performance estimates.",
        footer_style
    ))

    doc.build(story)
    buf.seek(0)
    return buf


# ------------------------------------------------------------
# REQUIRED FEATURES PER MODEL
# ------------------------------------------------------------
REQUIRED_FEATURES = {
    "energy": [
        "production_load", "cycle_time", "machine_temperature", "axis_speed",
        "tool_wear", "ambient_humidity", "vibration_level", "power_factor", "energy"
    ],
    "efficiency": [
        "production_load", "cycle_time", "machine_temperature", "axis_speed",
        "vibration_signal", "power_factor", "efficiency_class"
    ],
    "emission": [
        "production_load", "cycle_time", "machine_temperature", "axis_speed",
        "power_factor", "emission_class"
    ],
    "maintenance": [
        "production_load", "machine_temperature", "axis_speed", "vibration_level",
        "tool_wear", "oil_quality", "pressure", "maintenance_class"
    ],
}


# ------------------------------------------------------------
# TAB HANDLER
# ------------------------------------------------------------
def handle_tab(upload_key, model_title, train_fn, model_name):

    uploaded = st.file_uploader(
        f"Upload dataset for **{model_title}**",
        type=["csv", "xlsx"],
        key=upload_key
    )

    if not uploaded:
        return

    try:
        df = load_dataset(uploaded)
    except ValueError as e:
        st.error(f"❌ **Unsupported file format:** `{uploaded.name}`\n\n{e}")
        st.info("👆 Please upload a **CSV (.csv)** or **Excel (.xlsx / .xls)** file.")
        return

    # ── FEATURE VALIDATION ────────────────────────────────────
    required = REQUIRED_FEATURES.get(model_name, [])
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        st.error(
            f"❌ **Incorrect dataset.** The following required columns are missing:\n\n" +
            "\n".join(f"- `{c}`" for c in missing_cols)
        )
        st.info("👆 Please refer to the **Required Dataset Format** above and re-upload the correct file.")
        return
    # ─────────────────────────────────────────────────────────
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

        # ── METRICS VISUALIZATION ──────────────────────────────────────
        st.subheader("📊 Model Evaluation Metrics")

        if "r2" in metrics:
            # --- REGRESSION (Energy model) ---
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("🏆 Best Model", metrics["best_model"])
            m2.metric("CV R²", f"{metrics['r2']:.4f}",
                      help="5-fold CV R² on training set — used for model selection.")
            m3.metric("Train R²", f"{metrics.get('train_r2', 0):.4f}",
                      help="R² on training data. Compare with Test R² to check for overfitting.")
            m4.metric("Test R²", f"{metrics.get('test_r2', 0):.4f}",
                      help="R² on the held-out 20% test set — the unbiased generalisation estimate.")
            m5.metric("Test MAE", f"{metrics['mae']:.4f} kWh",
                      help="Mean Absolute Error on the held-out test set. Lower is better.")

        else:
            # --- CLASSIFIER (Efficiency / Emission / Maintenance) ---
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🏆 Best Model", metrics["best_model"])
            m2.metric("CV Accuracy", f"{metrics['accuracy']*100:.1f}%",
                      help="5-fold CV accuracy on training set — used for model selection.")
            m3.metric("Train Accuracy", f"{metrics.get('train_accuracy', 0)*100:.1f}%",
                      help="Accuracy on training data. Compare with Test Accuracy to check for overfitting.")
            m4.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0)*100:.1f}%",
                      help="Accuracy on the held-out 20% test set — the unbiased generalisation estimate.")

            # Per-class precision / recall / f1 table
            report = metrics["report"]
            class_rows = {k: v for k, v in report.items()
                          if k not in ("accuracy", "macro avg", "weighted avg")}

            if class_rows:
                st.markdown("**Per-Class Performance**")
                report_df = pd.DataFrame(class_rows).T[["precision", "recall", "f1-score", "support"]]
                report_df.index.name = "Class"
                report_df["precision"] = report_df["precision"].map("{:.2%}".format)
                report_df["recall"]    = report_df["recall"].map("{:.2%}".format)
                report_df["f1-score"]  = report_df["f1-score"].map("{:.2%}".format)
                report_df["support"]   = report_df["support"].astype(int)
                report_df.columns     = ["Precision", "Recall", "F1-Score", "Support"]
                st.dataframe(report_df, use_container_width=True)

            # Macro / Weighted avg summary
            avg_rows = {k: v for k, v in report.items()
                        if k in ("macro avg", "weighted avg")}
            if avg_rows:
                st.markdown("**Aggregate Averages**")
                avg_df = pd.DataFrame(avg_rows).T[["precision", "recall", "f1-score"]]
                avg_df.index.name = "Average"
                avg_df["precision"] = avg_df["precision"].map("{:.2%}".format)
                avg_df["recall"]    = avg_df["recall"].map("{:.2%}".format)
                avg_df["f1-score"]  = avg_df["f1-score"].map("{:.2%}".format)
                avg_df.columns      = ["Precision", "Recall", "F1-Score"]
                st.dataframe(avg_df, use_container_width=True)
        # ──────────────────────────────────────────────────────────────

        # 5. Visuals
        figs = []

        X = engineered.drop(columns=[target])
        y = engineered[target]

        if target != "energy":
            figs.append(confusion_matrix_plot(model, X, y))

        figs.append(feature_importance_plot(model, X, y))

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
    "🛠 Maintenance"
])

with tab1:
    with st.expander("📋 Required Dataset Format", expanded=True):
        st.markdown("""
**Your CSV / Excel file must contain these columns:**

| Column | Type | Description |
|---|---|---|
| `production_load` | float (0–1) | Machine load as a fraction of max capacity |
| `cycle_time` | float (s) | Time to complete one production cycle |
| `machine_temperature` | float (°C) | Operating temperature of the machine |
| `axis_speed` | float (m/s) | Speed of the machine axis |
| `tool_wear` | float (0–1) | Degree of tool wear |
| `ambient_humidity` | float (%) | Ambient humidity level |
| `vibration_level` | float | Vibration sensor reading |
| `power_factor` | float (0–1) | Electrical power factor |
| `energy` ⭐ | float (kWh) | **Target column** — energy consumption to predict |
        """)
    handle_tab(
        upload_key="energy_data",
        model_title="Energy Prediction Model",
        train_fn=lambda df, t: train_energy(df, t),
        model_name="energy"
    )

with tab2:
    with st.expander("📋 Required Dataset Format", expanded=True):
        st.markdown("""
**Your CSV / Excel file must contain these columns:**

| Column | Type | Description |
|---|---|---|
| `production_load` | float (0–1) | Machine load as a fraction of max capacity |
| `cycle_time` | float (s) | Time to complete one production cycle |
| `machine_temperature` | float (°C) | Operating temperature of the machine |
| `axis_speed` | float (m/s) | Speed of the machine axis |
| `vibration_signal` | float | Vibration sensor signal reading |
| `power_factor` | float (0–1) | Electrical power factor |
| `efficiency_class` ⭐ | string | **Target column** — one of `Low`, `Medium`, `High` |
        """)
    handle_tab(
        upload_key="efficiency_data",
        model_title="Efficiency Classifier",
        train_fn=lambda df, t: train_classifier(df, t, "efficiency"),
        model_name="efficiency"
    )

with tab3:
    with st.expander("📋 Required Dataset Format", expanded=True):
        st.markdown("""
**Your CSV / Excel file must contain these columns:**

| Column | Type | Description |
|---|---|---|
| `production_load` | float (0–1) | Machine load as a fraction of max capacity |
| `cycle_time` | float (s) | Time to complete one production cycle |
| `machine_temperature` | float (°C) | Operating temperature of the machine |
| `axis_speed` | float (m/s) | Speed of the machine axis |
| `power_factor` | float (0–1) | Electrical power factor |
| `emission_class` ⭐ | string | **Target column** — one of `Low`, `Medium`, `High` |
        """)
    handle_tab(
        upload_key="emission_data",
        model_title="Emission Classifier",
        train_fn=lambda df, t: train_classifier(df, t, "emission"),
        model_name="emission"
    )

with tab4:
    with st.expander("📋 Required Dataset Format", expanded=True):
        st.markdown("""
**Your CSV / Excel file must contain these columns:**

| Column | Type | Description |
|---|---|---|
| `production_load` | float (0–1) | Machine load as a fraction of max capacity |
| `machine_temperature` | float (°C) | Operating temperature of the machine |
| `axis_speed` | float (m/s) | Speed of the machine axis |
| `vibration_level` | float | Vibration sensor reading |
| `tool_wear` | float (0–1) | Degree of tool wear |
| `oil_quality` | float (0–1) | Oil quality / degradation score |
| `pressure` | float | System pressure reading |
| `maintenance_class` ⭐ | string | **Target column** — one of `Low`, `Medium`, `High` |
        """)
    handle_tab(
        upload_key="maintenance_data",
        model_title="Maintenance Model",
        train_fn=lambda df, t: train_classifier(df, t, "maintenance"),
        model_name="maintenance"
    )