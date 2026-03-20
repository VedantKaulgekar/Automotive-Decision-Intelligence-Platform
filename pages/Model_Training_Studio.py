import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from io import BytesIO
import concurrent.futures

# ML
from sklearn.metrics import (
    r2_score, mean_absolute_error, accuracy_score, classification_report,
    confusion_matrix
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier

# XGBoost / LightGBM — optional, graceful fallback if not installed
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# Viz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Modules
from modules.preprocessing import (
    load_dataset, preprocess, build_preprocessing_pipeline,
    add_missing_indicators, detect_and_winsorise_outliers, encode_target
)
from modules.feature_engineering import add_automotive_features
from modules.storage import ensure_dirs, PATHS

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Model Training Studio", layout="wide")
ensure_dirs()
pdfmetrics.registerFont(TTFont("DejaVu", "assets/fonts/DejaVuSans.ttf"))
st.title("🛠 Automotive Sustainability – Model Training Studio")


# ─────────────────────────────────────────────────────────────────────────────
# TARGET DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
def find_target(df):
    for t in ["energy", "efficiency_class", "emission_class", "maintenance_class"]:
        if t in df.columns:
            return t
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE POOLS
# ─────────────────────────────────────────────────────────────────────────────
def get_regression_candidates():
    candidates = {
        "RandomForest": (
            RandomForestRegressor(random_state=42),
            {"n_estimators": [100, 200, 300], "max_depth": [6, 8, 10, None],
             "min_samples_leaf": [5, 10, 20], "max_features": [0.5, 0.6, 0.8]}
        ),
        "ExtraTrees": (
            ExtraTreesRegressor(random_state=42),
            {"n_estimators": [100, 200, 300], "max_depth": [6, 8, 10, None],
             "min_samples_leaf": [5, 10, 20], "max_features": [0.5, 0.6, 0.8]}
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [3, 4, 5],
             "learning_rate": [0.03, 0.05, 0.1], "subsample": [0.7, 0.8, 1.0],
             "min_samples_leaf": [5, 10]}
        ),
        "Ridge": (
            Ridge(),
            {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
        ),
    }
    if HAS_XGB:
        candidates["XGBoost"] = (
            XGBRegressor(random_state=42, verbosity=0, eval_metric="rmse"),
            {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7],
             "learning_rate": [0.03, 0.05, 0.1], "subsample": [0.7, 0.8, 1.0],
             "colsample_bytree": [0.6, 0.8, 1.0], "reg_alpha": [0, 0.1, 1.0]}
        )
    if HAS_LGBM:
        candidates["LightGBM"] = (
            LGBMRegressor(random_state=42, verbosity=-1),
            {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7],
             "learning_rate": [0.03, 0.05, 0.1], "num_leaves": [15, 31, 63],
             "subsample": [0.7, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0],
             "bagging_freq": [1]}
        )
    return candidates


def get_classifier_candidates():
    candidates = {
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [100, 200, 300], "max_depth": [6, 8, 10, None],
             "min_samples_leaf": [10, 15, 25], "max_features": [0.5, 0.6, 0.8]}
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(random_state=42),
            {"n_estimators": [100, 200, 300], "max_depth": [6, 8, 10, None],
             "min_samples_leaf": [10, 15, 25], "max_features": [0.5, 0.6, 0.8]}
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [3, 4, 5],
             "learning_rate": [0.03, 0.05, 0.1], "subsample": [0.7, 0.8, 1.0],
             "min_samples_leaf": [5, 10]}
        ),
        "LogisticRegression": (
            LogisticRegression(max_iter=500, random_state=42),
            {"C": [0.1, 0.3, 0.5, 1.0, 2.0]}
        ),
        "MLP": (
            MLPClassifier(max_iter=500, random_state=42),
            {"hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
             "alpha": [0.001, 0.01, 0.05],
             "learning_rate_init": [0.001, 0.005]}
        ),
    }
    if HAS_XGB:
        candidates["XGBoost"] = (
            XGBClassifier(random_state=42, verbosity=0, eval_metric="mlogloss"),
            {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7],
             "learning_rate": [0.03, 0.05, 0.1], "subsample": [0.7, 0.8, 1.0],
             "colsample_bytree": [0.6, 0.8, 1.0], "reg_alpha": [0, 0.1, 1.0]}
        )
    if HAS_LGBM:
        candidates["LightGBM"] = (
            LGBMClassifier(random_state=42, verbosity=-1),
            {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7],
             "learning_rate": [0.03, 0.05, 0.1], "num_leaves": [15, 31, 63],
             "subsample": [0.7, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0],
             "bagging_freq": [1]}
        )
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ONE CANDIDATE  (runs in a thread)
# ─────────────────────────────────────────────────────────────────────────────
def _train_one(name, model, param_dist, X_train, y_train, scoring, n_iter=10):
    """
    Train a single candidate with RandomizedSearchCV.
    Returns (name, cv_score, fitted_model, error_msg).
    On failure: cv_score=-inf, fitted_model=None, error_msg=str(e).

    XGBoost and LightGBM require integer-encoded labels for multiclass.
    We encode here and decode after — transparent to the rest of the pipeline.
    """
    from sklearn.model_selection import cross_val_score, RandomizedSearchCV
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    # Encode string labels for XGB / LGBM (they need integer targets)
    needs_encoding = name in ("XGBoost", "LightGBM") and y_train.dtype == object
    le = None
    if needs_encoding:
        le = LabelEncoder()
        y_fit = le.fit_transform(y_train)
    else:
        y_fit = y_train

    try:
        # Wrap model in Pipeline: median impute → variance filter → RobustScale → model
        # This ensures the scaler is ALWAYS fit only on the training fold
        # inside each CV split — never on validation or test data.
        from modules.preprocessing import build_preprocessing_pipeline
        pipeline = build_preprocessing_pipeline(model)

        # Prefix hyperparameter keys with "model__" for Pipeline compatibility
        pipeline_params = {f"model__{k}": v for k, v in param_dist.items()}

        if pipeline_params:
            search = RandomizedSearchCV(
                pipeline, pipeline_params,
                n_iter=n_iter, cv=5, scoring=scoring,
                random_state=42, n_jobs=-1
            )
            search.fit(X_train, y_fit)
            fitted = search.best_estimator_
            if le is not None:
                fitted = _LabelDecodingWrapper(fitted, le)
            return name, float(search.best_score_), fitted, None
        else:
            scores = cross_val_score(pipeline, X_train, y_fit, cv=5, scoring=scoring)
            pipeline.fit(X_train, y_fit)
            fitted = _LabelDecodingWrapper(pipeline, le) if le is not None else pipeline
            return name, float(scores.mean()), fitted, None
    except Exception as e:
        return name, float("-inf"), None, str(e)


class _LabelDecodingWrapper:
    """
    Thin wrapper around XGB/LGBM that decodes integer predictions
    back to original string class labels, making it compatible with
    the rest of the pipeline (confusion_matrix, classification_report etc.).
    """
    def __init__(self, model, le):
        self._model = model
        self._le    = le

    def predict(self, X):
        import numpy as np
        raw = self._model.predict(X)
        return self._le.inverse_transform(raw.astype(int))

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    # Forward all other attribute access to the wrapped model
    def __getattr__(self, item):
        return getattr(self._model, item)


# ─────────────────────────────────────────────────────────────────────────────
# LIVE TRAINING UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _render_data_profile(df, target, placeholder):
    """Show data shape, class distribution / target stats, and feature correlations."""
    with placeholder.container():
        st.markdown("### 📊 Dataset Profile")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Features", len(df.columns) - 1)
        c3.metric("Target", target)

        col_left, col_right = st.columns(2)

        # Left: target distribution
        with col_left:
            if df[target].dtype == object or df[target].nunique() <= 5:
                counts = df[target].value_counts().sort_index()
                fig = px.bar(
                    x=counts.index.astype(str), y=counts.values,
                    color=counts.index.astype(str),
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    labels={"x": target, "y": "Count"},
                    title=f"Target Distribution — {target}"
                )
                fig.update_layout(showlegend=False, height=300, margin=dict(t=40, b=30))
                st.plotly_chart(fig, width="stretch")
            else:
                fig = px.histogram(
                    df, x=target, nbins=30,
                    color_discrete_sequence=["#4e8cff"],
                    title=f"Target Distribution — {target}",
                    labels={target: target, "count": "Frequency"}
                )
                fig.update_layout(height=300, margin=dict(t=40, b=30))
                st.plotly_chart(fig, width="stretch")

        # Right: correlation with target
        with col_right:
            numeric = df.select_dtypes(include=[np.number])
            if target in numeric.columns and len(numeric.columns) > 1:
                corr_with_target = (
                    numeric.corr(numeric_only=True)[target]
                    .drop(target)
                    .sort_values(key=abs, ascending=True)
                )
                fig2 = go.Figure(go.Bar(
                    x=corr_with_target.values,
                    y=corr_with_target.index,
                    orientation="h",
                    marker_color=["#e05c5c" if v < 0 else "#4e8cff"
                                  for v in corr_with_target.values],
                    hovertemplate="%{y}<br>r = %{x:.3f}<extra></extra>"
                ))
                fig2.update_layout(
                    title=f"Feature Correlation with {target}",
                    xaxis_title="Pearson r",
                    height=300, margin=dict(t=40, b=30, l=130)
                )
                st.plotly_chart(fig2, width="stretch")


def _render_live_scoreboard(results_so_far, scoring_label, placeholder, failed_names=None):
    """Render a live bar chart of CV scores as candidates finish.
    Failed candidates shown as grey bars with ❌ label."""
    failed_names = failed_names or []
    with placeholder.container():
        if not results_so_far and not failed_names:
            st.info("⏳ Waiting for first candidate to finish…")
            return

        names  = [r[0] for r in results_so_far]
        scores = [max(r[1], 0) for r in results_so_far]
        best   = max(scores) if scores else 1
        bar_colors = ["#27ae60" if s == best else "#4e8cff" for s in scores]
        labels = [f"{s:.4f}" for s in scores]

        # Append failed candidates as zero-score grey bars
        for fn in failed_names:
            names.append(f"❌ {fn}")
            scores.append(0)
            bar_colors.append("#888888")
            labels.append("failed")

        fig = go.Figure(go.Bar(
            x=scores, y=names, orientation="h",
            marker_color=bar_colors,
            text=labels,
            textposition="outside",
            hovertemplate="%{y}<br>CV Score: %{x:.4f}<extra></extra>"
        ))
        fig.update_layout(
            title=f"🏆 Live CV {scoring_label} Scoreboard",
            xaxis_title=scoring_label,
            height=max(200, 60 * len(names)),
            margin=dict(t=50, b=30, l=150, r=80),
            xaxis=dict(range=[0, best * 1.2 if best else 1])
        )
        st.plotly_chart(fig, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# AUTO ML — ENERGY REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
def train_energy(df, target, profile_ph, progress_ph, scoreboard_ph, status_ph):
    from sklearn.model_selection import train_test_split

    st.session_state["energy_raw_data"] = df.copy()
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Show data profile immediately
    _render_data_profile(df, target, profile_ph)

    candidates = get_regression_candidates()
    n_cands    = len(candidates)
    results    = []   # (name, cv_score, model)

    # ── Parallel training with live updates ──────────────────────────────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, n_cands)) as ex:
        futures = {
            ex.submit(_train_one, name, model, param_dist,
                      X_train, y_train, "r2", 12): name
            for name, (model, param_dist) in candidates.items()
        }

        done_count = 0
        failed = []
        for fut in concurrent.futures.as_completed(futures):
            name, score, model, err = fut.result()
            done_count += 1

            if model is not None:
                results.append((name, score, model))
                progress_ph.progress(
                    done_count / n_cands,
                    text=f"✅ {name} done  ({done_count}/{n_cands})"
                )
            else:
                failed.append((name, err))
                progress_ph.progress(
                    done_count / n_cands,
                    text=f"❌ {name} failed  ({done_count}/{n_cands}): {err[:60] if err else 'unknown error'}"
                )

            # Update live scoreboard (pass failed list for display)
            _render_live_scoreboard(
                [(r[0], r[1]) for r in results], "R²", scoreboard_ph,
                failed_names=[f[0] for f in failed]
            )

    if not results:
        status_ph.error("All candidates failed to train.")
        return None, None

    # Best by CV R²
    results.sort(key=lambda x: x[1], reverse=True)
    best_name, best_cv, best_model = results[0]

    train_r2 = float(r2_score(y_train, best_model.predict(X_train)))
    test_r2  = float(r2_score(y_test,  best_model.predict(X_test)))
    test_mae = float(mean_absolute_error(y_test, best_model.predict(X_test)))

    metrics = {
        "best_model": best_name,
        "r2":         best_cv,
        "train_r2":   train_r2,
        "test_r2":    test_r2,
        "mae":        test_mae,
        "all_results": [(r[0], r[1]) for r in results],
    }

    st.session_state["energy_model"]    = best_model
    st.session_state["energy_metadata"] = {
        "feature_order": list(X_train.columns),  # pre-pipeline column order
    }

    status_ph.success(
        f"🏆 **{best_name}** selected  |  CV R²: {best_cv:.4f}  |  "
        f"Train R²: {train_r2:.4f}  |  Test R²: {test_r2:.4f}  |  "
        f"Test MAE: {test_mae:.4f} kWh"
    )
    return best_model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# AUTO ML — CLASSIFIERS
# ─────────────────────────────────────────────────────────────────────────────
def train_classifier(df, target, name, profile_ph, progress_ph, scoreboard_ph, status_ph):
    from sklearn.model_selection import train_test_split

    st.session_state[f"{name}_raw_data"] = df.copy()
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    _render_data_profile(df, target, profile_ph)

    candidates = get_classifier_candidates()
    n_cands    = len(candidates)
    results    = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, n_cands)) as ex:
        futures = {
            ex.submit(_train_one, cname, model, param_dist,
                      X_train, y_train, "accuracy", 10): cname
            for cname, (model, param_dist) in candidates.items()
        }

        done_count = 0
        failed = []
        for fut in concurrent.futures.as_completed(futures):
            cname, score, model, err = fut.result()
            done_count += 1

            if model is not None:
                results.append((cname, score, model))
                progress_ph.progress(
                    done_count / n_cands,
                    text=f"✅ {cname} done  ({done_count}/{n_cands})"
                )
            else:
                failed.append((cname, err))
                progress_ph.progress(
                    done_count / n_cands,
                    text=f"❌ {cname} failed  ({done_count}/{n_cands}): {err[:60] if err else 'unknown error'}"
                )

            _render_live_scoreboard(
                [(r[0], r[1]) for r in results], "Accuracy", scoreboard_ph,
                failed_names=[f[0] for f in failed]
            )

    if not results:
        status_ph.error("All candidates failed to train.")
        return None, None

    results.sort(key=lambda x: x[1], reverse=True)
    best_cname, best_acc, best_model = results[0]

    train_acc = float(accuracy_score(y_train, best_model.predict(X_train)))
    test_acc  = float(accuracy_score(y_test,  best_model.predict(X_test)))

    metrics = {
        "best_model":     best_cname,
        "accuracy":       best_acc,
        "train_accuracy": train_acc,
        "test_accuracy":  test_acc,
        "report":         classification_report(
            y_test, best_model.predict(X_test), output_dict=True
        ),
        "all_results": [(r[0], r[1]) for r in results],
    }

    st.session_state[f"{name}_model"]    = best_model
    st.session_state[f"{name}_metadata"] = {
        "feature_order": list(X_train.columns),  # pre-pipeline column order
    }

    status_ph.success(
        f"🏆 **{best_cname}** selected  |  CV Acc: {best_acc:.1%}  |  "
        f"Train Acc: {train_acc:.1%}  |  Test Acc: {test_acc:.1%}"
    )
    return best_model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def feature_importance_plot(model, X, y=None):
    from sklearn.inspection import permutation_importance
    fig, ax = plt.subplots(figsize=(6, 4))

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx  = np.argsort(importances)
        sns.barplot(x=importances[sorted_idx],
                    y=np.array(X.columns)[sorted_idx], ax=ax)
        ax.set_title("Feature Importance (Gini / Split)")
        ax.set_xlabel("Importance Score")

    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = (np.mean(np.abs(coef), axis=0)
                       if coef.ndim > 1 else np.abs(coef.flatten()))
        sorted_idx = np.argsort(importances)
        sns.barplot(x=importances[sorted_idx],
                    y=np.array(X.columns)[sorted_idx], ax=ax)
        ax.set_title("Feature Coefficients (|coef|)")
        ax.set_xlabel("|Coefficient|")

    elif y is not None:
        try:
            result = permutation_importance(model, X, y, n_repeats=10,
                                            random_state=42, n_jobs=-1)
            importances = result.importances_mean
            sorted_idx  = np.argsort(importances)
            sns.barplot(x=importances[sorted_idx],
                        y=np.array(X.columns)[sorted_idx], ax=ax)
            ax.set_title("Feature Importance (Permutation)")
            ax.set_xlabel("Mean Score Drop")
        except Exception as e:
            ax.text(0.5, 0.5, f"Could not compute:\n{e}",
                    ha="center", va="center", fontsize=9)
            ax.axis("off")
    else:
        ax.text(0.5, 0.5, "Not available for this model type",
                ha="center", va="center", fontsize=11)
        ax.axis("off")

    ax.set_ylabel("Feature")
    plt.tight_layout()
    return fig


def confusion_matrix_plot(model, X, y):
    fig, ax = plt.subplots(figsize=(5, 3))
    cm = confusion_matrix(y, model.predict(X))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    return fig


def all_candidates_chart(all_results, scoring_label):
    """Final sorted bar chart of all candidates for the PDF."""
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.5 * len(all_results))))
    names  = [r[0] for r in all_results]
    scores = [max(r[1], 0) for r in all_results]
    colors = ["#27ae60"] + ["#4e8cff"] * (len(names) - 1)
    ax.barh(names[::-1], scores[::-1], color=colors[::-1])
    ax.set_xlabel(scoring_label)
    ax.set_title(f"All Candidates — CV {scoring_label}")
    for i, s in enumerate(scores[::-1]):
        ax.text(s + 0.002, i, f"{s:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PDF GENERATOR  (unchanged structure, just pass all_results fig)
# ─────────────────────────────────────────────────────────────────────────────
def export_pdf(title, metrics, figs):
    import datetime
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import HRFlowable, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER

    buf       = BytesIO()
    base_font = "DejaVu"

    cover_title_style = ParagraphStyle("CoverTitle", fontName=base_font,
        fontSize=22, leading=28, alignment=TA_CENTER, spaceAfter=6)
    cover_sub_style   = ParagraphStyle("CoverSub",   fontName=base_font,
        fontSize=11, leading=14, alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"), spaceAfter=4)
    section_style     = ParagraphStyle("Section",    fontName=base_font,
        fontSize=13, leading=16, textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=6, spaceBefore=14)
    body_style        = ParagraphStyle("Body",       fontName=base_font,
        fontSize=10, leading=14, spaceAfter=3)
    caption_style     = ParagraphStyle("Caption",    fontName=base_font,
        fontSize=9,  leading=12, textColor=colors.HexColor("#444444"),
        alignment=TA_CENTER, spaceAfter=10, spaceBefore=4)
    kv_key_style      = ParagraphStyle("KVKey",      fontName=base_font,
        fontSize=10, leading=13, textColor=colors.HexColor("#333333"))
    kv_val_style      = ParagraphStyle("KVVal",      fontName=base_font,
        fontSize=10, leading=13, textColor=colors.HexColor("#1a1a2e"))
    footer_style      = ParagraphStyle("Footer",     fontName=base_font,
        fontSize=8,  textColor=colors.HexColor("#888888"), alignment=TA_CENTER)

    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=18*mm, bottomMargin=18*mm)
    story = []

    story.append(Spacer(1, 8*mm))
    story.append(Paragraph("Automotive Decision Intelligence Platform", cover_sub_style))
    story.append(Paragraph(title, cover_title_style))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}",
        cover_sub_style))
    story.append(Spacer(1, 4*mm))
    story.append(HRFlowable(width="100%", thickness=1.5,
                             color=colors.HexColor("#4e8cff"), spaceAfter=6*mm))

    intros = {
        "Energy Prediction Model":
            "AutoML evaluated RandomForest, ExtraTrees, GradientBoosting, Ridge"
            + (", XGBoost" if HAS_XGB else "")
            + (", LightGBM" if HAS_LGBM else "")
            + " for energy regression. The best model was selected by 5-fold CV R².",
        "Efficiency Classifier":
            "AutoML evaluated RandomForest, ExtraTrees, GradientBoosting, "
            "LogisticRegression, MLP"
            + (", XGBoost" if HAS_XGB else "")
            + (", LightGBM" if HAS_LGBM else "")
            + " for efficiency classification. Best selected by CV accuracy.",
        "Emission Classifier":
            "AutoML evaluated the full classifier pool for emission class prediction.",
        "Maintenance Model":
            "AutoML evaluated the full classifier pool for maintenance risk prediction.",
    }
    story.append(Paragraph(
        intros.get(title, "AutoML training report."), body_style))
    story.append(Spacer(1, 6*mm))

    # Section 1 — Metrics
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#cccccc"), spaceAfter=4))
    story.append(Paragraph("1. Model Performance Metrics", section_style))

    is_regression = "r2" in metrics

    if is_regression:
        metric_data = [
            ["Metric", "Value", "Interpretation"],
            [Paragraph("Best Algorithm", kv_key_style),
             Paragraph(str(metrics.get("best_model", "—")), kv_val_style),
             Paragraph("Highest CV R² on training set", kv_key_style)],
            [Paragraph("CV R² (Train)", kv_key_style),
             Paragraph(f"{metrics.get('r2', 0):.4f}", kv_val_style),
             Paragraph("5-fold CV R² — model selection criterion", kv_key_style)],
            [Paragraph("Train R²", kv_key_style),
             Paragraph(f"{metrics.get('train_r2', 0):.4f}", kv_val_style),
             Paragraph("Compare with Test R² for overfitting check", kv_key_style)],
            [Paragraph("Test R²", kv_key_style),
             Paragraph(f"{metrics.get('test_r2', 0):.4f}", kv_val_style),
             Paragraph("Unbiased estimate on 20% held-out test set", kv_key_style)],
            [Paragraph("Test MAE", kv_key_style),
             Paragraph(f"{metrics.get('mae', 0):.4f} kWh", kv_val_style),
             Paragraph("Mean Absolute Error in kWh on test set", kv_key_style)],
        ]
    else:
        report     = metrics.get("report", {})
        class_rows = {k: v for k, v in report.items()
                      if k not in ("accuracy", "macro avg", "weighted avg")}
        metric_data = [["Metric", "Value", "Interpretation"]]
        for label, key, interp in [
            ("Best Algorithm",    "best_model",     "Highest CV accuracy"),
            ("CV Accuracy",       "accuracy",       "5-fold CV — model selection criterion"),
            ("Train Accuracy",    "train_accuracy", "Compare with Test for overfitting check"),
            ("Test Accuracy",     "test_accuracy",  "Unbiased estimate on 20% held-out test set"),
        ]:
            val = metrics.get(key, 0)
            fmt = f"{val*100:.1f}%" if isinstance(val, float) else str(val)
            metric_data.append([
                Paragraph(label, kv_key_style),
                Paragraph(fmt,   kv_val_style),
                Paragraph(interp, kv_key_style),
            ])

    t = Table(metric_data, colWidths=[65*mm, 40*mm, 65*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0), colors.HexColor("#4e8cff")),
        ("TEXTCOLOR",      (0,0), (-1,0), colors.white),
        ("FONTNAME",       (0,0), (-1,0), base_font),
        ("FONTSIZE",       (0,0), (-1,0), 10),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [colors.HexColor("#f5f7ff"), colors.white]),
        ("GRID",           (0,0), (-1,-1), 0.4, colors.HexColor("#dddddd")),
        ("LEFTPADDING",    (0,0), (-1,-1), 6),
        ("RIGHTPADDING",   (0,0), (-1,-1), 6),
        ("TOPPADDING",     (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 4*mm))

    if not is_regression and class_rows:
        story.append(Paragraph("Per-Class Breakdown", section_style))
        class_data = [["Class", "Precision", "Recall", "F1", "Support"]]
        for cls, vals in sorted(class_rows.items()):
            class_data.append([
                Paragraph(cls, kv_key_style),
                Paragraph(f"{vals['precision']:.2%}", kv_val_style),
                Paragraph(f"{vals['recall']:.2%}", kv_val_style),
                Paragraph(f"{vals['f1-score']:.2%}", kv_val_style),
                Paragraph(str(int(vals['support'])), kv_val_style),
            ])
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
            ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#27ae60")),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTNAME",    (0,0), (-1,0), base_font),
            ("FONTSIZE",    (0,0), (-1,0), 10),
            ("ROWBACKGROUNDS",(0,1),(-1,-3),
             [colors.HexColor("#f0fff4"), colors.white]),
            ("BACKGROUND",  (0,-2),(-1,-1), colors.HexColor("#e8e8e8")),
            ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#dddddd")),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("RIGHTPADDING",(0,0), (-1,-1), 6),
            ("TOPPADDING",  (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ]))
        story.append(t2)
        story.append(Spacer(1, 6*mm))

    # Section 2 — Visual Analytics
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#cccccc"), spaceAfter=4))
    story.append(Paragraph("2. Visual Analytics", section_style))
    story.append(Paragraph(
        "Charts produced during training for model interpretation.", body_style))
    story.append(Spacer(1, 3*mm))

    chart_info = {
        "Energy Prediction Model": [
            ("All Candidates — CV R²",
             "CV R² scores for all evaluated algorithms. The green bar is the selected model."),
            ("Feature Importance",
             "Relative importance of each input feature for predicting energy consumption."),
        ],
        "Efficiency Classifier": [
            ("All Candidates — CV Accuracy",
             "CV accuracy for all evaluated algorithms. Green bar = selected model."),
            ("Confusion Matrix",
             "Predicted vs actual efficiency class. Diagonal = correct predictions."),
            ("Feature Importance",
             "Strongest discriminators between Low / Medium / High efficiency."),
        ],
        "Emission Classifier": [
            ("All Candidates — CV Accuracy", "CV accuracy for all evaluated algorithms."),
            ("Confusion Matrix", "Predicted vs actual emission class counts."),
            ("Feature Importance", "Primary drivers of emission class prediction."),
        ],
        "Maintenance Model": [
            ("All Candidates — CV Accuracy", "CV accuracy for all evaluated algorithms."),
            ("Confusion Matrix",
             "High-risk samples misclassified as Low-risk are the most critical errors."),
            ("Feature Importance",
             "Early-warning indicators for maintenance risk."),
        ],
    }

    captions = chart_info.get(title, [])
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

    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#cccccc"), spaceAfter=4))
    story.append(Paragraph(
        "Auto-generated by the Automotive Decision Intelligence Platform. "
        "All metrics computed on held-out test data.", footer_style))

    doc.build(story)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# REQUIRED FEATURES
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# TAB HANDLER
# ─────────────────────────────────────────────────────────────────────────────
def handle_tab(upload_key, model_title, train_fn, model_name):
    uploaded = st.file_uploader(
        f"Upload dataset for **{model_title}**",
        type=["csv", "xlsx"], key=upload_key
    )
    if not uploaded:
        return

    try:
        df = load_dataset(uploaded)
    except ValueError as e:
        st.error(f"❌ **Unsupported file format:** `{uploaded.name}`\n\n{e}")
        st.info("👆 Please upload a **CSV (.csv)** or **Excel (.xlsx / .xls)** file.")
        return

    required     = REQUIRED_FEATURES.get(model_name, [])
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        st.error(
            "❌ **Incorrect dataset.** Missing columns:\n\n" +
            "\n".join(f"- `{c}`" for c in missing_cols)
        )
        st.info("👆 Please refer to the Required Dataset Format above.")
        return

    st.write("### Raw Data Preview", df.head())

    target = find_target(df)
    if not target:
        st.error("❌ Could not detect target column automatically.")
        return

    # ── Preprocessing steps (shown to user before training) ───────────────
    # Step 1: Missing value indicators
    df_indicators = add_missing_indicators(df)
    missing_added = [c for c in df_indicators.columns if c.endswith("_was_missing")]
    if missing_added:
        st.info(f"ℹ️ Added **{len(missing_added)}** missing-value indicator column(s): "
                + ", ".join(f"`{c}`" for c in missing_added))

    # Step 2: Winsorisation — cap outlier values at IQR fences, no rows removed
    df_clean, outlier_report = detect_and_winsorise_outliers(df_indicators, target_col=target)
    if outlier_report:
        with st.expander(
            f"⚠️ Winsorised outliers in {len(outlier_report)} column(s) "
            f"— values capped to IQR fences, all rows kept",
            expanded=False
        ):
            for col, info in outlier_report.items():
                st.markdown(
                    f"- **`{col}`** — {info['n_capped']} value(s) capped "
                    f"to [{info['lower_fence']}, {info['upper_fence']}]"
                )
    else:
        st.success(f"✅ No outliers — all {len(df_clean):,} rows and values intact.")

    # Step 3: Feature engineering
    engineered = add_automotive_features(df_clean)
    new_features = [c for c in engineered.columns
                    if c not in df_clean.columns]
    st.info(f"🔧 Feature engineering added **{len(new_features)}** derived column(s): "
            + ", ".join(f"`{f}`" for f in new_features))
    st.write("### Engineered Features Preview", engineered.head())

    # Show candidate pool
    is_regression = (target == "energy")
    cands = get_regression_candidates() if is_regression else get_classifier_candidates()
    st.info(
        f"🤖 **{len(cands)} candidates** will be evaluated in parallel: "
        + ", ".join(f"`{k}`" for k in cands.keys())
    )

    if not st.button(f"🚀 Train {model_title}"):
        return

    st.markdown("---")
    st.subheader("⚡ Training in Progress")

    # Placeholders — all visible before training starts
    profile_ph    = st.empty()
    progress_ph   = st.progress(0, text="Starting…")
    st.markdown("#### 🏆 Live Candidate Scoreboard")
    scoreboard_ph = st.empty()
    status_ph     = st.empty()

    with st.spinner("Training all candidates in parallel…"):
        model, metrics = train_fn(
            engineered, target,
            profile_ph, progress_ph, scoreboard_ph, status_ph
        )

    if model is None or metrics is None:
        return

    progress_ph.progress(1.0, text="✅ All candidates evaluated")

    st.markdown("---")
    st.subheader("📊 Final Model Evaluation")

    if "r2" in metrics:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("🏆 Best Model", metrics["best_model"])
        m2.metric("CV R²",    f"{metrics['r2']:.4f}")
        m3.metric("Train R²", f"{metrics.get('train_r2', 0):.4f}")
        m4.metric("Test R²",  f"{metrics.get('test_r2', 0):.4f}")
        m5.metric("Test MAE", f"{metrics['mae']:.4f} kWh")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🏆 Best Model",   metrics["best_model"])
        m2.metric("CV Accuracy",     f"{metrics['accuracy']*100:.1f}%")
        m3.metric("Train Accuracy",  f"{metrics.get('train_accuracy', 0)*100:.1f}%")
        m4.metric("Test Accuracy",   f"{metrics.get('test_accuracy', 0)*100:.1f}%")

        report     = metrics["report"]
        class_rows = {k: v for k, v in report.items()
                      if k not in ("accuracy", "macro avg", "weighted avg")}
        if class_rows:
            st.markdown("**Per-Class Performance**")
            rdf = pd.DataFrame(class_rows).T[["precision","recall","f1-score","support"]]
            rdf.index.name = "Class"
            for col in ["precision","recall","f1-score"]:
                rdf[col] = rdf[col].map("{:.2%}".format)
            rdf["support"] = rdf["support"].astype(int)
            rdf.columns    = ["Precision","Recall","F1-Score","Support"]
            st.dataframe(rdf, use_container_width=True)

        avg_rows = {k: v for k, v in report.items()
                    if k in ("macro avg", "weighted avg")}
        if avg_rows:
            st.markdown("**Aggregate Averages**")
            adf = pd.DataFrame(avg_rows).T[["precision","recall","f1-score"]]
            adf.index.name = "Average"
            for col in ["precision","recall","f1-score"]:
                adf[col] = adf[col].map("{:.2%}".format)
            adf.columns = ["Precision","Recall","F1-Score"]
            st.dataframe(adf, use_container_width=True)

    # Visuals for PDF
    figs = []
    scoring_label = "R²" if is_regression else "Accuracy"
    figs.append(all_candidates_chart(metrics["all_results"], scoring_label))

    X = engineered.drop(columns=[target])
    y = engineered[target]

    if target != "energy":
        figs.append(confusion_matrix_plot(model, X, y))

    figs.append(feature_importance_plot(model, X, y))

    # Show the static charts in UI too
    vis_cols = st.columns(min(len(figs), 3))
    for i, fig in enumerate(figs):
        with vis_cols[i % len(vis_cols)]:
            st.pyplot(fig)

    pdf = export_pdf(model_title, metrics, figs)
    st.download_button(
        f"📄 Download {model_title} Report",
        data=pdf,
        file_name=f"{model_name}_report.pdf",
        mime="application/pdf"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Energy Model",
    "📈 Efficiency Classifier",
    "🌍 Emission Classifier",
    "🛠 Maintenance"
])

with tab1:
    with st.expander("📋 Required Dataset Format", expanded=True):
        st.markdown("""
| Column | Type | Description |
|---|---|---|
| `production_load` | float (0–1) | Machine load as fraction of max capacity |
| `cycle_time` | float (s) | Time per production cycle |
| `machine_temperature` | float (°C) | Operating temperature |
| `axis_speed` | float (m/s) | Axis speed |
| `tool_wear` | float (0–1) | Tool wear |
| `ambient_humidity` | float (%) | Ambient humidity |
| `vibration_level` | float | Vibration reading |
| `power_factor` | float (0–1) | Electrical power factor |
| `energy` ⭐ | float (kWh) | **Target** |
""")
    handle_tab("energy_data",     "Energy Prediction Model",
               lambda df, t, p, pr, s, st_: train_energy(df, t, p, pr, s, st_),
               "energy")

with tab2:
    with st.expander("📋 Required Dataset Format", expanded=True):
        st.markdown("""
| Column | Type | Description |
|---|---|---|
| `production_load` | float (0–1) | Machine load |
| `cycle_time` | float (s) | Cycle time |
| `machine_temperature` | float (°C) | Temperature |
| `axis_speed` | float (m/s) | Axis speed |
| `vibration_signal` | float | Vibration signal |
| `power_factor` | float (0–1) | Power factor |
| `efficiency_class` ⭐ | string | **Target** — `Low`/`Medium`/`High` |
""")
    handle_tab("efficiency_data", "Efficiency Classifier",
               lambda df, t, p, pr, s, st_: train_classifier(df, t, "efficiency", p, pr, s, st_),
               "efficiency")

with tab3:
    with st.expander("📋 Required Dataset Format", expanded=True):
        st.markdown("""
| Column | Type | Description |
|---|---|---|
| `production_load` | float (0–1) | Machine load |
| `cycle_time` | float (s) | Cycle time |
| `machine_temperature` | float (°C) | Temperature |
| `axis_speed` | float (m/s) | Axis speed |
| `power_factor` | float (0–1) | Power factor |
| `emission_class` ⭐ | string | **Target** — `Low`/`Medium`/`High` |
""")
    handle_tab("emission_data",   "Emission Classifier",
               lambda df, t, p, pr, s, st_: train_classifier(df, t, "emission", p, pr, s, st_),
               "emission")

with tab4:
    with st.expander("📋 Required Dataset Format", expanded=True):
        st.markdown("""
| Column | Type | Description |
|---|---|---|
| `production_load` | float (0–1) | Machine load |
| `machine_temperature` | float (°C) | Temperature |
| `axis_speed` | float (m/s) | Axis speed |
| `vibration_level` | float | Vibration |
| `tool_wear` | float (0–1) | Tool wear |
| `oil_quality` | float (0–1) | Oil quality |
| `pressure` | float | Pressure |
| `maintenance_class` ⭐ | string | **Target** — `Low`/`Medium`/`High` |
""")
    handle_tab("maintenance_data","Maintenance Model",
               lambda df, t, p, pr, s, st_: train_classifier(df, t, "maintenance", p, pr, s, st_),
               "maintenance")