import os
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER


# ---------------------------------------------------------
# FONT REGISTRATION (ONLY DejaVuSans — no bold/italic!)
# ---------------------------------------------------------
FONT_PATH = "assets/fonts/DejaVuSans.ttf"
if os.path.exists(FONT_PATH):
    pdfmetrics.registerFont(TTFont("DejaVu", FONT_PATH))
else:
    raise FileNotFoundError("DejaVuSans.ttf not found in assets/fonts folder.")


# ---------------------------------------------------------
# CUSTOM SAFE STYLES (NO BOLD/ITALIC FLAGS)
# ---------------------------------------------------------
TitleStyle = ParagraphStyle(
    name="TitleStyle",
    fontName="DejaVu",
    fontSize=20,
    leading=22,
    alignment=TA_CENTER,
    spaceAfter=16
)

HeadingStyle = ParagraphStyle(
    name="HeadingStyle",
    fontName="DejaVu",
    fontSize=14,
    leading=16,
    alignment=TA_LEFT,
    spaceAfter=8
)

BodyStyle = ParagraphStyle(
    name="BodyStyle",
    fontName="DejaVu",
    fontSize=11,
    leading=13,
    alignment=TA_LEFT,
    spaceAfter=4
)


# ============================================================================================
# RAG REPORT
# ============================================================================================
def generate_rag_report(query, answer, sources):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    content = []

    content.append(Paragraph("RAG Query Report", TitleStyle))
    content.append(Spacer(1, 12))

    content.append(Paragraph("User Question", HeadingStyle))
    content.append(Paragraph(query.replace("\n", "<br/>"), BodyStyle))
    content.append(Spacer(1, 10))

    content.append(Paragraph("AI Answer", HeadingStyle))
    content.append(Paragraph(answer.replace("\n", "<br/>"), BodyStyle))
    content.append(Spacer(1, 14))

    content.append(Paragraph("Top Sources (Matched Paragraphs)", HeadingStyle))

    for s in sources:
        content.append(Paragraph(f"<b>Source:</b> {s['source']} (score={s['score']:.2f})", BodyStyle))
        content.append(Paragraph(s['paragraph'].replace("\n", "<br/>"), BodyStyle))
        content.append(Spacer(1, 12))


    doc.build(content)
    buffer.seek(0)
    return buffer





# ============================================================================================
# OPTIMIZATION REPORT
# ============================================================================================
def generate_optimization_report(
        inputs,
        outputs,
        fig_energy_path=None,
        fig_hist_path=None,
        fig_gauge_path=None,
        insights=""
):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    content = []

    content.append(Paragraph("Optimization Report", TitleStyle))
    content.append(Spacer(1, 12))

    # ---------------- INPUTS ----------------
    content.append(Paragraph("Input Parameters", HeadingStyle))
    for key, value in inputs.items():
        content.append(Paragraph(f"{key}: {value}", BodyStyle))
    content.append(Spacer(1, 10))

    # ---------------- OUTPUTS ----------------
    content.append(Paragraph("Optimization Results", HeadingStyle))
    for key, value in outputs.items():
        content.append(Paragraph(f"{key}: {value}", BodyStyle))
    content.append(Spacer(1, 10))

    # ---------------- INSIGHTS ----------------
    if insights:
        content.append(Paragraph("Insights", HeadingStyle))
        content.append(Paragraph(insights.replace("\n", "<br/>"), BodyStyle))
        content.append(Spacer(1, 10))

    # ---------------- CHARTS ----------------
    if fig_energy_path and os.path.exists(fig_energy_path):
        content.append(Paragraph("Energy Variation", HeadingStyle))
        content.append(Image(fig_energy_path, width=360, height=180))
        content.append(Spacer(1, 10))

    if fig_hist_path and os.path.exists(fig_hist_path):
        content.append(Paragraph("Energy Distribution", HeadingStyle))
        content.append(Image(fig_hist_path, width=360, height=180))
        content.append(Spacer(1, 10))

    if fig_gauge_path and os.path.exists(fig_gauge_path):
        content.append(Paragraph("Emission Intensity Gauge", HeadingStyle))
        content.append(Image(fig_gauge_path, width=260, height=260))
        content.append(Spacer(1, 10))

    doc.build(content)
    buffer.seek(0)
    return buffer


# ============================================================================================
# WHAT-IF REPORT
# ============================================================================================
def generate_whatif_report(query, result_text, metrics, fig_mc=None, fig_pareto=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    content = []

    content.append(Paragraph("What-If Analysis Report", TitleStyle))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Input Parameters", HeadingStyle))
    content.append(Paragraph(query.replace("\n", "<br/>"), BodyStyle))
    content.append(Spacer(1, 10))

    content.append(Paragraph("Simulation Metrics", HeadingStyle))
    for key, val in metrics.items():
        content.append(Paragraph(f"{key}: {val}", BodyStyle))
    content.append(Spacer(1, 10))

    content.append(Paragraph("Recommendation", HeadingStyle))
    content.append(Paragraph(result_text.replace("\n", "<br/>"), BodyStyle))
    content.append(Spacer(1, 10))

    # Charts
    if fig_mc:
        img = BytesIO()
        fig_mc.savefig(img, format="png", dpi=200, bbox_inches="tight")
        img.seek(0)
        content.append(Paragraph("Monte Carlo Simulation", HeadingStyle))
        content.append(Image(img, width=360, height=200))
        content.append(Spacer(1, 12))

    if fig_pareto:
        img = BytesIO()
        fig_pareto.savefig(img, format="png", dpi=200, bbox_inches="tight")
        img.seek(0)
        content.append(Paragraph("Pareto Frontier", HeadingStyle))
        content.append(Image(img, width=360, height=200))

    doc.build(content)
    buffer.seek(0)
    return buffer
