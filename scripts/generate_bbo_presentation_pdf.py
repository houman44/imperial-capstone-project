#!/usr/bin/env python3
"""Generate BBO capstone presentation PDF (standalone; requires fpdf2)."""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF


def add_section(pdf: FPDF, title: str, body: str) -> None:
    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(0, 9, title)
    pdf.ln(2)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, body)
    pdf.ln(4)


def build_pdf(out_path: Path) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, "BBO Capstone Project: Methodology and Reflection")
    pdf.ln(4)
    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 5, "Black-box optimisation: eight benchmark functions, sequential queries, GP + Expected Improvement.")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    add_section(
        pdf,
        "1. Overview: what we are trying to achieve",
        "Goal: improve the best observed score y for each hidden function f(x) on the box [0,1]^d, where d differs per "
        "function. We only see (x, y) pairs returned by the external evaluator (Kaggle); the analytic form of f is "
        "unknown.\n\n"
        "Core objective: maximise or minimise y according to each function's specification, under a limited query "
        "budget.\n\n"
        "Process (high level): propose query points, obtain y, update a surrogate belief about each landscape, repeat. "
        "Locally this is supported by importing CSV results and generating the next batch with capstone_manager.py "
        "(Gaussian Process surrogate, Expected Improvement, random candidate search).",
    )

    add_section(
        pdf,
        "2. How the strategy evolved",
        "Early rounds: broad exploration to learn scales, spot flat versus spiky behaviour, and test corners and edges "
        "of the domain where optima often lie under box constraints.\n\n"
        "Later rounds: a more systematic loop anchored on one reproducible pipeline; less ad hoc coordinate picking; "
        "clear tracking of the running best (incumbent) per function.\n\n"
        "Drivers of change: where past good y values clustered; surrogate uncertainty shrinking in some regions; "
        "standard ML intuition that smooth landscapes suit GP models, while surprises in y justified extra exploration.\n\n"
        "Heuristics now: allocate more queries to higher-dimensional functions where evidence is harder to accumulate; "
        "use acquisition scores to justify each proposed x; prefer reproducible batches from one script.",
    )

    add_section(
        pdf,
        "3. Patterns, data, and insights",
        "Trends: raw y values are not comparable across functions (different scales). Improvement is always judged "
        "relative to each function's own best-so-far and its maximise versus minimise rule.\n\n"
        "Strong influences: (1) proximity to boundaries of [0,1]^d; (2) effective dimension (higher D needs more "
        "evidence); (3) function-specific geometry (some objectives change slowly, others jump between basins).\n\n"
        "Interpretation: optimisation is eight parallel searches, not one. The useful 'directions' are regions of "
        "input space where uncertainty and upside overlap, analogous in spirit to focusing on high-variance directions "
        "in PCA, but here the goal is sequential decision-making under a budget.",
    )

    pdf.add_page()
    add_section(
        pdf,
        "4. Decision-making and iteration",
        "Exploration versus exploitation: Expected Improvement trades off sampling points predicted to beat the "
        "current best against probing where the model is uncertain. A small xi term preserves exploration late in "
        "the run.\n\n"
        "What worked: deliberate boundary and corner probes when earlier strong y appeared near edges.\n\n"
        "What was harder: queries too close to existing x, yielding little new information (redundancy).\n\n"
        "Uncertainty: unexpected y updates the surrogate; treat odd scales as something to verify before overfitting "
        "a story to a single row.",
    )

    add_section(
        pdf,
        "5. Next steps and broader reflection",
        "Next steps: more local refinement around incumbents where the model is confident; reserve exploratory shots "
        "only where posterior variance remains high (often higher-D functions). Optionally tune kernel length scales "
        "or candidate counts if time allows.\n\n"
        "Broader ML: black-box optimisation mirrors hyperparameter search and adaptive experiment design: expensive "
        "evaluations, a belief model, and sequential decisions. Ideas about which directions in x matter connect to "
        "importance and representation, even though x here is a design vector, not raw features.\n\n"
        "Stakeholder summary: 'We used a statistical model to suggest which tests would most improve our best "
        "result, balancing promising regions and uncertain ones, similar to disciplined A/B testing with memory of "
        "all past tests.'",
    )

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(
        0,
        5,
        "Generated for the BBO capstone presentation activity. Edit scripts/generate_bbo_presentation_pdf.py and "
        "re-run to customise.",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "docs" / "BBO_Capstone_Presentation.pdf"
    build_pdf(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
