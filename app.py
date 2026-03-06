"""
 AI-Powered Risk Assessment Assistant for Insurance Underwriting
TCS AI Fridays Season 2  Hackathon

Features:
  - Upload insurance application (PDF/TXT) or use sample data
  - AI extracts risk factors and calculates preliminary risk score
  - Compliance checks against IRDAI, AML, DPDP Act
  - Visual risk dashboard with score breakdown
  - Underwriter recommendation with suggested next steps

Run: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from backend.risk_engine import (
    assess_risk,
    chat_with_underwriter_assistant,
    generate_report,
    is_api_configured,
)
from backend.sample_data import (
    get_sample_application_text,
    get_risk_factors,
    get_compliance_checks,
    get_risk_thresholds,
)
from backend.doc_extractor import extract_text_from_pdf, extract_text_from_pdf_path, extract_text_from_txt
from backend.pdf_export import assessment_to_pdf_bytes


@st.cache_data(show_spinner=False)
def load_tnc_text() -> str:
    """Load local T&C PDF once and cache it for chatbot/compliance context."""
    tnc_path = os.path.join(os.path.dirname(__file__), "T&C.pdf")
    return extract_text_from_pdf_path(tnc_path)


# 
# Page Config
# 
st.set_page_config(
    page_title="Lifeline",
    page_icon="I",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 
# Custom CSS
# 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Source+Serif+4:wght@400;600;700&display=swap');
    .stApp { font-family: 'Source Serif 4', Georgia, serif; }
    [data-testid='stSidebar'] { display: none !important; }
    [data-testid='collapsedControl'] { display: none !important; }
    
    .main-header { text-align: center; padding: 1rem 0; }
    .main-header h1 {
        font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem;
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    .risk-gauge {
        background: #111827; border: 1px solid #1f2937;
        border-radius: 16px; padding: 1.5rem; text-align: center;
    }
    .score-big {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 3rem; font-weight: 700;
    }
    .score-label { font-size: 0.9rem; color: #9ca3af; margin-top: 0.3rem; }

    .risk-low { color: #34d399; }
    .risk-medium { color: #fbbf24; }
    .risk-mid { color: #fbbf24; }
    .risk-moderate { color: #fbbf24; }
    .risk-high { color: #f87171; }
    .risk-very_high { color: #dc2626; }
    .risk-severely_high { color: #dc2626; }

    .factor-card {
        background: #111827; border: 1px solid #1f2937;
        border-radius: 12px; padding: 1rem; margin-bottom: 0.5rem;
    }
    .compliance-pass { color: #34d399; }
    .compliance-flag { color: #fbbf24; }
    .compliance-fail { color: #f87171; }

    .rec-approve { background: #064e3b; border: 1px solid #34d399; }
    .rec-conditions { background: #451a03; border: 1px solid #fbbf24; }
    .rec-refer { background: #450a0a; border: 1px solid #f87171; }
    .rec-decline { background: #1c1917; border: 1px solid #dc2626; }
    .rec-box {
        border-radius: 12px; padding: 1.2rem; text-align: center;
        margin: 1rem 0; font-size: 1.2rem; font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# 
# Session State
# 
if "assessment" not in st.session_state:
    st.session_state.assessment = None
if "report" not in st.session_state:
    st.session_state.report = None
if "app_text" not in st.session_state:
    st.session_state.app_text = None
if "eval_result" not in st.session_state:
    st.session_state.eval_result = None
if "timings" not in st.session_state:
    st.session_state.timings = {}
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "tnc_text" not in st.session_state:
    st.session_state.tnc_text = load_tnc_text()


# 
# Header
# 
st.markdown("""
<div class="main-header">
    <h1>Lifeline</h1>
    <p style="color:#9ca3af;">Upload an application -> AI extracts risk factors -> Get preliminary score and recommendation</p>
</div>
""", unsafe_allow_html=True)

if not is_api_configured():
    st.warning("API key is not configured in the environment. Set `GENAILAB_API_KEY` in `.env` before running analysis.")


# 
# Tab Layout
# 
tab_assess, tab_chat, tab_eval, tab_guide, tab_compliance = st.tabs(
    ["Assess Application", "Underwriter Chatbot", "Accuracy Evaluation", "Risk Scoring Guide", "Compliance Checks"]
)


# 
# TAB 1: ASSESS APPLICATION
# 
with tab_assess:
    input_col, result_col = st.columns([1, 1])

    with input_col:
        st.markdown("### Input Application")

        input_method = st.radio(
            "How would you like to provide the application?",
            ["Upload PDF/TXT file", "Paste text manually"],
            horizontal=True,
        )

        application_text = ""

        if input_method == "Upload PDF/TXT file":
            uploaded = st.file_uploader(
                "Upload insurance application",
                type=["pdf", "txt"],
                help="Upload the proposal form or application document",
            )
            if uploaded:
                t0 = time.perf_counter()
                if uploaded.name.endswith(".pdf"):
                    application_text = extract_text_from_pdf(uploaded)
                else:
                    application_text = extract_text_from_txt(uploaded)
                st.session_state.timings["extraction_s"] = round(time.perf_counter() - t0, 2)
                st.success(f"Extracted {len(application_text)} characters from {uploaded.name}")

        elif input_method == "Paste text manually":
            application_text = st.text_area(
                "Paste application text here",
                height=400,
                placeholder="Paste the insurance application form text here...",
            )

        # Analyze button
        if application_text:
            if st.button("Analyze Risk", type="primary", use_container_width=True):
                if not is_api_configured():
                    st.error("API key not configured. Set `GENAILAB_API_KEY` in `.env`.")
                else:
                    with st.spinner("AI is analyzing the application..."):
                        if input_method != "Upload PDF/TXT file":
                            st.session_state.timings["extraction_s"] = 0.0
                        t0 = time.perf_counter()
                        st.session_state.assessment = assess_risk(
                            application_text,
                            tnc_text=st.session_state.tnc_text,
                        )
                        st.session_state.timings["assessment_s"] = round(time.perf_counter() - t0, 2)
                        st.session_state.app_text = application_text
                    st.session_state.report = None
                    st.session_state.timings["report_s"] = None
                    st.rerun()

    #  Results Panel 
    with result_col:
        if st.session_state.assessment and "error" not in st.session_state.assessment:
            assessment = st.session_state.assessment
            summary = assessment.get("applicant_summary", {})
            factors = assessment.get("risk_factors", [])
            score = assessment.get("overall_risk_score", 0)
            category = assessment.get("risk_category", "UNKNOWN")
            category = {
                "MODERATE": "MEDIUM",
                "MID": "MEDIUM",
                "VERY_HIGH": "SEVERELY_HIGH",
                "VERY HIGH": "SEVERELY_HIGH",
                "SEVERLY_HIGH": "SEVERELY_HIGH",
                "SEVERELY HIGH": "SEVERELY_HIGH",
            }.get(
                str(category).upper(), str(category).upper()
            )
            recommendation = assessment.get("recommendation", "N/A")
            concerns = assessment.get("key_concerns", [])
            compliance = assessment.get("compliance_flags", [])
            conditions = assessment.get("conditions", [])
            actions = assessment.get("suggested_actions", [])

            factor_lookup = {str(f.get("factor", "")).strip().lower(): f for f in factors}
            get_factor = lambda *names: next((factor_lookup.get(n.lower()) for n in names if factor_lookup.get(n.lower())), None)

            # Applicant summary
            st.markdown("### Applicant summary")
            a1, a2 = st.columns(2)
            with a1:
                st.markdown(f"**Name:** {summary.get('name', 'N/A')}")
                st.markdown(f"**Age:** {summary.get('age', 'N/A')} | **Gender:** {summary.get('gender', 'N/A')}")
                st.markdown(f"**City:** {summary.get('city', 'N/A')}")
                st.markdown(f"**Occupation:** {summary.get('occupation', 'N/A')}")
            with a2:
                st.markdown(f"**Policy Type:** {summary.get('policy_type', 'N/A')}")
                st.markdown(f"**Coverage Amount:** {summary.get('coverage_amount', 'N/A')}")
                st.markdown(f"**Income:** {summary.get('income', 'N/A')}")
                st.markdown(f"**Policy Term:** {summary.get('policy_term', 'N/A')}")

            # Applicant History
            st.markdown("### Applicant History")
            smoking_factor = get_factor("Smoking")
            pre_existing_factor = get_factor("Pre-existing Conditions")
            family_history_factor = get_factor("Family History")
            lifestyle_factor = get_factor("Lifestyle")
            history_lines = [
                ("Smoking", smoking_factor),
                ("Pre-existing Conditions", pre_existing_factor),
                ("Family History", family_history_factor),
                ("Lifestyle", lifestyle_factor),
            ]
            has_history = False
            for label, factor in history_lines:
                if factor:
                    has_history = True
                    st.markdown(f"- **{label}:** {factor.get('value', 'N/A')} ({factor.get('risk_level', 'N/A')})")
            if not has_history:
                st.markdown("- No applicant history details were extracted.")

            # Preliminary Risk Score
            st.markdown("### Preliminary Risk Score")
            score_color = {"LOW": "#34d399", "MEDIUM": "#fbbf24", "HIGH": "#fb923c", "SEVERELY_HIGH": "#dc2626"}.get(category, "#9ca3af")
            score_class = f"risk-{category.lower()}"
            st.markdown(f"""
            <div class="risk-gauge">
                <div class="score-label">PRELIMINARY RISK SCORE (0-100)</div>
                <div class="score-big {score_class}">{score}%</div>
                <div style="color:{score_color}; font-size:1.2rem; font-weight:700;">{category}</div>
            </div>
            """, unsafe_allow_html=True)
            timing_bits = []
            extraction_s = st.session_state.timings.get("extraction_s")
            assessment_s = st.session_state.timings.get("assessment_s")
            report_s = st.session_state.timings.get("report_s")
            if extraction_s is not None:
                timing_bits.append(f"Extraction: {extraction_s}s")
            if assessment_s is not None:
                timing_bits.append(f"Risk Analysis: {assessment_s}s")
            if report_s is not None:
                timing_bits.append(f"Report: {report_s}s")
            if timing_bits:
                st.caption(" | ".join(timing_bits))

            # Recommendation
            st.markdown("### Recommendation")
            rec_class = {
                "APPROVE": "rec-approve",
                "APPROVE_WITH_CONDITIONS": "rec-conditions",
                "REFER_TO_SENIOR": "rec-refer",
                "DECLINE": "rec-decline",
            }.get(recommendation, "rec-refer")
            rec_display = recommendation.replace("_", " ")
            st.markdown(f'<div class="rec-box {rec_class}">{rec_display}</div>', unsafe_allow_html=True)

            # Preliminary Risk Insights
            st.markdown("### Preliminary Risk Insights")
            category_insight = {
                "LOW": "Overall profile indicates low underwriting risk based on submitted details.",
                "MEDIUM": "Risk is moderate and requires focused review on a few factors.",
                "HIGH": "High risk profile detected; human intervention and senior underwriting review are required.",
                "SEVERELY_HIGH": "Severely high risk indicators detected; decline is likely pending senior review.",
            }.get(category, "Risk profile needs manual validation due to limited confidence.")
            rec_insight = {
                "APPROVE": "Current evidence supports approval.",
                "APPROVE_WITH_CONDITIONS": "Approval is possible with additional controls or conditions.",
                "REFER_TO_SENIOR": "Case should be reviewed by a senior underwriter before decision.",
                "DECLINE": "Current risk pattern does not meet underwriting acceptance criteria.",
            }.get(recommendation, "Recommendation requires underwriter confirmation.")
            top_factors = sorted(factors, key=lambda f: float(f.get("score", 0) or 0), reverse=True)[:3]
            top_factor_labels = [f"{f.get('factor', 'Unknown')} ({f.get('score', 0)}/6)" for f in top_factors]
            st.markdown(f"- **Category Insight:** {category_insight}")
            st.markdown(f"- **Recommendation Insight:** {rec_insight}")
            if top_factor_labels:
                st.markdown(f"- **Top Contributing Factors:** {', '.join(top_factor_labels)}")
            if concerns:
                st.markdown(f"- **Primary Concern:** {concerns[0]}")

            # Risk Factor Breakdown
            st.markdown("### Risk Factor Breakdown")
            if factors:
                factor_names = [f.get("factor", "") for f in factors]
                factor_scores = [f.get("score", 0) for f in factors]
                factor_colors = [
                    "#34d399" if f.get("risk_level") == "LOW"
                    else "#fbbf24" if f.get("risk_level") == "MEDIUM"
                    else "#f87171" for f in factors
                ]
                fig = go.Figure(go.Bar(
                    x=factor_scores, y=factor_names,
                    orientation="h",
                    marker_color=factor_colors,
                    text=[f"{s}/6" for s in factor_scores],
                    textposition="auto",
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9ca3af"),
                    height=300,
                    margin=dict(t=10, b=10, l=120, r=20),
                    xaxis=dict(range=[0, 6.5], title="Factor Risk Score (0-6)"),
                )
                st.plotly_chart(fig, use_container_width=True)
                for f in factors:
                    icon = {"LOW": "", "MEDIUM": "", "HIGH": ""}.get(f.get("risk_level", ""), "")
                    st.markdown(f"{icon} **{f.get('factor', '')}**: {f.get('value', '')} - *{f.get('explanation', '')}*")
            else:
                st.markdown("- No risk factors were extracted.")

            # Coverage needs
            st.markdown("### Coverage needs")
            coverage_factor = get_factor("Coverage Amount")
            st.markdown(f"- **Declared Coverage:** {summary.get('coverage_amount', 'N/A')}")
            st.markdown(f"- **Declared Income:** {summary.get('income', 'N/A')}")
            if coverage_factor:
                st.markdown(f"- **Coverage Risk Level:** {coverage_factor.get('risk_level', 'N/A')} ({coverage_factor.get('score', 0)}/6)")
                st.markdown(f"- **Coverage Insight:** {coverage_factor.get('explanation', 'N/A')}")
            else:
                st.markdown("- Coverage adequacy insight was not extracted.")

            # Compliance checks
            st.markdown("### Compliance checks")
            if compliance:
                for c in compliance:
                    icon = {"PASS": "", "FLAG": "", "FAIL": ""}.get(c.get("status", ""), "")
                    st.markdown(f"{icon} **{c.get('rule_id', '')}**: {c.get('note', c.get('rule', ''))}")
            else:
                st.markdown("- No compliance checks were returned.")

            # key concerns
            st.markdown("### key concerns")
            if concerns:
                for c in concerns:
                    st.markdown(f"- {c}")
            else:
                st.markdown("- No key concerns were flagged.")

            # Conditions(if approved)
            st.markdown("### Conditions(if approved)")
            if conditions:
                for c in conditions:
                    st.markdown(f"- {c}")
            else:
                st.markdown("- No approval conditions specified.")

            # Suggested Next Steps
            st.markdown("### Suggested Next Steps")
            if actions:
                for a in actions:
                    st.markdown(f"- {a}")
            else:
                st.markdown("- No next steps were suggested.")

            #  Full AI Report 
            if not st.session_state.report:
                if st.button("Generate Underwriter Report", use_container_width=True):
                    with st.spinner("Generating underwriter report..."):
                        t0 = time.perf_counter()
                        st.session_state.report = generate_report(assessment)
                        st.session_state.timings["report_s"] = round(time.perf_counter() - t0, 2)
                    st.rerun()
            else:
                with st.expander(" Full AI Underwriter Report", expanded=False):
                    st.markdown(st.session_state.report)

            #  Downloads 
            st.markdown("---")
            d1, d2 = st.columns(2)
            with d1:
                try:
                    assessment_pdf = assessment_to_pdf_bytes(assessment)
                    st.download_button(
                        " Download Assessment (PDF)",
                        assessment_pdf,
                        "risk_assessment.pdf",
                        "application/pdf",
                    )
                except Exception:
                    st.error("PDF export unavailable. Install dependencies from backend/requirements.txt and retry.")
            with d2:
                if st.session_state.report:
                    st.download_button(
                        " Download Report (MD)",
                        st.session_state.report,
                        "underwriter_report.md", "text/markdown",
                    )

        elif st.session_state.assessment and "error" in st.session_state.assessment:
            st.error(st.session_state.assessment["error"])
        else:
            st.info(" Upload a PDF/TXT or paste application text, then click **Analyze Risk**.")


# 
# TAB 2: UNDERWRITER CHATBOT
# 
with tab_chat:
    st.markdown("### Underwriter Chatbot")
    st.markdown(
        "Ask underwriting questions grounded in uploaded application details, generated risk output, and local policy T&C."
    )

    if str(st.session_state.tnc_text).startswith("[ERROR]"):
        st.warning("T&C PDF could not be parsed. Chat will answer from application and assessment context only.")
    else:
        st.caption(f"T&C context loaded ({len(st.session_state.tnc_text)} chars from T&C.pdf)")

    if st.button("Clear Chat", use_container_width=False):
        st.session_state.chat_messages = []
        st.rerun()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about risk factors, recommendation rationale, or compliance gaps...")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing context..."):
                answer = chat_with_underwriter_assistant(
                    question=prompt,
                    application_text=st.session_state.app_text or "",
                    assessment=st.session_state.assessment or {},
                    report=st.session_state.report or "",
                    tnc_text=st.session_state.tnc_text or "",
                    chat_history=st.session_state.chat_messages,
                )
                st.markdown(answer)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})

    if not st.session_state.assessment:
        st.info("For case-specific answers, run **Analyze Risk** first in the Assess Application tab.")


# 
# TAB 3: ACCURACY EVALUATION
# 
with tab_eval:
    st.markdown("###  Accuracy Evaluation  Does the AI meet 80% target?")
    st.markdown("""
    This runs the AI on all **5 sample applications** and compares results against 
    **human-labeled ground truth**. Each application is checked for:
    
    -  **Factor Detection**  Did the AI find all 8 risk factors?
    -  **Risk Level Match**  Did it assign the correct LOW/MEDIUM/HIGH per factor?
    -  **Score in Range**  Is the numeric score within acceptable bounds?
    -  **Category Match**  Is the overall risk category correct?
    -  **Recommendation Match**  Is the final recommendation correct?
    """)

    if not is_api_configured():
        st.warning("API key not configured. Set `GENAILAB_API_KEY` in `.env` to run evaluation.")
    else:
        if st.button(" Run Full Accuracy Evaluation (5 applications)", type="primary", use_container_width=True):
            from backend.evaluator import GROUND_TRUTH, run_full_evaluation, format_eval_report

            assessments = []
            progress = st.progress(0, text="Starting evaluation...")

            for i in range(5):
                progress.progress((i) / 5, text=f"Assessing application {i+1}/5: {GROUND_TRUTH[i]['applicant']}...")
                app_text = get_sample_application_text(i)
                result = assess_risk(app_text)
                assessments.append(result)

            progress.progress(1.0, text="Running accuracy checks...")
            st.session_state.eval_result = run_full_evaluation(assessments)
            st.rerun()

        if st.session_state.eval_result:
            ev = st.session_state.eval_result

            # Big accuracy number
            overall = ev["overall_accuracy"]
            meets = ev["meets_80_target"]
            color = "#34d399" if meets else "#f87171"
            badge = " TARGET MET" if meets else " BELOW TARGET"

            st.markdown(f"""
            <div style="background:#111827; border:2px solid {color}; border-radius:16px; 
                        padding:2rem; text-align:center; margin:1rem 0;">
                <div style="color:#9ca3af; font-size:0.9rem;">OVERALL ACCURACY</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:3.5rem; 
                            font-weight:700; color:{color};">{overall}%</div>
                <div style="color:{color}; font-size:1.2rem; font-weight:600;">{badge}</div>
                <div style="color:#6b7280; font-size:0.85rem; margin-top:0.5rem;">
                    {ev['total_passed']}/{ev['total_checks']} checks passed | Target: 80%
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Per-applicant results
            st.markdown("### Per-Applicant Results")
            for r in ev["per_applicant"]:
                if "error" in r:
                    st.error(f" {r['applicant']}: {r['error']}")
                    continue

                acc = r["accuracy"]
                icon = "" if acc >= 80 else "" if acc >= 60 else ""
                acc_color = "#34d399" if acc >= 80 else "#fbbf24" if acc >= 60 else "#f87171"

                with st.expander(f"{icon} **{r['applicant']}**  {acc}% accuracy"):
                    mc1, mc2, mc3 = st.columns(3)
                    with mc1:
                        st.metric("Factors Detected", f"{r['factors_detected']}/{r['factors_expected']}")
                    with mc2:
                        st.metric("Risk Levels Correct", f"{r['risk_levels_correct']}/{r['factors_expected']}")
                    with mc3:
                        st.metric("Scores in Range", f"{r['scores_in_range']}/{r['factors_expected']}")

                    cat_icon = "" if r["category_match"] else ""
                    rec_icon = "" if r["recommendation_match"] else ""
                    score_icon = "" if r["score_in_range"] else ""

                    st.markdown(f"{cat_icon} **Category:** AI={r['ai_category']} vs Expected={r['expected_category']}")
                    st.markdown(f"{score_icon} **Score:** AI={r['ai_score']} vs Expected={r['expected_score_range']}")
                    st.markdown(f"{rec_icon} **Recommendation:** AI={r['ai_recommendation']} vs Expected={r['expected_recommendation']}")

                    # Per-factor detail
                    for fr in r.get("factor_results", []):
                        det = "" if fr["detected"] else ""
                        lvl = "" if fr["risk_level_match"] else ""
                        scr = "" if fr["score_in_range"] else ""
                        st.markdown(
                            f"&nbsp;&nbsp;&nbsp;&nbsp;{det} **{fr['factor']}**  "
                            f"Level: {lvl} (AI={fr['ai_risk_level']} vs {fr['expected_risk_level']}) | "
                            f"Score: {scr} (AI={fr['ai_score']} vs {fr['expected_score_range']})"
                        )

            # Per-factor accuracy table
            st.markdown("### Per-Factor Accuracy")
            import pandas as pd
            factor_data = []
            for fname, facc in ev.get("per_factor", {}).items():
                factor_data.append({
                    "Factor": fname,
                    "Detection %": facc["detection_rate"],
                    "Level Accuracy %": facc["level_accuracy"],
                    "Score Accuracy %": facc["score_accuracy"],
                })
            if factor_data:
                st.dataframe(pd.DataFrame(factor_data), use_container_width=True, hide_index=True)

            # Download full report
            st.markdown("---")
            from backend.evaluator import format_eval_report
            report_md = format_eval_report(ev)
            d1, d2 = st.columns(2)
            with d1:
                st.download_button(" Download Eval Report (MD)", report_md, "accuracy_report.md", "text/markdown")
            with d2:
                st.download_button(" Download Raw Data (JSON)", json.dumps(ev, indent=2), "accuracy_data.json", "application/json")


# 
# TAB 3: RISK SCORING GUIDE
# 
with tab_guide:
    st.markdown("###  Risk Factor Definitions & Scoring Guide")
    st.markdown("Reference guide for how the AI scores each risk factor.")

    factors = get_risk_factors()
    for factor_name, levels in factors.items():
        st.markdown(f"#### {factor_name.replace('_', ' ').title()}")
        for level_name, info in levels.items():
            score = info.get("score", 0)
            desc = info.get("description", info.get("examples", info.get("range", "")))
            bar = "" * min(score, 6) + "" * (6 - min(score, 6))
            st.markdown(f"- **{level_name}**: {desc}  Score: {score} {bar}")
        st.markdown("")

    st.markdown("---")
    st.markdown("###  Score Thresholds")
    thresholds = get_risk_thresholds()
    for level, info in thresholds.items():
        st.markdown(f"**{level}** ({info['min_score']}-{info['max_score']}): {info['recommendation']}")

    st.markdown("---")
    st.markdown("###  Regulatory Framework")
    st.markdown("""
    - **IRDAI**: Insurance Regulatory and Development Authority of India
    - **Section 45 (Insurance Act)**: Non-disclosure of material facts voids policy
    - **DPDP Act 2023**: Digital Personal Data Protection  governs applicant data handling
    - **AML/KYC**: Anti-Money Laundering & Know Your Customer for policies above 50,000
    - **Free-look period**: 15 days for life insurance, 30 days for health insurance
    - **Pre-existing condition waiting**: 48 months standard under IRDAI guidelines
    """)


# 
# TAB 4: COMPLIANCE CHECKS
# 
with tab_compliance:
    st.markdown("### Compliance Checks")
    st.markdown("Regulatory and underwriting checks used by the risk engine.")
    checks = get_compliance_checks()
    for c in checks:
        st.markdown(f"- **{c['id']}** ({c.get('category', 'General')}): {c['rule']}")

