"""
PDF export helpers for assessment output.
"""

import io
from typing import Any


def _plain_text(value: Any) -> str:
    """Convert values to PDF-safe plain text without currency symbols."""
    text = str(value or "")
    replacements = {
        "₹": "",
        "$": "",
        "€": "",
        "£": "",
        "—": "-",
        "–": "-",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return " ".join(text.split())


def assessment_to_pdf_bytes(assessment: dict[str, Any]) -> bytes:
    """
    Render the assessment into a readable text-style PDF and return raw bytes.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    left = 40
    top = height - 40
    bottom = 40
    font_name = "Helvetica"
    font_size = 10
    line_gap = 14

    c.setTitle("Risk Assessment")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(left, top, "Insurance Risk Assessment")
    y = top - 24

    summary = assessment.get("applicant_summary", {}) or {}
    factors = assessment.get("risk_factors", []) or []
    compliance = assessment.get("compliance_flags", []) or []
    concerns = assessment.get("key_concerns", []) or []
    actions = assessment.get("suggested_actions", []) or []
    conditions = assessment.get("conditions", []) or []
    score = assessment.get("overall_risk_score", "N/A")
    category = assessment.get("risk_category", "N/A")
    recommendation = assessment.get("recommendation", "N/A")

    lines: list[str] = []
    lines.append("Applicant Summary")
    lines.append(f"Name: {_plain_text(summary.get('name', 'N/A'))}")
    lines.append(f"Age: {_plain_text(summary.get('age', 'N/A'))}    Gender: {_plain_text(summary.get('gender', 'N/A'))}")
    lines.append(f"Occupation: {_plain_text(summary.get('occupation', 'N/A'))}")
    lines.append(f"City: {_plain_text(summary.get('city', 'N/A'))}")
    lines.append(f"Income: {_plain_text(summary.get('income', 'N/A'))}")
    lines.append(f"Policy Type: {_plain_text(summary.get('policy_type', 'N/A'))}")
    lines.append(f"Coverage Amount: {_plain_text(summary.get('coverage_amount', 'N/A'))}")
    lines.append(f"Policy Term: {_plain_text(summary.get('policy_term', 'N/A'))}")
    lines.append("")

    lines.append("Risk Score and Recommendation")
    lines.append(f"Overall Risk Score: {_plain_text(score)}/100")
    lines.append(f"Risk Category: {_plain_text(category)}")
    lines.append(f"Recommendation: {_plain_text(recommendation)}")
    lines.append("")

    lines.append("Risk Factors")
    if factors:
        for f in factors:
            lines.append(
                f"- {_plain_text(f.get('factor', 'Unknown'))}: {_plain_text(f.get('value', 'N/A'))} | "
                f"Level: {_plain_text(f.get('risk_level', 'N/A'))} | Score: {_plain_text(f.get('score', 'N/A'))}/6"
            )
            explanation = str(f.get("explanation", "") or "").strip()
            if explanation:
                lines.append(f"  Why: {_plain_text(explanation)}")
    else:
        lines.append("- No risk factors found.")
    lines.append("")

    lines.append("Compliance Checks")
    if compliance:
        for item in compliance:
            lines.append(
                f"- {item.get('rule_id', 'N/A')} | {item.get('status', 'N/A')}: "
                f"{_plain_text(item.get('note', item.get('rule', 'N/A')))}"
            )
    else:
        lines.append("- No compliance checks returned.")
    lines.append("")

    lines.append("Key Concerns")
    if concerns:
        for item in concerns:
            lines.append(f"- {_plain_text(item)}")
    else:
        lines.append("- No key concerns.")
    lines.append("")

    lines.append("Conditions")
    if conditions:
        for item in conditions:
            lines.append(f"- {_plain_text(item)}")
    else:
        lines.append("- No conditions specified.")
    lines.append("")

    lines.append("Suggested Next Steps")
    if actions:
        for item in actions:
            lines.append(f"- {_plain_text(item)}")
    else:
        lines.append("- No suggested next steps.")

    c.setFont(font_name, font_size)
    for raw_line in lines:
        # Wrap long lines to fit page width.
        line = raw_line or " "
        while line:
            max_chars = max(1, int((width - (left * 2)) / (font_size * 0.52)))
            chunk = line[:max_chars]

            if y < bottom:
                c.showPage()
                c.setFont(font_name, font_size)
                y = top

            c.drawString(left, y, chunk)
            y -= line_gap
            line = line[max_chars:]

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
