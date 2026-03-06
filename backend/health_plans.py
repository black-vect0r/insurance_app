"""Synthetic health insurance plans and recommendation helper."""

from __future__ import annotations

from typing import Any


HEALTH_PLANS = [
    {
        "plan_id": "HL-ESS-01",
        "name": "Lifeline Essential Care",
        "sum_insured": "Rs 5L to Rs 15L",
        "age_band": "18-45",
        "premium_band": "Rs 7,500 - Rs 12,500/year",
        "waiting_period": "24 months for named pre-existing diseases",
        "copay": "No mandatory copay till age 60",
        "good_for": "Low-risk applicants seeking affordable hospitalization cover",
        "max_risk_score": 35,
    },
    {
        "plan_id": "HL-BAL-02",
        "name": "Lifeline Balanced Shield",
        "sum_insured": "Rs 10L to Rs 30L",
        "age_band": "25-60",
        "premium_band": "Rs 12,000 - Rs 22,000/year",
        "waiting_period": "24-36 months depending on condition",
        "copay": "10% copay above age 60",
        "good_for": "Moderate-risk applicants with family history or minor conditions",
        "max_risk_score": 60,
    },
    {
        "plan_id": "HL-PRM-03",
        "name": "Lifeline Prime Health Plus",
        "sum_insured": "Rs 25L to Rs 1Cr",
        "age_band": "30-65",
        "premium_band": "Rs 20,000 - Rs 48,000/year",
        "waiting_period": "24 months standard, disease-wise sub-limits",
        "copay": "Optional 10% copay discount rider",
        "good_for": "High coverage need, urban professionals, larger family requirements",
        "max_risk_score": 75,
    },
    {
        "plan_id": "HL-SEN-04",
        "name": "Lifeline Senior Secure",
        "sum_insured": "Rs 5L to Rs 20L",
        "age_band": "55-80",
        "premium_band": "Rs 28,000 - Rs 65,000/year",
        "waiting_period": "36 months with mandatory health check",
        "copay": "20% mandatory copay",
        "good_for": "Senior citizens with managed chronic conditions",
        "max_risk_score": 90,
    },
    {
        "plan_id": "HL-CRI-05",
        "name": "Lifeline Critical Protect Rider Pack",
        "sum_insured": "Rs 10L to Rs 50L",
        "age_band": "21-60",
        "premium_band": "Rs 5,000 - Rs 18,000/year (add-on)",
        "waiting_period": "90 days initial survival period",
        "copay": "Not applicable (fixed benefit rider)",
        "good_for": "Applicants with family history of cancer/cardiac risk wanting lump-sum support",
        "max_risk_score": 70,
    },
]


def _age_in_band(age: int, band: str) -> bool:
    try:
        low, high = band.split("-")
        return int(low) <= int(age) <= int(high)
    except Exception:
        return True


def get_health_plan_catalog() -> list[dict[str, Any]]:
    return HEALTH_PLANS


def recommend_health_plans(
    assessment: dict | None,
    max_plans: int = 3,
    profile: dict | None = None,
    risk_score: int | None = None,
) -> list[dict[str, Any]]:
    data = assessment or {}
    profile = profile or {}
    summary = data.get("applicant_summary", {})

    if risk_score is None:
        try:
            risk_score = int(data.get("overall_risk_score", 50) or 50)
        except Exception:
            risk_score = 50

    try:
        age = int(profile.get("age", summary.get("age", 40)) or 40)
    except Exception:
        age = 40

    smoking = str(profile.get("smoking", "")).lower()
    pre_existing = str(profile.get("pre_existing", "")).lower()

    scored = []
    for plan in HEALTH_PLANS:
        fit_score = 0

        if _age_in_band(age, str(plan.get("age_band", ""))):
            fit_score += 3
        else:
            fit_score -= 3

        if risk_score <= int(plan.get("max_risk_score", 100)):
            fit_score += 3
        else:
            fit_score -= 4

        pid = str(plan.get("plan_id", ""))
        name = str(plan.get("name", ""))

        if age >= 55 and ("SEN" in pid or "Senior" in name):
            fit_score += 4
        if age < 45 and "SEN" in pid:
            fit_score -= 2

        if smoking == "current":
            if "ESS" in pid:
                fit_score -= 3
            if "BAL" in pid or "PRM" in pid or "SEN" in pid:
                fit_score += 1

        if pre_existing in {"moderate", "severe"}:
            if "ESS" in pid:
                fit_score -= 3
            if "BAL" in pid or "PRM" in pid or "SEN" in pid:
                fit_score += 2

        scored.append((fit_score, plan))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [plan for _, plan in scored[:max_plans]]


def format_plan_recommendations(plans: list[dict[str, Any]], risk_summary: str = "") -> str:
    if not plans:
        return "No suitable plan recommendations available from current profile."

    lines = []
    if risk_summary:
        lines.append(risk_summary)
        lines.append("")

    lines.append("Recommended Health Insurance Plans:")
    for idx, plan in enumerate(plans, start=1):
        lines.append(
            f"{idx}. {plan['name']} ({plan['plan_id']}) | Sum Insured: {plan['sum_insured']} | "
            f"Premium: {plan['premium_band']} | Waiting: {plan['waiting_period']}"
        )
        lines.append(f"   Best fit: {plan['good_for']}")
    lines.append("Note: Final premium and eligibility depend on underwriting and medical tests.")
    return "\n".join(lines)
