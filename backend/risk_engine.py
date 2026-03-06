"""
AI Risk Assessment Engine
Uses TCS GenAI Lab to analyze insurance applications and generate:
  - Risk factor extraction
  - Preliminary risk score
  - Compliance flags
  - Underwriter recommendations
"""

import json
import os
import re
from pathlib import Path

import httpx

from backend.health_plans import (
    format_plan_recommendations,
    get_health_plan_catalog,
    recommend_health_plans,
)

# -----------------------------------------------------------------------------
# Secure API Key Loading
# -----------------------------------------------------------------------------
def _load_env_file():
    env_paths = [Path(__file__).parent.parent / ".env", Path.cwd() / ".env"]
    for env_path in env_paths:
        if env_path.exists():
            try:
                raw = env_path.read_bytes()
                raw = raw.replace(b"\x00", b"").replace(b"\xff\xfe", b"")
                raw = raw.replace(b"\xfe\xff", b"").replace(b"\xef\xbb\xbf", b"")
                text = raw.decode("utf-8", errors="ignore")
                for line in text.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
                return True
            except Exception:
                continue
    return False


def _get_api_key():
    _load_env_file()
    return os.environ.get("GENAILAB_API_KEY", os.environ.get("LLM_API_KEY", "")).strip()


def is_api_configured():
    key = _get_api_key()
    return bool(key and key.startswith("sk-") and len(key) > 10)


def get_api_status():
    key = _get_api_key()
    if key and key.startswith("sk-") and len(key) > 10:
        return f"Key: {key[:6]}...{key[-4:]}"
    return "Not configured - add key in .env file"


def set_api_key_runtime(key):
    os.environ["GENAILAB_API_KEY"] = key.strip()


# -----------------------------------------------------------------------------
# LLM Configuration
# -----------------------------------------------------------------------------
BASE_URL = os.environ.get("GENAILAB_BASE_URL", "https://genailab.tcs.in")
MODEL = os.environ.get("GENAILAB_MODEL", "azure_ai/genailab-maas-DeepSeek-V3-0324")
MODEL_FAST = os.environ.get("GENAILAB_MODEL_FAST", "azure/genailab-maas-gpt-4o-mini")
MAX_APPLICATION_CHARS = int(os.environ.get("RISK_INPUT_MAX_CHARS", "12000"))
MAX_TNC_CONTEXT_CHARS = int(os.environ.get("TNC_CONTEXT_MAX_CHARS", "5000"))

http_client = httpx.Client(verify=False)


def _create_llm(model, temperature=0.2, max_tokens=3000):
    from langchain_openai import ChatOpenAI

    key = _get_api_key()
    if not key:
        return None
    return ChatOpenAI(
        base_url=BASE_URL,
        model=model,
        api_key=key,
        http_client=http_client,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _trim_application_text(text: str) -> str:
    """Limit input size to reduce latency and token usage for large uploads."""
    content = str(text or "").strip()
    if len(content) <= MAX_APPLICATION_CHARS:
        return content

    head_len = int(MAX_APPLICATION_CHARS * 0.75)
    tail_len = MAX_APPLICATION_CHARS - head_len
    return content[:head_len] + "\n\n...[TRUNCATED FOR PERFORMANCE]...\n\n" + content[-tail_len:]


def _trim_tnc_text(text: str) -> str:
    content = str(text or "").strip()
    if len(content) <= MAX_TNC_CONTEXT_CHARS:
        return content
    return content[:MAX_TNC_CONTEXT_CHARS] + "\n\n...[TRUNCATED TNC CONTEXT]..."


# -----------------------------------------------------------------------------
# System Prompts
# -----------------------------------------------------------------------------
RISK_ASSESSMENT_PROMPT = """You are an expert Insurance Underwriting AI Assistant for the Indian insurance market.
You analyze insurance applications and provide structured risk assessments.

Given an insurance application text, you MUST return a JSON response (no markdown fences) with this EXACT structure:

{
  "applicant_summary": {
    "name": "string",
    "age": number,
    "gender": "string",
    "occupation": "string",
    "city": "string",
    "income": "string",
    "policy_type": "string",
    "coverage_amount": "string",
    "policy_term": "string"
  },
  "risk_factors": [
    {
      "factor": "string (e.g., Age, BMI, Smoking, Pre-existing Conditions, Occupation, Family History, Coverage Amount, Lifestyle)",
      "value": "string (actual value from application)",
      "risk_level": "LOW | MEDIUM | HIGH",
      "score": number (0-6),
      "explanation": "string (why this is flagged)"
    }
  ],
  "compliance_flags": [
    {
      "rule_id": "string (e.g., IRDAI-01, AML-01)",
      "rule": "string",
      "status": "PASS | FLAG | FAIL",
      "note": "string"
    }
  ],
  "overall_risk_score": number (0-100 risk percentage),
  "risk_category": "LOW | MEDIUM | HIGH | SEVERELY_HIGH",
  "recommendation": "APPROVE | APPROVE_WITH_CONDITIONS | REFER_TO_SENIOR | DECLINE",
  "conditions": ["string array of conditions if applicable"],
  "key_concerns": ["string array of top concerns"],
  "suggested_actions": ["string array of next steps for underwriter"]
}

SCORING GUIDE:
- Compute raw factor scores using the factor rubric below.
- Convert to percentage: overall_risk_score = round((sum_of_factor_scores / 48) * 100).
- Category thresholds:
  - <30: LOW
  - 30-59: MEDIUM
  - 60-80: HIGH
  - >80: SEVERELY_HIGH
- Recommendation guidance:
  - LOW: APPROVE
  - MEDIUM: APPROVE_WITH_CONDITIONS
  - HIGH: REFER_TO_SENIOR (human intervention required)
  - SEVERELY_HIGH: DECLINE (likely), or REFER_TO_SENIOR if uncertainty is high

RISK FACTORS TO EVALUATE:
1. Age: 25-45 (low=1), 46-60 or 18-24 (medium=3), 61+ (high=5)
2. BMI: 18.5-24.9 (low=1), 25-29.9 (medium=3), 30+ (high=5)
3. Smoking: Non-smoker (0), Former (2), Current (5)
4. Pre-existing Conditions: None (0), Mild (2), Moderate (4), Severe (6)
5. Occupation Risk: Office/IT (1), Sales/Nurse (3), Construction/Mining (5)
6. Family History: None (0), Moderate (2), Serious (4)
7. Coverage to Income ratio: Under 10x (1), 10-15x (2), Over 15x (3)
8. Lifestyle: Low risk (0), Moderate (1), High risk/extreme sports (4)

COMPLIANCE CHECKS (Indian regulatory):
- IRDAI-01: Full medical disclosure present?
- IRDAI-02: Pre-existing conditions waiting period applicable?
- IRDAI-03: Free-look period communicated?
- IRDAI-04: Sum assured within income multiple limits?
- IRDAI-05: Nominee or beneficiary details present?
- IRDAI-06: Policy exclusions and terms acknowledged?
- AML-01: KYC verification needed? (policies above Rs 50,000)
- AML-02: PEP screening needed?
- AML-03: Source of funds/income proof captured for higher-value policy?
- AML-04: Sanctions/adverse media screening required?
- DATA-01: DPDP Act 2023 compliance?
- DATA-02: Explicit applicant consent captured?
- DISC-01: Any material fact non-disclosure risk?
- UW-01: Any contradiction in medical declaration vs medications?
- UW-02: Prior claims/rejections explicitly disclosed?

Be thorough but concise. Return ONLY valid JSON."""


EXPLANATION_PROMPT = """You are a senior insurance underwriter AI assistant.
Given a structured risk assessment (JSON), write a clear, professional summary report suitable for an underwriter review.

Format the report in markdown with these sections:
## Applicant Overview
Brief summary of who the applicant is.

## Key Risk Factors
List each risk factor found, with severity and explanation.

## Risk Score and Recommendation
State the total score, risk category, and recommendation clearly.

## Key Concerns
Top concerns that need attention.

## Compliance Notes
Any regulatory flags (IRDAI, AML, DPDP Act).

## Suggested Next Steps
What the underwriter should do next.

Be professional, data-driven, and reference Indian insurance regulations (IRDAI) where relevant.
Use Rs for monetary values and Indian terminology."""


CHATBOT_PROMPT = """You are a Health Insurance Underwriting Chatbot for underwriters and advisors.
Answer only using:
1) Provided application details,
2) Structured risk assessment,
3) Provided Terms and Conditions excerpt.

Rules:
- Be concise and practical.
- If evidence is missing, say "Not found in provided documents".
- Mention relevant compliance rule IDs where possible.
- Do not provide legal advice; provide underwriting guidance.
- Prioritize health insurance discussion (hospitalization, PED waiting period, copay, exclusions, room rent limits).
- If the user asks for plan options, recommend only from the provided plan_catalog context.
- For any risk prediction request, ask for missing profile inputs before estimating risk.
- Use uploaded application/assessment context only when the question is explicitly case-specific.
- For general questions, answer with general insurance guidance and do not anchor to the uploaded case.
"""

PLAN_RECOMMENDATION_PROMPT = """You are a health insurance plan recommendation assistant.
Use ONLY the provided context JSON.

Output format:
1) One-line profile summary
2) One-line risk summary (if available)
3) Exactly 3 recommended plans from shortlisted_plans, ordered best to acceptable
4) For each plan, explain why it fits this user's profile
5) List missing details that would improve recommendation quality (if any)

Rules:
- Do not invent plans outside plan_catalog/shortlisted_plans.
- Prioritize suitability for age, smoking status, pre-existing conditions, and inferred risk.
- Keep response concise and practical.
"""


# -----------------------------------------------------------------------------
# Normalization and Helpers
# -----------------------------------------------------------------------------
def _normalize_risk_category(category: str) -> str:
    cat = str(category or "").strip().upper()
    if cat in {"LOW", "MEDIUM", "HIGH", "SEVERELY_HIGH"}:
        return cat
    if cat in {"MODERATE", "MID"}:
        return "MEDIUM"
    if cat in {"VERY_HIGH", "SEVERE_HIGH", "SEVERLY_HIGH"}:
        return "SEVERELY_HIGH"
    return cat or "UNKNOWN"


def _normalize_assessment(result: dict) -> dict:
    normalized = dict(result)

    factors = normalized.get("risk_factors", []) or []
    factor_total = 0.0
    for factor in factors:
        try:
            factor_total += float(factor.get("score", 0) or 0)
        except Exception:
            continue
    max_possible_score = max(1.0, float(len(factors) * 6)) if factors else 1.0
    normalized_score_from_factors = round((factor_total / max_possible_score) * 100) if factors else 0

    raw_score = normalized.get("overall_risk_score", 0)
    try:
        score = float(raw_score)
    except Exception:
        score = None

    if factors:
        score = normalized_score_from_factors
    elif score is None:
        score = normalized_score_from_factors
    elif score > 100:
        score = normalized_score_from_factors
    else:
        score = round(max(0, min(100, score)))

    normalized["overall_risk_score"] = int(max(0, min(100, score)))

    if score < 30:
        normalized["risk_category"] = "LOW"
    elif score < 60:
        normalized["risk_category"] = "MEDIUM"
    elif score <= 80:
        normalized["risk_category"] = "HIGH"
    else:
        normalized["risk_category"] = "SEVERELY_HIGH"

    rec = str(normalized.get("recommendation", "")).strip().upper()
    category = normalized["risk_category"]
    if category == "LOW":
        normalized["recommendation"] = "APPROVE"
    elif category == "MEDIUM":
        normalized["recommendation"] = "APPROVE_WITH_CONDITIONS"
    elif category == "HIGH":
        normalized["recommendation"] = "REFER_TO_SENIOR"
    else:
        normalized["recommendation"] = "REFER_TO_SENIOR" if rec == "REFER_TO_SENIOR" else "DECLINE"

    normalized["risk_category"] = _normalize_risk_category(normalized.get("risk_category"))
    return normalized


def _contains_any(text: str, words) -> bool:
    return any(w in text for w in words)


def _is_plan_request(question: str) -> bool:
    q = str(question or "").lower()
    keywords = [
        "plan",
        "policy options",
        "recommend plan",
        "suggest plan",
        "best plan",
        "health plan",
        "which policy",
        "which plan",
        "coverage option",
        "product recommendation",
    ]
    return _contains_any(q, keywords)


def _is_risk_prediction_request(question: str) -> bool:
    q = str(question or "").lower()
    keywords = [
        "risk factor",
        "risk score",
        "my risk",
        "underwriting risk",
        "insurance risk",
        "how risky",
        "what will be my risk",
        "what is my risk",
    ]
    return _contains_any(q, keywords)


def _is_partial_risk_followup(question: str) -> bool:
    q = str(question or "").lower()
    keywords = [
        "just based on this",
        "based on this",
        "try to calculate",
        "can you calculate",
        "can you estimate",
        "rough estimate",
        "approximate risk",
    ]
    return _contains_any(q, keywords)


def _is_application_specific_request(question: str) -> bool:
    q = str(question or "").lower()
    keywords = [
        "my application",
        "this application",
        "uploaded application",
        "my case",
        "this case",
        "in the document",
        "from the form",
        "from the uploaded",
        "my submitted details",
        "for this applicant",
    ]
    return _contains_any(q, keywords)


def _extract_money_for_keyword(text: str, keyword: str):
    pattern = rf"{keyword}[^0-9]{{0,30}}([0-9][0-9,]*(?:\.[0-9]+)?)\s*(crore|cr|lakh|lac|l)?"
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    num = m.group(1)
    unit = m.group(2) or ""
    return _parse_rupee_value(f"{num} {unit}".strip())


def _build_profile_from_chat(question: str, chat_history: list | None = None):
    user_lines = []
    for item in (chat_history or []):
        if str(item.get("role", "")).lower() == "user":
            user_lines.append(str(item.get("content", "")))
    user_lines.append(str(question or ""))
    text = " ".join(user_lines).lower()

    profile = {}

    age_match = re.search(r"\bage\s*(?:is|:)?\s*(\d{1,3})\b", text) or re.search(r"\b(\d{1,3})\s*(?:years|yrs)\b", text)
    if age_match:
        profile["age"] = int(age_match.group(1))

    bmi_match = re.search(r"\bbmi\s*(?:is|:)?\s*(\d{1,2}(?:\.\d+)?)\b", text)
    if bmi_match:
        profile["bmi"] = float(bmi_match.group(1))

    if _contains_any(text, ["smoke daily", "current smoker", "i smoke", "smoker", "tobacco"]):
        profile["smoking"] = "current"
    elif _contains_any(text, ["former smoker", "quit smoking", "ex-smoker", "used to smoke"]):
        profile["smoking"] = "former"
    elif _contains_any(text, ["non smoker", "non-smoker", "never smoked", "do not smoke"]):
        profile["smoking"] = "non"

    if _contains_any(text, ["no pre-existing", "none pre-existing", "no medical condition", "healthy"]):
        profile["pre_existing"] = "none"
    elif _contains_any(text, ["mild", "allergy", "asthma mild", "thyroid mild"]):
        profile["pre_existing"] = "mild"
    elif _contains_any(text, ["hypertension", "diabetes", "moderate", "managed condition"]):
        profile["pre_existing"] = "moderate"
    elif _contains_any(text, ["cancer", "heart disease", "kidney failure", "severe", "stroke"]):
        profile["pre_existing"] = "severe"

    if re.search(r"occupation\s*(?:risk)?\s*(?:is|:)?\s*high", text):
        profile["occupation_risk"] = "high"
    elif re.search(r"occupation\s*(?:risk)?\s*(?:is|:)?\s*medium", text):
        profile["occupation_risk"] = "medium"
    elif re.search(r"occupation\s*(?:risk)?\s*(?:is|:)?\s*low", text):
        profile["occupation_risk"] = "low"
    elif _contains_any(text, ["construction", "mining", "military", "pilot", "hazardous occupation"]):
        profile["occupation_risk"] = "high"
    elif _contains_any(text, ["sales", "nurse", "delivery", "field job"]):
        profile["occupation_risk"] = "medium"
    elif _contains_any(text, ["office", "it", "teacher", "accountant", "desk job"]):
        profile["occupation_risk"] = "low"

    if _contains_any(text, ["no family history", "family history none"]):
        profile["family_history"] = "none"
    elif _contains_any(text, ["family history serious", "serious family history"]):
        profile["family_history"] = "serious"
    elif _contains_any(text, ["family history", "father had", "mother had", "diabetes in family", "bp in family"]):
        profile["family_history"] = "moderate"
    if _contains_any(text, ["family history of cancer", "heart attack in family", "stroke in family", "serious family history"]):
        profile["family_history"] = "serious"

    if re.search(r"lifestyle\s*(?:risk)?\s*(?:is|:)?\s*high", text):
        profile["lifestyle"] = "high"
    elif re.search(r"lifestyle\s*(?:risk)?\s*(?:is|:)?\s*moderate", text):
        profile["lifestyle"] = "moderate"
    elif re.search(r"lifestyle\s*(?:risk)?\s*(?:is|:)?\s*low", text):
        profile["lifestyle"] = "low"
    elif _contains_any(text, ["extreme sports", "adventure sports", "heavy alcohol", "racing"]):
        profile["lifestyle"] = "high"
    elif _contains_any(text, ["regular travel", "moderate sports", "social drinking"]):
        profile["lifestyle"] = "moderate"
    elif _contains_any(text, ["no alcohol", "sedentary", "low risk lifestyle", "regular walking"]):
        profile["lifestyle"] = "low"

    ratio_match = re.search(r"(\d+(?:\.\d+)?)\s*x\s*(?:annual\s*)?income", text)
    if ratio_match:
        profile["coverage_income_ratio"] = float(ratio_match.group(1))
    else:
        coverage_amount = (
            _extract_money_for_keyword(text, "coverage")
            or _extract_money_for_keyword(text, "sum assured")
            or _extract_money_for_keyword(text, "sum insured")
        )
        income_amount = _extract_money_for_keyword(text, "income") or _extract_money_for_keyword(text, "salary")
        if coverage_amount and income_amount and income_amount > 0:
            profile["coverage_income_ratio"] = round(coverage_amount / income_amount, 2)

    return profile


def _missing_profile_fields(profile: dict):
    required = [
        ("age", "Age"),
        ("bmi", "BMI"),
        ("smoking", "Smoking status (non/former/current)"),
        ("pre_existing", "Pre-existing condition severity (none/mild/moderate/severe)"),
        ("occupation_risk", "Occupation risk (low/medium/high)"),
        ("family_history", "Family history severity (none/moderate/serious)"),
        ("coverage_income_ratio", "Coverage-to-income ratio (e.g., 12x)"),
        ("lifestyle", "Lifestyle risk (low/moderate/high)"),
    ]
    missing = [label for key, label in required if key not in profile]
    return missing


def _calculate_risk_from_profile(profile: dict):
    age = int(profile["age"])
    bmi = float(profile["bmi"])
    smoking = str(profile["smoking"]).lower()
    pre_existing = str(profile["pre_existing"]).lower()
    occupation_risk = str(profile["occupation_risk"]).lower()
    family_history = str(profile["family_history"]).lower()
    ratio = float(profile["coverage_income_ratio"])
    lifestyle = str(profile["lifestyle"]).lower()

    factors = []

    age_score = 1 if 25 <= age <= 45 else 3 if (46 <= age <= 60 or 18 <= age <= 24) else 5
    factors.append(("Age", age_score, f"{age} years"))

    bmi_score = 1 if 18.5 <= bmi <= 24.9 else 3 if 25 <= bmi <= 29.9 else 5
    factors.append(("BMI", bmi_score, str(bmi)))

    smoking_score_map = {"non": 0, "former": 2, "current": 5}
    factors.append(("Smoking", smoking_score_map.get(smoking, 5), smoking))

    pre_existing_map = {"none": 0, "mild": 2, "moderate": 4, "severe": 6}
    factors.append(("Pre-existing Conditions", pre_existing_map.get(pre_existing, 4), pre_existing))

    occupation_map = {"low": 1, "medium": 3, "high": 5}
    factors.append(("Occupation Risk", occupation_map.get(occupation_risk, 3), occupation_risk))

    family_map = {"none": 0, "moderate": 2, "serious": 4}
    factors.append(("Family History", family_map.get(family_history, 2), family_history))

    coverage_score = 1 if ratio < 10 else 2 if ratio <= 15 else 3
    factors.append(("Coverage to Income Ratio", coverage_score, f"{ratio}x"))

    lifestyle_map = {"low": 0, "moderate": 1, "high": 4}
    factors.append(("Lifestyle", lifestyle_map.get(lifestyle, 1), lifestyle))

    total = sum(item[1] for item in factors)
    score_pct = round((total / 48) * 100)
    category = "LOW" if score_pct < 30 else "MEDIUM" if score_pct < 60 else "HIGH" if score_pct <= 80 else "SEVERELY_HIGH"
    recommendation = (
        "APPROVE"
        if category == "LOW"
        else "APPROVE_WITH_CONDITIONS"
        if category == "MEDIUM"
        else "REFER_TO_SENIOR"
        if category == "HIGH"
        else "DECLINE"
    )
    return {"factors": factors, "score_pct": score_pct, "category": category, "recommendation": recommendation}


def _score_category(score_pct: int) -> str:
    if score_pct < 30:
        return "LOW"
    if score_pct < 60:
        return "MEDIUM"
    if score_pct <= 80:
        return "HIGH"
    return "SEVERELY_HIGH"


def _calculate_partial_risk_from_profile(profile: dict):
    known = {}
    if "age" in profile:
        age = int(profile["age"])
        known["Age"] = 1 if 25 <= age <= 45 else 3 if (46 <= age <= 60 or 18 <= age <= 24) else 5
    if "bmi" in profile:
        bmi = float(profile["bmi"])
        known["BMI"] = 1 if 18.5 <= bmi <= 24.9 else 3 if 25 <= bmi <= 29.9 else 5
    if "smoking" in profile:
        known["Smoking"] = {"non": 0, "former": 2, "current": 5}.get(str(profile["smoking"]).lower(), 5)
    if "pre_existing" in profile:
        known["Pre-existing Conditions"] = {"none": 0, "mild": 2, "moderate": 4, "severe": 6}.get(
            str(profile["pre_existing"]).lower(), 4
        )
    if "occupation_risk" in profile:
        known["Occupation Risk"] = {"low": 1, "medium": 3, "high": 5}.get(str(profile["occupation_risk"]).lower(), 3)
    if "family_history" in profile:
        known["Family History"] = {"none": 0, "moderate": 2, "serious": 4}.get(str(profile["family_history"]).lower(), 2)
    if "coverage_income_ratio" in profile:
        ratio = float(profile["coverage_income_ratio"])
        known["Coverage to Income Ratio"] = 1 if ratio < 10 else 2 if ratio <= 15 else 3
    if "lifestyle" in profile:
        known["Lifestyle"] = {"low": 0, "moderate": 1, "high": 4}.get(str(profile["lifestyle"]).lower(), 1)

    bounds = {
        "Age": (1, 5, 3),
        "BMI": (1, 5, 3),
        "Smoking": (0, 5, 3),
        "Pre-existing Conditions": (0, 6, 3),
        "Occupation Risk": (1, 5, 3),
        "Family History": (0, 4, 2),
        "Coverage to Income Ratio": (1, 3, 2),
        "Lifestyle": (0, 4, 1),
    }

    min_total = 0
    max_total = 0
    mid_total = 0
    for factor, (vmin, vmax, vmid) in bounds.items():
        if factor in known:
            s = known[factor]
            min_total += s
            max_total += s
            mid_total += s
        else:
            min_total += vmin
            max_total += vmax
            mid_total += vmid

    min_score = round((min_total / 48) * 100)
    max_score = round((max_total / 48) * 100)
    estimate_score = round((mid_total / 48) * 100)

    return {
        "known_factor_scores": known,
        "estimate_score": estimate_score,
        "min_score": min_score,
        "max_score": max_score,
        "estimate_category": _score_category(estimate_score),
        "min_category": _score_category(min_score),
        "max_category": _score_category(max_score),
    }


def _enrich_profile_from_assessment(profile: dict, assessment: dict):
    """Fill missing profile fields from assessed summary/factors when available."""
    enriched = dict(profile or {})
    data = assessment or {}
    summary = data.get("applicant_summary", {}) if isinstance(data, dict) else {}
    factors = data.get("risk_factors", []) if isinstance(data, dict) else []

    if "age" not in enriched:
        try:
            enriched["age"] = int(summary.get("age"))
        except Exception:
            pass

    factor_map = {str(item.get("factor", "")).strip().lower(): str(item.get("value", "")).lower() for item in factors}

    if "smoking" not in enriched:
        smoking_val = factor_map.get("smoking", "")
        if "current" in smoking_val:
            enriched["smoking"] = "current"
        elif "former" in smoking_val:
            enriched["smoking"] = "former"
        elif "non" in smoking_val:
            enriched["smoking"] = "non"

    if "pre_existing" not in enriched:
        ped = factor_map.get("pre-existing conditions", "")
        if "none" in ped:
            enriched["pre_existing"] = "none"
        elif "mild" in ped:
            enriched["pre_existing"] = "mild"
        elif "severe" in ped:
            enriched["pre_existing"] = "severe"
        elif ped:
            enriched["pre_existing"] = "moderate"

    if "bmi" not in enriched:
        bmi_txt = factor_map.get("bmi", "")
        bmi_match = re.search(r"\d{1,2}(?:\.\d+)?", bmi_txt)
        if bmi_match:
            enriched["bmi"] = float(bmi_match.group(0))

    if "coverage_income_ratio" not in enriched:
        ratio = _coverage_income_ratio(summary)
        if ratio is not None:
            enriched["coverage_income_ratio"] = round(float(ratio), 2)

    return enriched


def _build_personalized_plan_response(
    question: str,
    assessment: dict,
    history: list | None,
    application_text: str,
    tnc_text: str,
    report: str,
) -> str:
    use_case_context = _is_application_specific_request(question)
    profile = _build_profile_from_chat(question, chat_history=history)
    if use_case_context:
        profile = _enrich_profile_from_assessment(profile, assessment)

    missing = _missing_profile_fields(profile)
    risk_computed = None
    if not missing:
        risk_computed = _calculate_risk_from_profile(profile)

    if risk_computed:
        risk_score = risk_computed["score_pct"]
    elif use_case_context:
        try:
            risk_score = int((assessment or {}).get("overall_risk_score"))
        except Exception:
            risk_score = None
    else:
        risk_score = None

    plans = recommend_health_plans(
        assessment=assessment,
        max_plans=3,
        profile=profile,
        risk_score=risk_score,
    )

    risk_summary = ""
    if risk_computed:
        risk_summary = (
            f"Inferred risk from provided details: {risk_computed['score_pct']}/100 "
            f"({risk_computed['category']}), recommendation: {risk_computed['recommendation']}."
        )

    if is_api_configured():
        try:
            llm = _create_llm(MODEL_FAST, temperature=0.2, max_tokens=1000)
            if llm:
                context_payload = {
                    "use_case_context": use_case_context,
                    "profile": profile,
                    "missing_details": missing,
                    "risk_computed": risk_computed,
                    "assessment": (assessment or {}) if use_case_context else {},
                    "report": str(report or "")[:3000] if use_case_context else "",
                    "application_excerpt": _trim_application_text(application_text) if use_case_context else "",
                    "tnc_excerpt": _trim_tnc_text(tnc_text),
                    "plan_catalog": get_health_plan_catalog(),
                    "shortlisted_plans": plans,
                    "user_question": question,
                }
                messages = [
                    {"role": "system", "content": PLAN_RECOMMENDATION_PROMPT},
                    {"role": "user", "content": json.dumps(context_payload, ensure_ascii=False)},
                ]
                response = llm.invoke(messages)
                text = str(response.content).strip()
                if text:
                    return text
        except Exception:
            pass

    base = format_plan_recommendations(plans, risk_summary=risk_summary)
    if missing:
        base += "\n\nTo personalize further, share: " + ", ".join(missing) + "."
    return base


def _parse_rupee_value(value: str):
    text = str(value or "")
    if not text:
        return None

    num_match = re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))
    if not num_match:
        return None
    amount = float(num_match[0])

    lower = text.lower()
    if "crore" in lower or " cr" in lower:
        amount *= 10_000_000
    elif "lakh" in lower or " lac" in lower or " l" in lower:
        amount *= 100_000

    if amount < 1000 and ("000" in text or "00" in text):
        maybe = re.sub(r"[^0-9.]", "", text)
        try:
            amount = float(maybe)
        except Exception:
            pass

    return amount


def _coverage_income_ratio(summary: dict):
    coverage = _parse_rupee_value(summary.get("coverage_amount", ""))
    income = _parse_rupee_value(summary.get("income", ""))
    if not coverage or not income or income <= 0:
        return None
    return coverage / income


def _merge_compliance_flags(existing_flags, extra_flags):
    merged = []
    seen = set()
    for item in (existing_flags or []) + (extra_flags or []):
        rid = str(item.get("rule_id", "")).strip().upper()
        if not rid:
            continue
        if rid in seen:
            continue
        seen.add(rid)
        merged.append(item)
    return merged


def _run_additional_compliance_checks(application_text: str, assessment: dict, tnc_text: str = ""):
    text = str(application_text or "").lower()
    tnc = str(tnc_text or "").lower()
    summary = assessment.get("applicant_summary", {}) if isinstance(assessment, dict) else {}
    score = int(assessment.get("overall_risk_score", 0) or 0) if isinstance(assessment, dict) else 0

    checks = []

    has_nominee = _contains_any(text, ["nominee", "beneficiary", "appointee"])
    checks.append({
        "rule_id": "IRDAI-05",
        "rule": "Nominee or beneficiary details should be present in the proposal",
        "status": "PASS" if has_nominee else "FLAG",
        "note": "Nominee/beneficiary details found." if has_nominee else "Nominee/beneficiary details not explicitly found.",
    })

    has_terms_ack = _contains_any(text, ["terms and conditions", "i agree", "declaration", "exclusions"])
    checks.append({
        "rule_id": "IRDAI-06",
        "rule": "Policy exclusions and terms acknowledgement should be captured",
        "status": "PASS" if has_terms_ack else "FLAG",
        "note": "Terms acknowledgement found in application." if has_terms_ack else "No explicit terms/exclusions acknowledgement found.",
    })

    has_income_proof = _contains_any(text, ["itr", "form 16", "salary slip", "bank statement", "income proof"]) \
        or _contains_any(str(summary.get("income", "")).lower(), ["rs", "inr", "lakh", "crore"])
    checks.append({
        "rule_id": "AML-03",
        "rule": "Source of funds / income proof should be documented for high-value policies",
        "status": "PASS" if has_income_proof else "FLAG",
        "note": "Income/source of funds indicators found." if has_income_proof else "No explicit income-proof artifacts found in text.",
    })

    ratio = _coverage_income_ratio(summary)
    high_value_or_high_risk = (ratio is not None and ratio > 12) or score >= 60
    checks.append({
        "rule_id": "AML-04",
        "rule": "Sanctions and adverse media screening should be completed before issuance",
        "status": "FLAG" if high_value_or_high_risk else "PASS",
        "note": "Enhanced AML screening recommended due to risk profile." if high_value_or_high_risk else "Standard screening appears sufficient.",
    })

    has_consent = _contains_any(text, ["consent", "authorize", "authorise", "agree to process", "dpdp"]) or _contains_any(
        tnc, ["consent", "data processing", "personal data"]
    )
    checks.append({
        "rule_id": "DATA-02",
        "rule": "Explicit applicant consent for data processing should be captured",
        "status": "PASS" if has_consent else "FLAG",
        "note": "Consent language identified." if has_consent else "No explicit data-processing consent statement found.",
    })

    factors = assessment.get("risk_factors", []) if isinstance(assessment, dict) else []
    pre_existing = ""
    for factor in factors:
        if str(factor.get("factor", "")).strip().lower() == "pre-existing conditions":
            pre_existing = str(factor.get("value", "")).lower()
            break
    meds_present = _contains_any(text, ["medication", "metformin", "insulin", "amlodipine", "tablet", "mg"]) \
        and not _contains_any(text, ["current medications: none", "medications: none"])
    contradiction = ("none" in pre_existing and meds_present)
    checks.append({
        "rule_id": "UW-01",
        "rule": "Medical declaration should be internally consistent with medication/disclosure details",
        "status": "FLAG" if contradiction else "PASS",
        "note": "Potential mismatch between declared conditions and medication details." if contradiction else "No clear medical declaration contradiction detected.",
    })

    has_history_fields = _contains_any(text, ["previous claims", "policy rejections", "previous policy rejections"])
    checks.append({
        "rule_id": "UW-02",
        "rule": "Prior claims or rejections must be explicitly disclosed for underwriting review",
        "status": "PASS" if has_history_fields else "FLAG",
        "note": "Claims/rejections disclosure section present." if has_history_fields else "Prior claims/rejections disclosure not clearly present.",
    })

    if tnc:
        free_look_present = _contains_any(tnc, ["free-look", "free look", "15 days", "30 days"])
        checks.append({
            "rule_id": "TNC-01",
            "rule": "T&C document should mention free-look provisions",
            "status": "PASS" if free_look_present else "FLAG",
            "note": "Free-look clause found in provided T&C." if free_look_present else "Free-look clause not detected in provided T&C excerpt.",
        })

    return checks


# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------
def assess_risk(application_text: str, tnc_text: str = "") -> dict:
    """Main function: Analyze insurance application and return structured risk assessment."""
    if not is_api_configured():
        return {"error": "API key not configured. Add your key in the sidebar or .env file."}

    text = ""
    try:
        llm = _create_llm(MODEL, temperature=0.1, max_tokens=1600)
        if not llm:
            return {"error": "Could not initialize LLM."}

        prepared_text = _trim_application_text(application_text)
        messages = [
            {"role": "system", "content": RISK_ASSESSMENT_PROMPT},
            {"role": "user", "content": f"Analyze this insurance application:\n\n{prepared_text}"},
        ]
        response = llm.invoke(messages)
        text = response.content.strip().replace("```json", "").replace("```", "").strip()

        result = json.loads(text)
        normalized = _normalize_assessment(result)
        extra_checks = _run_additional_compliance_checks(application_text, normalized, tnc_text=tnc_text)
        normalized["compliance_flags"] = _merge_compliance_flags(normalized.get("compliance_flags", []), extra_checks)
        return normalized

    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse AI response as JSON: {e}", "raw_response": text}
    except Exception as e:
        err = str(e)
        if "401" in err:
            return {"error": "Authentication failed. Check your API key."}
        if "connect" in err.lower():
            return {"error": "Cannot reach genailab.tcs.in. Check VPN/network."}
        return {"error": f"Assessment failed: {err}"}


def generate_report(assessment: dict) -> str:
    """Generate a human-readable markdown report from structured assessment."""
    if "error" in assessment:
        return f"[ERROR] {assessment['error']}"

    if not is_api_configured():
        return _generate_report_local(assessment)

    try:
        llm = _create_llm(MODEL_FAST, temperature=0.3, max_tokens=2000)
        if not llm:
            return _generate_report_local(assessment)

        messages = [
            {"role": "system", "content": EXPLANATION_PROMPT},
            {"role": "user", "content": json.dumps(assessment, indent=2)},
        ]
        response = llm.invoke(messages)
        return response.content

    except Exception:
        return _generate_report_local(assessment)


def chat_with_underwriter_assistant(
    question: str,
    application_text: str = "",
    assessment: dict | None = None,
    report: str = "",
    tnc_text: str = "",
    chat_history: list | None = None,
) -> str:
    """Chatbot helper for underwriting Q&A with application, assessment and T&C context."""
    question = str(question or "").strip()
    if not question:
        return "Please enter a question."

    assessment = assessment or {}
    history = chat_history or []
    compact_history = history[-8:]
    profile = _build_profile_from_chat(question, chat_history=history)
    risk_intent = _is_risk_prediction_request(question) or (_is_partial_risk_followup(question) and len(profile) >= 2)
    if risk_intent:
        missing = _missing_profile_fields(profile)
        if missing:
            partial = _calculate_partial_risk_from_profile(profile)
            captured = []
            for key in ["age", "bmi", "smoking", "pre_existing", "occupation_risk", "family_history", "coverage_income_ratio", "lifestyle"]:
                if key in profile:
                    captured.append(f"{key}={profile[key]}")
            captured_text = ", ".join(captured) if captured else "none yet"
            lines = [
                "Provisional risk estimate (partial inputs):",
                f"- Estimated Score: {partial['estimate_score']}/100 ({partial['estimate_category']})",
                f"- Possible Range: {partial['min_score']} to {partial['max_score']} "
                f"({partial['min_category']} to {partial['max_category']})",
                f"- Captured so far: {captured_text}",
                f"- Missing for final score: {', '.join(missing)}",
                "Share missing details to get a final factor-by-factor score.",
            ]
            return "\n".join(lines)
        computed = _calculate_risk_from_profile(profile)
        lines = ["Risk estimate using scoring guide:"]
        for factor_name, factor_score, factor_value in computed["factors"]:
            lines.append(f"- {factor_name}: {factor_value} -> {factor_score}/6")
        lines.append(
            f"Total Risk Score: {computed['score_pct']}/100 | Category: {computed['category']} | "
            f"Recommendation: {computed['recommendation']}"
        )
        lines.append("This is a preliminary estimate from self-reported details and should be validated by underwriting.")
        return "\n".join(lines)

    if _is_plan_request(question):
        return _build_personalized_plan_response(
            question=question,
            assessment=assessment,
            history=history,
            application_text=application_text,
            tnc_text=tnc_text,
            report=report,
        )

    if is_api_configured():
        try:
            llm = _create_llm(MODEL_FAST, temperature=0.2, max_tokens=1200)
            if llm:
                messages = [{"role": "system", "content": CHATBOT_PROMPT}]

                for item in compact_history:
                    role = str(item.get("role", "")).strip().lower()
                    content = str(item.get("content", "")).strip()
                    if role in {"user", "assistant"} and content:
                        messages.append({"role": role, "content": content})

                use_case_context = _is_application_specific_request(question)
                context_payload = {
                    "use_case_context": use_case_context,
                    "tnc_excerpt": _trim_tnc_text(tnc_text),
                    "plan_catalog": get_health_plan_catalog(),
                }
                if use_case_context:
                    context_payload.update(
                        {
                            "assessment": assessment,
                            "report": str(report or "")[:4000],
                            "application_excerpt": _trim_application_text(application_text),
                        }
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": "Context JSON:\n"
                        + json.dumps(context_payload, ensure_ascii=False)
                        + f"\n\nQuestion: {question}",
                    }
                )
                response = llm.invoke(messages)
                return str(response.content).strip()
        except Exception:
            pass

    return _chat_fallback_local(question, assessment, chat_history=history)


def _chat_fallback_local(question: str, assessment: dict, chat_history: list | None = None) -> str:
    q = question.lower()
    score = assessment.get("overall_risk_score", "N/A")
    category = assessment.get("risk_category", "N/A")
    recommendation = assessment.get("recommendation", "N/A")

    if "score" in q or "risk" in q:
        return f"Current preliminary risk is {score} ({category}) with recommendation {recommendation}."

    if "compliance" in q or "regulatory" in q:
        flags = assessment.get("compliance_flags", [])
        if not flags:
            return "No compliance flags were generated yet. Analyze an application first."
        top = [f"{c.get('rule_id', 'NA')}: {c.get('status', 'NA')}" for c in flags[:8]]
        return "Compliance snapshot: " + "; ".join(top)

    if "concern" in q or "red flag" in q:
        concerns = assessment.get("key_concerns", [])
        if not concerns:
            return "No key concerns were extracted."
        return "Top concerns: " + "; ".join(concerns[:5])

    if _is_plan_request(question):
        profile = _build_profile_from_chat(question, chat_history=chat_history)
        if _is_application_specific_request(question):
            profile = _enrich_profile_from_assessment(profile, assessment)
        missing = _missing_profile_fields(profile)
        risk = _calculate_risk_from_profile(profile) if not missing else None
        plans = recommend_health_plans(
            assessment=assessment,
            max_plans=3,
            profile=profile,
            risk_score=(risk or {}).get("score_pct"),
        )
        summary = ""
        if risk:
            summary = (
                f"Inferred risk from provided details: {risk['score_pct']}/100 "
                f"({risk['category']}), recommendation: {risk['recommendation']}."
            )
        text = format_plan_recommendations(plans, risk_summary=summary)
        if missing:
            text += "\n\nTo personalize further, share: " + ", ".join(missing) + "."
        return text

    return (
        "I can help with health insurance risk score, PED/waiting-period implications, recommendation rationale, "
        "compliance checks, and plan suggestions."
    )


def _generate_report_local(assessment: dict) -> str:
    """Fallback: Generate report without LLM."""
    summary = assessment.get("applicant_summary", {})
    factors = assessment.get("risk_factors", [])
    score = assessment.get("overall_risk_score", 0)
    category = assessment.get("risk_category", "UNKNOWN")
    recommendation = assessment.get("recommendation", "N/A")
    concerns = assessment.get("key_concerns", [])
    actions = assessment.get("suggested_actions", [])
    compliance = assessment.get("compliance_flags", [])

    icons = {"LOW": "[LOW]", "MEDIUM": "[MED]", "HIGH": "[HIGH]", "SEVERELY_HIGH": "[SEV]"}

    report = f"""## Applicant Overview
**{summary.get('name', 'N/A')}**, {summary.get('age', 'N/A')} years, {summary.get('gender', 'N/A')}
**Occupation:** {summary.get('occupation', 'N/A')} | **City:** {summary.get('city', 'N/A')}
**Policy:** {summary.get('policy_type', 'N/A')} | **Coverage:** {summary.get('coverage_amount', 'N/A')}

## Risk Factors
"""
    for item in factors:
        icon = icons.get(item.get("risk_level", ""), "[?]")
        report += (
            f"- {icon} **{item.get('factor', '')}**: {item.get('value', '')} - "
            f"Score: {item.get('score', 0)}/6 - {item.get('explanation', '')}\n"
        )

    report += f"""
## Risk Score and Recommendation
**Total Score:** {score} | **Category:** {category} | **Recommendation:** {recommendation}

## Key Concerns
"""
    for concern in concerns:
        report += f"- {concern}\n"

    report += "\n## Compliance\n"
    for flag in compliance:
        status_icon = {"PASS": "[PASS]", "FLAG": "[FLAG]", "FAIL": "[FAIL]"}.get(flag.get("status", ""), "[?]")
        report += f"- {status_icon} **{flag.get('rule_id', '')}**: {flag.get('note', flag.get('rule', ''))}\n"

    report += "\n## Next Steps\n"
    for action in actions:
        report += f"- {action}\n"

    return report
