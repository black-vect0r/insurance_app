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


CHATBOT_PROMPT = """You are an Insurance Underwriting Chatbot for underwriters.
Answer only using:
1) Provided application details,
2) Structured risk assessment,
3) Provided Terms and Conditions excerpt.

Rules:
- Be concise and practical.
- If evidence is missing, say "Not found in provided documents".
- Mention relevant compliance rule IDs where possible.
- Do not provide legal advice; provide underwriting guidance.
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

                context_payload = {
                    "assessment": assessment,
                    "report": str(report or "")[:4000],
                    "application_excerpt": _trim_application_text(application_text),
                    "tnc_excerpt": _trim_tnc_text(tnc_text),
                }
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

    return _chat_fallback_local(question, assessment)


def _chat_fallback_local(question: str, assessment: dict) -> str:
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

    return (
        "I can help with risk score, factor explanations, recommendation rationale, and compliance checks. "
        "Analyze an application first for detailed case-specific answers."
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
