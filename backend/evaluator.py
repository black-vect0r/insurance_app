"""
ðŸ“Š Accuracy Evaluation Module
Measures AI risk factor extraction accuracy against human-labeled ground truth.

Methodology:
  1. Define ground truth for each sample application (manually labeled)
  2. Run AI assessment on each
  3. Compare extracted factors vs ground truth
  4. Calculate accuracy per factor and overall
  5. Generate eval report

Success Criteria: â‰¥80% accuracy in risk factor extraction
"""

import json
from datetime import datetime


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# GROUND TRUTH â€” Manually labeled
# (This is what a human underwriter would identify)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
GROUND_TRUTH = [
    {
        "applicant": "Rajesh Kumar Sharma",
        "expected_factors": {
            "Age":                   {"value": "42", "risk_level": "LOW",    "score_range": [1, 2]},
            "BMI":                   {"value": "24.2", "risk_level": "LOW",  "score_range": [1, 2]},
            "Smoking":               {"value": "Non-smoker", "risk_level": "LOW", "score_range": [0, 1]},
            "Pre-existing Conditions": {"value": "None", "risk_level": "LOW", "score_range": [0, 1]},
            "Occupation":            {"value": "Software Engineer", "risk_level": "LOW", "score_range": [1, 2]},
            "Family History":        {"value": "Father diabetes", "risk_level": "MEDIUM", "score_range": [1, 3]},
            "Coverage Amount":       {"value": "1 Crore / 5.4x income", "risk_level": "LOW", "score_range": [1, 2]},
            "Lifestyle":             {"value": "Regular exercise", "risk_level": "LOW", "score_range": [0, 1]},
        },
        "expected_category": "LOW",
        "expected_recommendation": "APPROVE",
        "expected_score_range": [8, 25],
    },
    {
        "applicant": "Priya Venkatesh",
        "expected_factors": {
            "Age":                   {"value": "55", "risk_level": "MEDIUM", "score_range": [2, 4]},
            "BMI":                   {"value": "28.5", "risk_level": "MEDIUM", "score_range": [2, 4]},
            "Smoking":               {"value": "Former smoker", "risk_level": "MEDIUM", "score_range": [1, 3]},
            "Pre-existing Conditions": {"value": "Hypertension, thyroid", "risk_level": "MEDIUM", "score_range": [3, 5]},
            "Occupation":            {"value": "Bank Manager", "risk_level": "LOW", "score_range": [1, 2]},
            "Family History":        {"value": "Mother cancer, father heart", "risk_level": "HIGH", "score_range": [3, 5]},
            "Coverage Amount":       {"value": "50L / 1.5x income", "risk_level": "LOW", "score_range": [1, 2]},
            "Lifestyle":             {"value": "Normal", "risk_level": "LOW", "score_range": [0, 1]},
        },
        "expected_category": "MEDIUM",
        "expected_recommendation": "APPROVE_WITH_CONDITIONS",
        "expected_score_range": [30, 50],
    },
    {
        "applicant": "Mohammad Arif Khan",
        "expected_factors": {
            "Age":                   {"value": "29", "risk_level": "LOW",    "score_range": [1, 2]},
            "BMI":                   {"value": "31.2", "risk_level": "HIGH", "score_range": [4, 6]},
            "Smoking":               {"value": "Current smoker", "risk_level": "HIGH", "score_range": [4, 6]},
            "Pre-existing Conditions": {"value": "None diagnosed", "risk_level": "LOW", "score_range": [0, 2]},
            "Occupation":            {"value": "Construction", "risk_level": "HIGH", "score_range": [4, 6]},
            "Family History":        {"value": "Father heart attack, uncle cancer", "risk_level": "HIGH", "score_range": [3, 5]},
            "Coverage Amount":       {"value": "75L / 11.5x income", "risk_level": "MEDIUM", "score_range": [2, 3]},
            "Lifestyle":             {"value": "Motorcycle racing", "risk_level": "HIGH", "score_range": [3, 5]},
        },
        "expected_category": "HIGH",
        "expected_recommendation": "REFER_TO_SENIOR",
        "expected_score_range": [55, 80],
    },
    {
        "applicant": "Anita Desai",
        "expected_factors": {
            "Age":                   {"value": "35", "risk_level": "LOW",    "score_range": [1, 2]},
            "BMI":                   {"value": "22.1", "risk_level": "LOW",  "score_range": [0, 2]},
            "Smoking":               {"value": "Non-smoker", "risk_level": "LOW", "score_range": [0, 1]},
            "Pre-existing Conditions": {"value": "None", "risk_level": "LOW", "score_range": [0, 1]},
            "Occupation":            {"value": "Chartered Accountant", "risk_level": "LOW", "score_range": [1, 2]},
            "Family History":        {"value": "None significant", "risk_level": "LOW", "score_range": [0, 1]},
            "Coverage Amount":       {"value": "2Cr / 8.3x income", "risk_level": "LOW", "score_range": [1, 2]},
            "Lifestyle":             {"value": "Yoga, swimming", "risk_level": "LOW", "score_range": [0, 1]},
        },
        "expected_category": "LOW",
        "expected_recommendation": "APPROVE",
        "expected_score_range": [6, 21],
    },
    {
        "applicant": "Vikram Singh Rathore",
        "expected_factors": {
            "Age":                   {"value": "63", "risk_level": "HIGH",   "score_range": [4, 6]},
            "BMI":                   {"value": "26.8", "risk_level": "MEDIUM","score_range": [2, 4]},
            "Smoking":               {"value": "Former smoker 15yr ago", "risk_level": "LOW", "score_range": [0, 2]},
            "Pre-existing Conditions": {"value": "Diabetes, arthritis, kidney stones", "risk_level": "HIGH", "score_range": [4, 6]},
            "Occupation":            {"value": "Retired military", "risk_level": "MEDIUM", "score_range": [1, 3]},
            "Family History":        {"value": "Father stroke, mother diabetes", "risk_level": "HIGH", "score_range": [3, 5]},
            "Coverage Amount":       {"value": "25L / 2x income", "risk_level": "LOW", "score_range": [1, 2]},
            "Lifestyle":             {"value": "No extreme activities", "risk_level": "LOW", "score_range": [0, 1]},
        },
        "expected_category": "MEDIUM",
        "expected_recommendation": "APPROVE_WITH_CONDITIONS",
        "expected_score_range": [35, 59],
    },
]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Factor name normalization
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
FACTOR_ALIASES = {
    "age":                    "Age",
    "bmi":                    "BMI",
    "body mass index":        "BMI",
    "smoking":                "Smoking",
    "smoking status":         "Smoking",
    "tobacco":                "Smoking",
    "pre-existing":           "Pre-existing Conditions",
    "pre-existing conditions":"Pre-existing Conditions",
    "medical history":        "Pre-existing Conditions",
    "pre existing":           "Pre-existing Conditions",
    "occupation":             "Occupation",
    "occupation risk":        "Occupation",
    "job risk":               "Occupation",
    "family history":         "Family History",
    "family medical history": "Family History",
    "coverage":               "Coverage Amount",
    "coverage amount":        "Coverage Amount",
    "sum assured":            "Coverage Amount",
    "coverage ratio":         "Coverage Amount",
    "lifestyle":              "Lifestyle",
    "lifestyle risk":         "Lifestyle",
    "activities":             "Lifestyle",
    "dangerous activities":   "Lifestyle",
}


def normalize_factor_name(name: str) -> str:
    """Map AI's factor name to standard name."""
    key = name.lower().strip()
    return FACTOR_ALIASES.get(key, name)


def normalize_category(name: str) -> str:
    """Normalize category labels to LOW/MEDIUM/HIGH/SEVERELY_HIGH."""
    cat = str(name or "").strip().upper()
    cat = cat.replace(" ", "_")
    if cat in {"LOW", "MEDIUM", "HIGH", "SEVERELY_HIGH"}:
        return cat
    if cat in {"MODERATE", "MID"}:
        return "MEDIUM"
    if cat in {"VERY_HIGH", "SEVERE_HIGH", "SEVERLY_HIGH"}:
        return "SEVERELY_HIGH"
    return cat


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Evaluation Logic
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def evaluate_single(ai_assessment: dict, ground_truth: dict) -> dict:
    """
    Compare one AI assessment against ground truth.
    Returns detailed accuracy metrics.
    """
    expected = ground_truth["expected_factors"]
    ai_factors = ai_assessment.get("risk_factors", [])

    # Build lookup of AI factors by normalized name
    ai_lookup = {}
    for f in ai_factors:
        normalized = normalize_factor_name(f.get("factor", ""))
        ai_lookup[normalized] = f

    results = []
    total_checks = 0
    passed_checks = 0

    for factor_name, truth in expected.items():
        ai_factor = ai_lookup.get(factor_name, None)

        check = {
            "factor": factor_name,
            "expected_risk_level": truth["risk_level"],
            "expected_score_range": truth["score_range"],
            "detected": ai_factor is not None,
            "risk_level_match": False,
            "score_in_range": False,
            "ai_risk_level": None,
            "ai_score": None,
        }

        # Check 1: Was the factor detected?
        total_checks += 1
        if ai_factor:
            passed_checks += 1
            check["ai_risk_level"] = ai_factor.get("risk_level", "").upper()
            check["ai_score"] = ai_factor.get("score", 0)

            # Check 2: Risk level match
            total_checks += 1
            if check["ai_risk_level"] == truth["risk_level"]:
                check["risk_level_match"] = True
                passed_checks += 1

            # Check 3: Score within acceptable range
            total_checks += 1
            score = check["ai_score"]
            lo, hi = truth["score_range"]
            if lo <= score <= hi:
                check["score_in_range"] = True
                passed_checks += 1

        results.append(check)

    # Category match
    total_checks += 1
    ai_category = normalize_category(ai_assessment.get("risk_category", ""))
    expected_category = normalize_category(ground_truth["expected_category"])
    category_match = ai_category == expected_category
    if category_match:
        passed_checks += 1

    # Overall score in range
    total_checks += 1
    ai_score = ai_assessment.get("overall_risk_score", 0)
    lo, hi = ground_truth["expected_score_range"]
    score_in_range = lo <= ai_score <= hi
    if score_in_range:
        passed_checks += 1

    # Recommendation match
    total_checks += 1
    ai_rec = ai_assessment.get("recommendation", "").upper()
    expected_rec = ground_truth["expected_recommendation"]
    rec_match = ai_rec == expected_rec
    if rec_match:
        passed_checks += 1

    accuracy = round((passed_checks / total_checks) * 100, 1) if total_checks > 0 else 0

    return {
        "applicant": ground_truth["applicant"],
        "factor_results": results,
        "factors_detected": sum(1 for r in results if r["detected"]),
        "factors_expected": len(expected),
        "risk_levels_correct": sum(1 for r in results if r["risk_level_match"]),
        "scores_in_range": sum(1 for r in results if r["score_in_range"]),
        "category_match": category_match,
        "ai_category": ai_category,
        "expected_category": expected_category,
        "score_in_range": score_in_range,
        "ai_score": ai_score,
        "expected_score_range": ground_truth["expected_score_range"],
        "recommendation_match": rec_match,
        "ai_recommendation": ai_rec,
        "expected_recommendation": expected_rec,
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "accuracy": accuracy,
    }


def run_full_evaluation(assessments: list, ground_truth: list | None = None) -> dict:
    """
    Run evaluation across provided sample applications.
    
    Args:
        assessments: list of AI assessment dicts (one per sample)
        ground_truth: optional list of ground-truth entries to evaluate against
    
    Returns:
        Full evaluation report with per-applicant and overall accuracy
    """
    truth_set = ground_truth if ground_truth is not None else GROUND_TRUTH
    if len(assessments) != len(truth_set):
        return {"error": f"Expected {len(truth_set)} assessments, got {len(assessments)}"}

    eval_results = []
    total_passed = 0
    total_checks = 0

    for i, (ai_result, truth) in enumerate(zip(assessments, truth_set)):
        if "error" in ai_result:
            eval_results.append({"applicant": truth["applicant"], "error": ai_result["error"], "accuracy": 0})
            continue

        result = evaluate_single(ai_result, truth)
        eval_results.append(result)
        total_passed += result["passed_checks"]
        total_checks += result["total_checks"]

    overall_accuracy = round((total_passed / total_checks) * 100, 1) if total_checks > 0 else 0
    meets_target = overall_accuracy >= 80.0

    # Per-factor accuracy across all applications
    factor_accuracy = {}
    for factor_name in ["Age", "BMI", "Smoking", "Pre-existing Conditions",
                         "Occupation", "Family History", "Coverage Amount", "Lifestyle"]:
        detected = 0
        level_correct = 0
        score_correct = 0
        count = 0
        for r in eval_results:
            if "error" in r:
                continue
            for fr in r.get("factor_results", []):
                if fr["factor"] == factor_name:
                    count += 1
                    if fr["detected"]:
                        detected += 1
                    if fr["risk_level_match"]:
                        level_correct += 1
                    if fr["score_in_range"]:
                        score_correct += 1

        factor_accuracy[factor_name] = {
            "detection_rate": round((detected / count) * 100, 1) if count > 0 else 0,
            "level_accuracy": round((level_correct / count) * 100, 1) if count > 0 else 0,
            "score_accuracy": round((score_correct / count) * 100, 1) if count > 0 else 0,
        }

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall_accuracy": overall_accuracy,
        "meets_80_target": meets_target,
        "total_checks": total_checks,
        "total_passed": total_passed,
        "per_applicant": eval_results,
        "per_factor": factor_accuracy,
        "summary": {
            "applicants_evaluated": len(eval_results),
            "category_matches": sum(1 for r in eval_results if r.get("category_match", False)),
            "recommendation_matches": sum(1 for r in eval_results if r.get("recommendation_match", False)),
        }
    }


def format_eval_report(eval_result: dict) -> str:
    """Format evaluation result as a readable markdown report."""
    if "error" in eval_result:
        return f"âš ï¸ {eval_result['error']}"

    overall = eval_result["overall_accuracy"]
    meets = eval_result["meets_80_target"]
    badge = "PASS" if meets else "FAIL"
    apps_evaluated = int(eval_result.get("summary", {}).get("applicants_evaluated", 0))
    report = f"""# ðŸ“Š Risk Factor Extraction â€” Accuracy Report
**Date:** {eval_result['timestamp']}
**Target:** â‰¥80% accuracy | **Result:** {overall}% | **Status:** {badge}

---

## Overall Results
- **Accuracy:** {overall}% ({eval_result['total_passed']}/{eval_result['total_checks']} checks passed)
- **Target Met:** {'Yes âœ…' if meets else 'No âŒ'}
- **Applications Evaluated:** {eval_result['summary']['applicants_evaluated']}
- **Category Matches:** {eval_result['summary']['category_matches']}/{apps_evaluated}
- **Recommendation Matches:** {eval_result['summary']['recommendation_matches']}/{apps_evaluated}

---

## Per-Applicant Results
"""
    for r in eval_result["per_applicant"]:
        if "error" in r:
            report += f"\n### âŒ {r['applicant']}\nError: {r['error']}\n"
            continue

        icon = "âœ…" if r["accuracy"] >= 80 else "âš ï¸" if r["accuracy"] >= 60 else "âŒ"
        report += f"""
### {icon} {r['applicant']} â€” {r['accuracy']}%
- Factors Detected: {r['factors_detected']}/{r['factors_expected']}
- Risk Levels Correct: {r['risk_levels_correct']}/{r['factors_expected']}
- Scores in Range: {r['scores_in_range']}/{r['factors_expected']}
- Category: AI={r['ai_category']} vs Expected={r['expected_category']} {'âœ…' if r['category_match'] else 'âŒ'}
- Score: AI={r['ai_score']} vs Expected Range={r['expected_score_range']} {'âœ…' if r['score_in_range'] else 'âŒ'}
- Recommendation: AI={r['ai_recommendation']} vs Expected={r['expected_recommendation']} {'âœ…' if r['recommendation_match'] else 'âŒ'}
"""

    report += "\n---\n\n## Per-Factor Accuracy\n"
    report += "| Factor | Detection | Level Accuracy | Score Accuracy |\n"
    report += "|--------|-----------|---------------|----------------|\n"
    for factor, acc in eval_result.get("per_factor", {}).items():
        report += f"| {factor} | {acc['detection_rate']}% | {acc['level_accuracy']}% | {acc['score_accuracy']}% |\n"

    return report

