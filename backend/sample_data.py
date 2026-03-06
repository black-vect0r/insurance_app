"""
📋 Sample Insurance Application Generator
Generates synthetic insurance applications for testing the risk assessment agent.
"""

import json
import random
import os
from datetime import datetime, timedelta

# ═══════════════════════════════════════
# Risk Factor Definitions & Scoring
# ═══════════════════════════════════════
RISK_FACTORS = {
    "age": {
        "low":    {"range": "25-45", "score": 1, "description": "Prime insurable age"},
        "medium": {"range": "46-60 or 18-24", "score": 3, "description": "Moderate age risk"},
        "high":   {"range": "61+", "score": 5, "description": "Elevated age-related risk"},
    },
    "bmi": {
        "low":    {"range": "18.5-24.9", "score": 1, "description": "Normal BMI"},
        "medium": {"range": "25-29.9", "score": 3, "description": "Overweight"},
        "high":   {"range": "30+", "score": 5, "description": "Obese — elevated health risk"},
    },
    "smoking_status": {
        "non_smoker":    {"score": 0, "description": "No tobacco use"},
        "former_smoker": {"score": 2, "description": "Quit within last 5 years"},
        "current_smoker":{"score": 5, "description": "Active tobacco use — major risk"},
    },
    "pre_existing_conditions": {
        "none":         {"score": 0, "description": "No pre-existing conditions"},
        "mild":         {"score": 2, "description": "Controlled conditions (e.g., mild asthma, allergies)"},
        "moderate":     {"score": 4, "description": "Managed conditions (e.g., hypertension, Type 2 diabetes)"},
        "severe":       {"score": 6, "description": "Serious conditions (e.g., heart disease, cancer history)"},
    },
    "occupation_risk": {
        "low":    {"examples": "Office worker, Teacher, IT Professional", "score": 1},
        "medium": {"examples": "Sales, Delivery driver, Nurse", "score": 3},
        "high":   {"examples": "Construction, Mining, Military, Pilot", "score": 5},
    },
    "family_history": {
        "none":    {"score": 0, "description": "No significant family medical history"},
        "moderate":{"score": 2, "description": "Family history of manageable conditions"},
        "high":    {"score": 4, "description": "Family history of heart disease, cancer, or diabetes"},
    },
    "coverage_amount": {
        "standard":  {"range": "Up to ₹25L", "score": 1},
        "elevated":  {"range": "₹25L - ₹1Cr", "score": 2},
        "high":      {"range": "Above ₹1Cr", "score": 3},
    },
    "lifestyle": {
        "low_risk":  {"examples": "Sedentary, no extreme activities", "score": 0},
        "moderate":  {"examples": "Regular travel, moderate sports", "score": 1},
        "high_risk": {"examples": "Extreme sports, heavy alcohol use, adventure activities", "score": 4},
    },
}

# Scoring thresholds
RISK_THRESHOLDS = {
    "LOW": {"min_score": 0, "max_score": 29, "recommendation": "APPROVE", "color": "green"},
    "MEDIUM": {"min_score": 30, "max_score": 59, "recommendation": "APPROVE WITH CONDITIONS", "color": "yellow"},
    "HIGH": {"min_score": 60, "max_score": 80, "recommendation": "HUMAN INTERVENTION REQUIRED (REFER TO SENIOR UNDERWRITER)", "color": "orange"},
    "SEVERELY_HIGH": {"min_score": 81, "max_score": 100, "recommendation": "LIKELY DECLINE / SENIOR REVIEW", "color": "red"},
}

# Regulatory compliance checks
COMPLIANCE_CHECKS = [
    {"id": "IRDAI-01", "rule": "Proposal form must include full medical disclosure", "category": "IRDAI Guidelines"},
    {"id": "IRDAI-02", "rule": "Pre-existing conditions waiting period: 48 months per IRDAI", "category": "IRDAI Guidelines"},
    {"id": "IRDAI-03", "rule": "Free-look period of 15 days must be communicated", "category": "IRDAI Guidelines"},
    {"id": "IRDAI-04", "rule": "Sum assured must comply with income multiple limits", "category": "IRDAI Guidelines"},
    {"id": "AML-01",   "rule": "KYC verification required for all policies above ₹50,000", "category": "Anti-Money Laundering"},
    {"id": "AML-02",   "rule": "PEP (Politically Exposed Person) screening mandatory", "category": "Anti-Money Laundering"},
    {"id": "AML-03",   "rule": "Source of funds / income proof should be documented for high-value policies", "category": "Anti-Money Laundering"},
    {"id": "AML-04",   "rule": "Sanctions and adverse media screening should be completed before issuance", "category": "Anti-Money Laundering"},
    {"id": "DATA-01",  "rule": "Personal data handling per DPDP Act 2023", "category": "Data Protection"},
    {"id": "DATA-02",  "rule": "Explicit applicant consent for data processing should be captured", "category": "Data Protection"},
    {"id": "DISC-01",  "rule": "Material facts non-disclosure voids policy under Section 45", "category": "Insurance Act"},
    {"id": "IRDAI-05", "rule": "Nominee or beneficiary details should be present in the proposal", "category": "IRDAI Guidelines"},
    {"id": "IRDAI-06", "rule": "Policy exclusions and terms acknowledgement should be captured", "category": "IRDAI Guidelines"},
    {"id": "UW-01",    "rule": "Medical declaration should be internally consistent with medication/disclosure details", "category": "Underwriting Quality"},
    {"id": "UW-02",    "rule": "Prior claims or rejections must be explicitly disclosed for underwriting review", "category": "Underwriting Quality"},
]

# ═══════════════════════════════════════
# Sample Application Generator
# ═══════════════════════════════════════
SAMPLE_APPLICATIONS = [
    {
        "applicant_name": "Rajesh Kumar Sharma",
        "age": 42,
        "gender": "Male",
        "occupation": "Software Engineer at Infosys",
        "annual_income": "₹18,50,000",
        "city": "Bangalore",
        "policy_type": "Term Life Insurance",
        "coverage_amount": "₹1,00,00,000 (1 Crore)",
        "policy_term": "25 years",
        "smoking_status": "Non-smoker",
        "alcohol_consumption": "Social drinker (occasional)",
        "bmi": 24.2,
        "height": "5'9\"",
        "weight": "72 kg",
        "pre_existing_conditions": "None",
        "current_medications": "None",
        "family_medical_history": "Father had Type 2 diabetes diagnosed at age 58. Mother is healthy at 68.",
        "previous_claims": "None",
        "previous_policy_rejections": "None",
        "dangerous_activities": "None",
        "travel_history": "Regular domestic travel. One international trip to USA in 2024.",
        "additional_notes": "Applicant exercises regularly, runs 5km three times a week. No hospitalization in last 10 years.",
    },
    {
        "applicant_name": "Priya Venkatesh",
        "age": 55,
        "gender": "Female",
        "occupation": "Senior Manager at HDFC Bank",
        "annual_income": "₹32,00,000",
        "city": "Mumbai",
        "policy_type": "Health Insurance + Critical Illness Rider",
        "coverage_amount": "₹50,00,000 (50 Lakhs)",
        "policy_term": "Renewable annually",
        "smoking_status": "Former smoker (quit 3 years ago)",
        "alcohol_consumption": "Non-drinker",
        "bmi": 28.5,
        "height": "5'4\"",
        "weight": "74 kg",
        "pre_existing_conditions": "Hypertension (controlled with medication since 2020). Mild thyroid disorder.",
        "current_medications": "Amlodipine 5mg daily, Thyronorm 50mcg daily",
        "family_medical_history": "Mother passed away from breast cancer at age 62. Father had coronary artery disease.",
        "previous_claims": "One hospitalization claim in 2022 for gallbladder surgery (₹2,80,000)",
        "previous_policy_rejections": "None",
        "dangerous_activities": "None",
        "travel_history": "Frequent domestic travel for work.",
        "additional_notes": "Applicant has regular health checkups. Last cardiac stress test in Jan 2025 was normal. BP well controlled at 130/82.",
    },
    {
        "applicant_name": "Mohammad Arif Khan",
        "age": 29,
        "gender": "Male",
        "occupation": "Construction Site Supervisor",
        "annual_income": "₹6,50,000",
        "city": "Hyderabad",
        "policy_type": "Term Life Insurance + Accidental Death Benefit",
        "coverage_amount": "₹75,00,000 (75 Lakhs)",
        "policy_term": "30 years",
        "smoking_status": "Current smoker (10 cigarettes/day for 8 years)",
        "alcohol_consumption": "Regular (3-4 drinks per week)",
        "bmi": 31.2,
        "height": "5'7\"",
        "weight": "90 kg",
        "pre_existing_conditions": "None diagnosed. Reports occasional back pain.",
        "current_medications": "None",
        "family_medical_history": "Father died of heart attack at age 52. Uncle had lung cancer.",
        "previous_claims": "Work-related injury claim in 2023 (fractured wrist)",
        "previous_policy_rejections": "One rejection from LIC in 2024 (reason not disclosed)",
        "dangerous_activities": "Works at heights regularly (construction). Occasional motorcycle racing.",
        "travel_history": "Local travel only.",
        "additional_notes": "Coverage amount is ~11.5x annual income which exceeds standard 10x income multiple guideline. Previous rejection warrants investigation.",
    },
    {
        "applicant_name": "Anita Desai",
        "age": 35,
        "gender": "Female",
        "occupation": "Chartered Accountant (Self-employed)",
        "annual_income": "₹24,00,000",
        "city": "Pune",
        "policy_type": "Whole Life Insurance",
        "coverage_amount": "₹2,00,00,000 (2 Crore)",
        "policy_term": "Whole life",
        "smoking_status": "Non-smoker",
        "alcohol_consumption": "Non-drinker",
        "bmi": 22.1,
        "height": "5'5\"",
        "weight": "58 kg",
        "pre_existing_conditions": "None",
        "current_medications": "Multivitamins only",
        "family_medical_history": "No significant family history. Both parents alive and healthy in their 60s.",
        "previous_claims": "None",
        "previous_policy_rejections": "None",
        "dangerous_activities": "Yoga and swimming regularly",
        "travel_history": "Annual international vacations. Recently visited Europe.",
        "additional_notes": "Excellent health profile. Coverage amount is within 8.3x income. Clean application with no red flags.",
    },
    {
        "applicant_name": "Vikram Singh Rathore",
        "age": 63,
        "gender": "Male",
        "occupation": "Retired Army Colonel",
        "annual_income": "₹12,00,000 (Pension + Investments)",
        "city": "Jaipur",
        "policy_type": "Health Insurance (Super Top-Up)",
        "coverage_amount": "₹25,00,000 (25 Lakhs) over base ₹5L",
        "policy_term": "Renewable annually",
        "smoking_status": "Former smoker (quit 15 years ago)",
        "alcohol_consumption": "Occasional (1-2 drinks per month)",
        "bmi": 26.8,
        "height": "5'11\"",
        "weight": "86 kg",
        "pre_existing_conditions": "Type 2 Diabetes (since 2015, HbA1c: 7.2%). Mild osteoarthritis in knees. History of kidney stones (2018).",
        "current_medications": "Metformin 1000mg twice daily, Glucosamine supplements",
        "family_medical_history": "Father had stroke at 70. Mother had diabetes.",
        "previous_claims": "Two claims: knee arthroscopy (2021, ₹1,50,000) and kidney stone removal (2018, ₹95,000)",
        "previous_policy_rejections": "None, but previous insurer loaded premium by 25%",
        "dangerous_activities": "None currently. Former military service in high-risk zones.",
        "travel_history": "Domestic travel. No recent international travel.",
        "additional_notes": "Multiple pre-existing conditions with medication dependency. Previous premium loading suggests elevated risk recognized by prior insurer. Age-related concerns for new health policy.",
    },
]


def get_sample_applications():
    """Return all sample applications."""
    return SAMPLE_APPLICATIONS


def get_sample_application_text(index=0):
    """Return a sample application as plain text (simulating uploaded document)."""
    app = SAMPLE_APPLICATIONS[index % len(SAMPLE_APPLICATIONS)]

    text = f"""
INSURANCE APPLICATION FORM
===========================
Date: {datetime.now().strftime('%d-%b-%Y')}
Application Reference: APP-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000,9999)}

SECTION 1: PERSONAL DETAILS
----------------------------
Full Name: {app['applicant_name']}
Age: {app['age']} years
Gender: {app['gender']}
City: {app['city']}
Occupation: {app['occupation']}
Annual Income: {app['annual_income']}

SECTION 2: POLICY DETAILS
--------------------------
Policy Type: {app['policy_type']}
Sum Assured / Coverage: {app['coverage_amount']}
Policy Term: {app['policy_term']}

SECTION 3: HEALTH DECLARATION
------------------------------
Height: {app['height']}
Weight: {app['weight']}
BMI: {app['bmi']}
Smoking Status: {app['smoking_status']}
Alcohol Consumption: {app['alcohol_consumption']}

Pre-existing Conditions: {app['pre_existing_conditions']}
Current Medications: {app['current_medications']}
Family Medical History: {app['family_medical_history']}

SECTION 4: CLAIMS & POLICY HISTORY
------------------------------------
Previous Claims: {app['previous_claims']}
Previous Policy Rejections: {app['previous_policy_rejections']}

SECTION 5: LIFESTYLE & ACTIVITIES
-----------------------------------
Dangerous Activities / Hobbies: {app['dangerous_activities']}
Travel History: {app['travel_history']}

SECTION 6: ADDITIONAL INFORMATION
-----------------------------------
{app['additional_notes']}

DECLARATION: I hereby declare that the above information is true and complete to the best of my knowledge.
Signature: _______________
Date: _______________
"""
    return text.strip()


def get_risk_factors():
    """Return risk factor definitions."""
    return RISK_FACTORS


def get_compliance_checks():
    """Return regulatory compliance checklist."""
    return COMPLIANCE_CHECKS


def get_risk_thresholds():
    """Return risk scoring thresholds."""
    return RISK_THRESHOLDS


if __name__ == "__main__":
    print("=" * 60)
    print("Sample Insurance Applications")
    print("=" * 60)
    for i, app in enumerate(SAMPLE_APPLICATIONS):
        print(f"\n[{i+1}] {app['applicant_name']} — {app['policy_type']}")
        print(f"    Age: {app['age']}, BMI: {app['bmi']}, Smoking: {app['smoking_status']}")
        print(f"    Coverage: {app['coverage_amount']}")

    print(f"\n\n--- Sample Application Text ---")
    print(get_sample_application_text(0))
