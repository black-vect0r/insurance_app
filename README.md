# ðŸ›¡ï¸ AI-Powered Risk Assessment Assistant for Insurance Underwriting

**TCS AI Fridays Season 2 â€” Hackathon Submission**

An AI agent that reads insurance applications, extracts risk factors, calculates a preliminary risk score, flags compliance issues, and provides underwriter recommendations â€” powered by TCS GenAI Lab.

---

## ðŸš€ Quick Start

```bash
cd insurance-risk-agent
python -m venv venv
venv\Scripts\activate                    # Windows

pip install -r backend/requirements.txt

# Create .env with your API key
Set-Content -Path .env -Value "GENAILAB_API_KEY=sk-your-key" -Encoding UTF8

streamlit run app.py
```

---

## ðŸ—ï¸ Architecture

```
Insurance Application (PDF/TXT/Paste)
        â”‚
        â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Document Text Extractor  â”‚  pdfminer.six / raw text
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
            â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  DeepSeek-V3 (TCS Lab)   â”‚  Structured risk extraction
â”‚  â†’ JSON: factors, scores, â”‚  (8 risk dimensions scored 0-6 + overall 0-100)
â”‚    compliance, recommend   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
            â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  GPT-4o-mini (TCS Lab)   â”‚  Human-readable report
â”‚  â†’ Markdown report for    â”‚
â”‚    underwriter review      â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
            â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Streamlit Dashboard      â”‚  Score gauge, bar chart,
â”‚  â†’ Risk visualization     â”‚  compliance checks, download
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## ðŸ“ Project Structure

```
insurance-risk-agent/
â”œâ”€â”€ app.py                          # Main Streamlit app (3 tabs)
â”œâ”€â”€ backend/
â”‚   â”œâ”€â”€ risk_engine.py              # AI risk assessment + report generation
â”‚   â”œâ”€â”€ sample_data.py              # 5 synthetic applications + scoring guide
â”‚   â”œâ”€â”€ doc_extractor.py            # PDF/TXT text extraction
â”‚   â””â”€â”€ requirements.txt
â”œâ”€â”€ .env.example                    # Template (safe to commit)
â”œâ”€â”€ .env                            # Your actual key (gitignored)
â”œâ”€â”€ .gitignore
â”œâ”€â”€ .vscode/launch.json
â””â”€â”€ README.md
```

---

## ðŸ” Risk Factors Scored (8 Dimensions)

| Factor | Low (1) | Medium (3) | High (5-6) |
|--------|---------|------------|------------|
| Age | 25-45 | 46-60 / 18-24 | 61+ |
| BMI | 18.5-24.9 | 25-29.9 | 30+ |
| Smoking | Non-smoker (0) | Former (2) | Current (5) |
| Pre-existing | None (0) | Controlled (2-4) | Serious (6) |
| Occupation | Office/IT | Sales/Nurse | Construction/Mining |
| Family History | None (0) | Moderate (2) | Heart/Cancer (4) |
| Coverage Ratio | Under 10x income | 10-15x | Over 15x |
| Lifestyle | Sedentary (0) | Moderate (1) | Extreme sports (4) |

**Score Thresholds (0-100):**
- 0-29: 🟢 LOW → Approve
- 30-59: 🟡 MEDIUM → Approve with Conditions
- 60-80: 🟠 HIGH → Human Intervention / Senior Underwriter
- 81-100: 🔴 SEVERELY HIGH → Likely Decline / Senior Review
- 

---

## âš–ï¸ Compliance Checks

- **IRDAI-01**: Full medical disclosure
- **IRDAI-02**: Pre-existing conditions 48-month waiting period
- **IRDAI-03**: Free-look period (15/30 days)
- **IRDAI-04**: Sum assured within income multiple limits
- **AML-01**: KYC for policies above â‚¹50,000
- **AML-02**: PEP (Politically Exposed Person) screening
- **DATA-01**: DPDP Act 2023 compliance
- **DISC-01**: Material facts non-disclosure (Section 45)

---

## ðŸ“‹ 5 Sample Applications Included

1. **Rajesh Sharma** (42M, IT) â€” Clean profile, low risk
2. **Priya Venkatesh** (55F, Banking) â€” Hypertension, former smoker, family cancer history
3. **Mohammad Arif Khan** (29M, Construction) â€” Current smoker, high BMI, previous rejection
4. **Anita Desai** (35F, CA) â€” Ideal profile, no flags
5. **Vikram Singh Rathore** (63M, Retired Army) â€” Multiple pre-existing, high age

---

## âš™ï¸ Models Used

| Model | Purpose |
|-------|---------|
| `azure_ai/genailab-maas-DeepSeek-V3-0324` | Risk factor extraction + scoring |
| `azure/genailab-maas-gpt-4o-mini` | Human-readable report generation |

