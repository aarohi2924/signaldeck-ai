# LeadRank AI – Setup & Run Guide

## Prerequisites
- Python 3.9 or higher
- pip

---

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Run the app

```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

---

## 3. Using the app

1. Click **"Download sample CSV"** to get a ready-to-use test file, OR prepare your own CSV with these columns:

| Column | Type | Example |
|---|---|---|
| `job_title` | text | CEO, VP Sales, Manager |
| `company_size` | text | 1-10, 51-200, 1000+ |
| `industry` | text | SaaS, FinTech, Healthcare |
| `location` | text | New York, London |
| `seniority` | text | C-Suite, VP, Director, Senior |
| `engagement_signal` | 0 or 1 | 1 |

2. Upload your CSV using the file uploader
3. The model scores each lead instantly
4. Review the priority table and summary metrics
5. Download the scored CSV with the **"Download scored CSV"** button

---

## 4. Model details

- **Algorithm**: Random Forest (200 estimators) or Logistic Regression (selectable in sidebar)
- **Training data**: 2,000 synthetic leads generated on app startup
- **Features**: OneHotEncoded categoricals + numeric `engagement_signal`
- **Target**: `high_value_lead` (binary)

### Priority thresholds
| Score | Priority | Action |
|---|---|---|
| ≥ 0.70 | High | Contact now |
| 0.40 – 0.69 | Medium | Nurture |
| < 0.40 | Low | Hold |

---

## 5. Project structure

```
.
├── app.py           # Main Streamlit application
├── requirements.txt # Python dependencies
└── README.md        # This file
```

No external APIs. No pre-trained model file needed — the model trains in-memory on launch.
