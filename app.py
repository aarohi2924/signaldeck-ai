import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LeadRank AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --border: #1e1e2e;
    --accent: #6ee7b7;
    --accent2: #f472b6;
    --accent3: #fbbf24;
    --text: #e2e8f0;
    --muted: #64748b;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background: var(--bg); }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1400px; }

/* ── Hero Header ── */
.hero {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 0.25rem;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* ── Metric Cards ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.green::before { background: var(--accent); }
.metric-card.pink::before  { background: var(--accent2); }
.metric-card.gold::before  { background: var(--accent3); }
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}
.metric-value {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
}
.metric-card.green .metric-value { color: var(--accent); }
.metric-card.pink  .metric-value { color: var(--accent2); }
.metric-card.gold  .metric-value { color: var(--accent3); }

/* ── Upload Zone ── */
.upload-zone {
    border: 1.5px dashed var(--border);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    background: var(--surface);
    margin-bottom: 2rem;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: var(--accent); }

/* ── Section label ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.75rem;
}

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.05em;
}
.badge-high   { background: rgba(110,231,183,0.15); color: #6ee7b7; border: 1px solid rgba(110,231,183,0.3); }
.badge-medium { background: rgba(251,191,36,0.15);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.badge-low    { background: rgba(100,116,139,0.15); color: #94a3b8; border: 1px solid rgba(100,116,139,0.3); }

/* ── Dataframe override ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }
[data-testid="stDataFrame"] div { font-family: 'DM Mono', monospace; font-size: 0.78rem; }

/* ── Buttons ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, var(--accent), #34d399) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stDownloadButton > button:hover { opacity: 0.85 !important; }

.stButton > button {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
}

/* Selectbox / radio */
.stSelectbox label, .stRadio label { color: var(--muted) !important; font-size: 0.8rem; }

/* Info box */
.info-box {
    background: rgba(110,231,183,0.07);
    border: 1px solid rgba(110,231,183,0.2);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--accent);
    margin-bottom: 1.5rem;
}
.warn-box {
    background: rgba(251,191,36,0.07);
    border: 1px solid rgba(251,191,36,0.25);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--accent3);
    margin-bottom: 1.5rem;
}

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
</style>
""", unsafe_allow_html=True)


# ── Synthetic training data ───────────────────────────────────────────────────
@st.cache_data
def generate_training_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    job_titles   = ["CEO","CTO","VP Sales","Director","Manager","Analyst","Engineer","Consultant","Founder","Head of Growth"]
    company_sizes= ["1-10","11-50","51-200","201-500","501-1000","1000+"]
    industries   = ["SaaS","FinTech","Healthcare","E-commerce","Manufacturing","Consulting","EdTech","Real Estate","Media","Retail"]
    locations    = ["New York","San Francisco","London","Austin","Chicago","Boston","Seattle","Los Angeles","Berlin","Singapore"]
    seniorities  = ["C-Suite","VP","Director","Senior","Mid","Junior","Intern"]

    df = pd.DataFrame({
        "job_title"        : rng.choice(job_titles,   n),
        "company_size"     : rng.choice(company_sizes, n),
        "industry"         : rng.choice(industries,   n),
        "location"         : rng.choice(locations,    n),
        "seniority"        : rng.choice(seniorities,  n),
        "engagement_signal": rng.integers(0, 2, n),
    })

    # Simulate realistic label
    score = np.zeros(n)
    score += np.isin(df["job_title"],   ["CEO","CTO","Founder","VP Sales"]).astype(float) * 0.35
    score += np.isin(df["seniority"],   ["C-Suite","VP","Director"]).astype(float)        * 0.25
    score += np.isin(df["company_size"],["201-500","501-1000","1000+"]).astype(float)     * 0.2
    score += np.isin(df["industry"],    ["SaaS","FinTech","Healthcare"]).astype(float)    * 0.15
    score += df["engagement_signal"] * 0.2
    score += rng.uniform(-0.15, 0.15, n)
    df["high_value_lead"] = (score > 0.45).astype(int)
    return df


# ── Model builder ─────────────────────────────────────────────────────────────
@st.cache_resource
def build_model(model_type="Random Forest"):
    df = generate_training_data()
    cat_cols = ["job_title","company_size","industry","location","seniority"]
    num_cols = ["engagement_signal"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    if model_type == "Random Forest":
        clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=42)

    pipeline = Pipeline([("prep", preprocessor), ("clf", clf)])
    X = df.drop(columns=["high_value_lead"])
    y = df["high_value_lead"]
    pipeline.fit(X, y)
    return pipeline


# ── Scoring helpers ───────────────────────────────────────────────────────────
def score_leads(model, df: pd.DataFrame) -> pd.DataFrame:
    required = ["job_title","company_size","industry","location","seniority","engagement_signal"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    proba = model.predict_proba(df[required])[:, 1]
    out = df.copy()
    out["score"] = np.round(proba, 3)
    out["priority"] = pd.cut(
        out["score"],
        bins=[-np.inf, 0.4, 0.7, np.inf],
        labels=["Low","Medium","High"]
    )
    action_map = {"High": "Contact now", "Medium": "Nurture", "Low": "Hold"}
    out["recommended_action"] = out["priority"].map(action_map)
    return out


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def sample_csv() -> bytes:
    rng = np.random.default_rng(99)
    job_titles   = ["CEO","CTO","VP Sales","Director","Manager","Analyst","Engineer","Consultant","Founder","Head of Growth"]
    company_sizes= ["1-10","11-50","51-200","201-500","501-1000","1000+"]
    industries   = ["SaaS","FinTech","Healthcare","E-commerce","Manufacturing","Consulting","EdTech","Real Estate","Media","Retail"]
    locations    = ["New York","San Francisco","London","Austin","Chicago","Boston","Seattle","Los Angeles","Berlin","Singapore"]
    seniorities  = ["C-Suite","VP","Director","Senior","Mid","Junior","Intern"]
    n = 20
    df = pd.DataFrame({
        "job_title"        : rng.choice(job_titles,   n),
        "company_size"     : rng.choice(company_sizes, n),
        "industry"         : rng.choice(industries,   n),
        "location"         : rng.choice(locations,    n),
        "seniority"        : rng.choice(seniorities,  n),
        "engagement_signal": rng.integers(0, 2, n),
    })
    return to_csv_bytes(df)


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="hero">
    <span class="hero-title">LeadRank</span>
    <span style="font-size:1.4rem;color:#6ee7b7;font-weight:700;">AI</span>
</div>
<div class="hero-sub">AI-Based People Prioritization System · v1.0</div>
""", unsafe_allow_html=True)

# Sidebar – model selection
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")
    model_type = st.radio("Algorithm", ["Random Forest", "Logistic Regression"], index=0)
    st.markdown("---")
    st.markdown("**Column reference**")
    st.caption("`job_title` · `company_size` · `industry` · `location` · `seniority` · `engagement_signal`")
    st.markdown("---")
    st.markdown("**Priority thresholds**")
    st.caption("≥ 0.70 → High 🟢")
    st.caption("≥ 0.40 → Medium 🟡")
    st.caption("< 0.40 → Low ⚪")

# Build / cache model
with st.spinner("Training model on synthetic data…"):
    model = build_model(model_type)

st.markdown('<div class="info-box">✓ Model ready &nbsp;·&nbsp; Trained on 2,000 synthetic leads &nbsp;·&nbsp; Algorithm: ' + model_type + '</div>', unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
col_up, col_dl = st.columns([3, 1])
with col_up:
    st.markdown('<div class="section-label">Upload leads CSV</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

with col_dl:
    st.markdown('<div class="section-label">Need a sample?</div>', unsafe_allow_html=True)
    st.download_button(
        "⬇ Download sample CSV",
        data=sample_csv(),
        file_name="sample_leads.csv",
        mime="text/csv",
    )

# ── Process ───────────────────────────────────────────────────────────────────
if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        result = score_leads(model, raw)

        # Metrics
        total  = len(result)
        n_high = int((result["priority"] == "High").sum())
        avg_sc = float(result["score"].mean())

        st.markdown(f"""
        <div class="metrics-row">
            <div class="metric-card green">
                <div class="metric-label">Total Leads</div>
                <div class="metric-value">{total}</div>
            </div>
            <div class="metric-card pink">
                <div class="metric-label">High Priority</div>
                <div class="metric-value">{n_high}</div>
            </div>
            <div class="metric-card gold">
                <div class="metric-label">Average Score</div>
                <div class="metric-value">{avg_sc:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Priority breakdown bar
        n_med = int((result["priority"] == "Medium").sum())
        n_low = int((result["priority"] == "Low").sum())
        pct_h = n_high / total * 100
        pct_m = n_med  / total * 100
        pct_l = n_low  / total * 100

        st.markdown(f"""
        <div style="margin-bottom:2rem;">
            <div class="section-label">Priority breakdown</div>
            <div style="display:flex;height:8px;border-radius:999px;overflow:hidden;gap:2px;">
                <div style="width:{pct_h:.1f}%;background:#6ee7b7;border-radius:999px 0 0 999px;"></div>
                <div style="width:{pct_m:.1f}%;background:#fbbf24;"></div>
                <div style="width:{pct_l:.1f}%;background:#334155;border-radius:0 999px 999px 0;"></div>
            </div>
            <div style="display:flex;gap:1.5rem;margin-top:0.6rem;">
                <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#6ee7b7;">High {n_high} ({pct_h:.0f}%)</span>
                <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#fbbf24;">Medium {n_med} ({pct_m:.0f}%)</span>
                <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#64748b;">Low {n_low} ({pct_l:.0f}%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Table
        st.markdown('<div class="section-label">Scored leads</div>', unsafe_allow_html=True)

        display = result.copy()
        display["score"] = display["score"].apply(lambda x: f"{x:.3f}")

        st.dataframe(
            display,
            use_container_width=True,
            height=420,
            column_config={
                "score":              st.column_config.TextColumn("Score"),
                "priority":           st.column_config.TextColumn("Priority"),
                "recommended_action": st.column_config.TextColumn("Action"),
            }
        )

        # Download
        st.markdown("---")
        st.download_button(
            "⬇ Download scored CSV",
            data=to_csv_bytes(result),
            file_name="scored_leads.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.markdown(f'<div class="warn-box">⚠ Error processing file: {e}</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="border:1.5px dashed #1e1e2e;border-radius:16px;padding:3rem;text-align:center;background:#12121a;color:#334155;margin-top:1rem;">
        <div style="font-size:2.5rem;margin-bottom:0.75rem;">📂</div>
        <div style="font-size:1rem;color:#475569;">Upload a CSV file above to start scoring your leads</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#334155;margin-top:0.5rem;">
            Required columns: job_title · company_size · industry · location · seniority · engagement_signal
        </div>
    </div>
    """, unsafe_allow_html=True)
