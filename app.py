from __future__ import annotations

from datetime import datetime
from io import BytesIO
import json
import os
import random
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="Lead Scoring Demo", page_icon=":bar_chart:", layout="wide")


REQUIRED_COLUMNS = [
    "lead_id",
    "company_size",
    "industry",
    "source",
    "country",
    "budget",
    "engagement_score",
    "last_activity_days",
    "opened_email",
    "clicked_link",
    "requested_demo",
]

PASSTHROUGH_COLUMNS = ["company_name", "role_title"]

NUMERIC_FEATURES = [
    "budget",
    "engagement_score",
    "last_activity_days",
    "opened_email",
    "clicked_link",
    "requested_demo",
]

CATEGORICAL_FEATURES = ["company_size", "industry", "source", "country"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DISPLAY_COLUMNS = [
    "lead_id",
    "company_name",
    "role_title",
    "industry",
    "country",
    "score",
    "priority",
    "recommended_action",
    "why_this_lead",
]


# Mock enrichment boundary for future API integrations.
# Future real providers could include:
# - Clearbit or Apollo for firmographic/contact enrichment
# - Hunter.io for email lookup
# - Google Custom Search or SerpAPI for company context
# - News API for recent company signals
# This demo intentionally does not add scraping or require API keys.
# Optional API enrichment is controlled by environment variables only.


def mock_company_summary(company_name: str) -> str:
    return (
        f"{company_name} is a simulated account in this workspace, standing in for a realistic GTM target. "
        "This summary is mock enrichment today and can later be replaced by a real company intelligence provider."
    )


def mock_contact_email(company_name: str, role_title: str) -> str:
    slug = "".join(character.lower() for character in company_name if character.isalnum())
    role_hint = "".join(character.lower() for character in role_title.split()[0] if character.isalpha()) or "team"
    return f"{role_hint}@{slug}.com"


def mock_recent_signals(company_name: str) -> str:
    return (
        f"Recent mock signals for {company_name}: stronger inbound engagement, repeat page views, "
        "and renewed activity across recent campaign touchpoints."
    )


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def enrichment_available() -> bool:
    pairs = [
        ("COMPANY_SUMMARY_API_URL", "COMPANY_SUMMARY_API_KEY"),
        ("CONTACT_EMAIL_API_URL", "CONTACT_EMAIL_API_KEY"),
        ("RECENT_SIGNALS_API_URL", "RECENT_SIGNALS_API_KEY"),
    ]
    return any(_env(url_name) and _env(key_name) for url_name, key_name in pairs)


def _fetch_optional_api_text(
    *,
    url_env: str,
    key_env: str,
    response_field_env: str,
    default_field: str,
    params: dict[str, str],
) -> str | None:
    base_url = _env(url_env)
    api_key = _env(key_env)
    if not base_url or not api_key:
        return None

    field_name = _env(response_field_env) or default_field
    query = urlencode(params)
    separator = "&" if "?" in base_url else "?"
    request = Request(f"{base_url}{separator}{query}")
    request.add_header("Authorization", f"Bearer {api_key}")
    request.add_header("Accept", "application/json")

    try:
        with urlopen(request, timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None

    value = payload.get(field_name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def fetch_company_summary(company_name: str) -> str:
    api_value = _fetch_optional_api_text(
        url_env="COMPANY_SUMMARY_API_URL",
        key_env="COMPANY_SUMMARY_API_KEY",
        response_field_env="COMPANY_SUMMARY_RESPONSE_FIELD",
        default_field="summary",
        params={"company_name": company_name},
    )
    return api_value or mock_company_summary(company_name)


def fetch_contact_email(company_name: str, role_title: str) -> str:
    api_value = _fetch_optional_api_text(
        url_env="CONTACT_EMAIL_API_URL",
        key_env="CONTACT_EMAIL_API_KEY",
        response_field_env="CONTACT_EMAIL_RESPONSE_FIELD",
        default_field="email",
        params={"company_name": company_name, "role_title": role_title},
    )
    return api_value or mock_contact_email(company_name, role_title)


def fetch_recent_signals(company_name: str) -> str:
    api_value = _fetch_optional_api_text(
        url_env="RECENT_SIGNALS_API_URL",
        key_env="RECENT_SIGNALS_API_KEY",
        response_field_env="RECENT_SIGNALS_RESPONSE_FIELD",
        default_field="signals",
        params={"company_name": company_name},
    )
    return api_value or mock_recent_signals(company_name)


def build_sample_leads() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "lead_id": "L-1001",
                "company_name": "Northstar Labs",
                "role_title": "VP Marketing",
                "company_size": 25,
                "industry": "SaaS",
                "source": "Website",
                "country": "USA",
                "budget": 12000,
                "engagement_score": 88,
                "last_activity_days": 1,
                "opened_email": 1,
                "clicked_link": 1,
                "requested_demo": 1,
            },
            {
                "lead_id": "L-1002",
                "company_name": "Atlas Works",
                "role_title": "Revenue Operations Manager",
                "company_size": 350,
                "industry": "Manufacturing",
                "source": "Referral",
                "country": "Canada",
                "budget": 45000,
                "engagement_score": 73,
                "last_activity_days": 4,
                "opened_email": 1,
                "clicked_link": 1,
                "requested_demo": 0,
            },
            {
                "lead_id": "L-1003",
                "company_name": "Cedar Education",
                "role_title": "Marketing Director",
                "company_size": 15,
                "industry": "Education",
                "source": "Outbound",
                "country": "USA",
                "budget": 5000,
                "engagement_score": 35,
                "last_activity_days": 12,
                "opened_email": 1,
                "clicked_link": 0,
                "requested_demo": 0,
            },
        ]
    )


def build_training_data() -> pd.DataFrame:
    rows = [
        {"company_size": 20, "budget": 12000, "engagement_score": 90, "last_activity_days": 1, "opened_email": 1, "clicked_link": 1, "requested_demo": 1, "industry": "SaaS", "source": "Website", "country": "USA", "converted": 1},
        {"company_size": 300, "budget": 60000, "engagement_score": 85, "last_activity_days": 2, "opened_email": 1, "clicked_link": 1, "requested_demo": 1, "industry": "Finance", "source": "Referral", "country": "USA", "converted": 1},
        {"company_size": 60, "budget": 20000, "engagement_score": 68, "last_activity_days": 4, "opened_email": 1, "clicked_link": 1, "requested_demo": 0, "industry": "Healthcare", "source": "Event", "country": "UK", "converted": 1},
        {"company_size": 15, "budget": 4000, "engagement_score": 30, "last_activity_days": 14, "opened_email": 1, "clicked_link": 0, "requested_demo": 0, "industry": "Retail", "source": "Outbound", "country": "Canada", "converted": 0},
        {"company_size": 8, "budget": 2500, "engagement_score": 20, "last_activity_days": 21, "opened_email": 0, "clicked_link": 0, "requested_demo": 0, "industry": "Education", "source": "Website", "country": "India", "converted": 0},
        {"company_size": 500, "budget": 90000, "engagement_score": 88, "last_activity_days": 3, "opened_email": 1, "clicked_link": 1, "requested_demo": 1, "industry": "Manufacturing", "source": "Partner", "country": "Germany", "converted": 1},
        {"company_size": 45, "budget": 10000, "engagement_score": 55, "last_activity_days": 7, "opened_email": 1, "clicked_link": 0, "requested_demo": 0, "industry": "SaaS", "source": "Paid Search", "country": "USA", "converted": 0},
        {"company_size": 200, "budget": 30000, "engagement_score": 80, "last_activity_days": 2, "opened_email": 1, "clicked_link": 1, "requested_demo": 1, "industry": "Logistics", "source": "Referral", "country": "USA", "converted": 1},
        {"company_size": 90, "budget": 15000, "engagement_score": 65, "last_activity_days": 5, "opened_email": 1, "clicked_link": 1, "requested_demo": 0, "industry": "Healthcare", "source": "Website", "country": "Australia", "converted": 1},
        {"company_size": 12, "budget": 3500, "engagement_score": 25, "last_activity_days": 18, "opened_email": 0, "clicked_link": 0, "requested_demo": 0, "industry": "Consulting", "source": "Outbound", "country": "UK", "converted": 0},
        {"company_size": 150, "budget": 45000, "engagement_score": 74, "last_activity_days": 5, "opened_email": 1, "clicked_link": 1, "requested_demo": 0, "industry": "Finance", "source": "Event", "country": "Canada", "converted": 1},
        {"company_size": 25, "budget": 7000, "engagement_score": 42, "last_activity_days": 10, "opened_email": 1, "clicked_link": 0, "requested_demo": 0, "industry": "Education", "source": "Website", "country": "USA", "converted": 0},
        {"company_size": 380, "budget": 70000, "engagement_score": 92, "last_activity_days": 1, "opened_email": 1, "clicked_link": 1, "requested_demo": 1, "industry": "SaaS", "source": "Partner", "country": "USA", "converted": 1},
        {"company_size": 55, "budget": 9000, "engagement_score": 48, "last_activity_days": 8, "opened_email": 1, "clicked_link": 0, "requested_demo": 0, "industry": "Retail", "source": "Paid Search", "country": "Germany", "converted": 0},
        {"company_size": 240, "budget": 38000, "engagement_score": 81, "last_activity_days": 2, "opened_email": 1, "clicked_link": 1, "requested_demo": 1, "industry": "Healthcare", "source": "Referral", "country": "USA", "converted": 1},
        {"company_size": 18, "budget": 6000, "engagement_score": 34, "last_activity_days": 11, "opened_email": 1, "clicked_link": 0, "requested_demo": 0, "industry": "Manufacturing", "source": "Outbound", "country": "India", "converted": 0},
        {"company_size": 410, "budget": 52000, "engagement_score": 86, "last_activity_days": 1, "opened_email": 1, "clicked_link": 1, "requested_demo": 1, "industry": "Logistics", "source": "Website", "country": "USA", "converted": 1},
        {"company_size": 35, "budget": 11000, "engagement_score": 60, "last_activity_days": 6, "opened_email": 1, "clicked_link": 1, "requested_demo": 0, "industry": "Consulting", "source": "Event", "country": "UK", "converted": 1},
        {"company_size": 75, "budget": 22000, "engagement_score": 71, "last_activity_days": 4, "opened_email": 1, "clicked_link": 1, "requested_demo": 0, "industry": "Finance", "source": "Website", "country": "Australia", "converted": 1},
        {"company_size": 10, "budget": 2000, "engagement_score": 15, "last_activity_days": 27, "opened_email": 0, "clicked_link": 0, "requested_demo": 0, "industry": "Retail", "source": "Outbound", "country": "Canada", "converted": 0},
    ]
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["converted"]
    return pd.DataFrame(rows)[columns]


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@st.cache_resource
def train_model() -> Pipeline:
    training_df = build_training_data()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    model.fit(training_df[MODEL_FEATURES], training_df["converted"])
    return model


def synthetic_leads(seed: int, count: int = 96) -> pd.DataFrame:
    rng = random.Random(seed)
    companies = [
        "Northstar", "Rivet", "Bluewave", "Summit", "Cobalt", "Maple", "Brightline", "Pioneer",
        "Crescent", "Evergreen", "Altitude", "Granite", "Horizon", "Meridian", "Beacon", "Axis",
    ]
    suffixes = ["Labs", "Systems", "Works", "Health", "Retail", "Cloud", "Logistics", "Capital", "Partners"]
    titles = [
        "VP Marketing", "Head of Growth", "Marketing Director", "Demand Gen Manager",
        "Revenue Operations Manager", "Director of Sales", "VP Revenue", "Growth Lead",
        "Partnerships Director", "Chief Marketing Officer",
    ]
    industries = ["SaaS", "Healthcare", "Finance", "Retail", "Manufacturing", "Education", "Logistics", "Consulting"]
    sources = ["Website", "Referral", "Paid Search", "Event", "Outbound", "Partner"]
    countries = ["USA", "Canada", "UK", "Germany", "Australia", "India"]
    company_sizes = [10, 25, 40, 75, 120, 250, 500]

    rows: list[dict[str, object]] = []
    for index in range(count):
        requested_demo = 1 if rng.random() < 0.22 else 0
        clicked_link = 1 if rng.random() < 0.46 else 0
        opened_email = 1 if rng.random() < 0.70 else 0
        activity = rng.randint(0, 24)
        engagement = max(
            5,
            min(
                98,
                int(
                    rng.gauss(52, 18)
                    + requested_demo * 18
                    + clicked_link * 10
                    + opened_email * 4
                    - min(activity, 12)
                ),
            ),
        )
        rows.append(
            {
                "lead_id": f"L-{seed % 1000:03d}{index + 1:03d}",
                "company_name": f"{rng.choice(companies)} {rng.choice(suffixes)}",
                "role_title": rng.choice(titles),
                "company_size": rng.choice(company_sizes),
                "industry": rng.choice(industries),
                "source": rng.choice(sources),
                "country": rng.choice(countries),
                "budget": max(1500, int(rng.gauss(22000, 16000))),
                "engagement_score": engagement,
                "last_activity_days": activity,
                "opened_email": opened_email,
                "clicked_link": clicked_link,
                "requested_demo": requested_demo,
            }
        )
    return pd.DataFrame(rows)


def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in normalized.columns]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    passthrough_data = {}
    for column in PASSTHROUGH_COLUMNS:
        if column in normalized.columns:
            passthrough_data[column] = normalized[column]

    normalized = normalized[REQUIRED_COLUMNS].copy()

    for column in NUMERIC_FEATURES:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        normalized[column] = normalized[column].where(normalized[column].notna(), "Unknown")
        normalized[column] = normalized[column].map(lambda value: str(value).strip() if str(value).strip() else "Unknown")
        normalized[column] = normalized[column].astype(object)

    for column in ["opened_email", "clicked_link", "requested_demo"]:
        normalized[column] = normalized[column].fillna(0).clip(lower=0, upper=1).round().astype(int)

    normalized["budget"] = normalized["budget"].fillna(0).clip(lower=0)
    normalized["engagement_score"] = normalized["engagement_score"].fillna(0).clip(lower=0, upper=100)
    normalized["last_activity_days"] = normalized["last_activity_days"].fillna(999).clip(lower=0)
    normalized["lead_id"] = normalized["lead_id"].where(normalized["lead_id"].notna(), "").map(lambda value: str(value).strip())

    if normalized["lead_id"].eq("").any():
        raise ValueError("Each row must include a non-empty lead_id.")

    for column in PASSTHROUGH_COLUMNS:
        if column in passthrough_data:
            normalized[column] = passthrough_data[column].where(passthrough_data[column].notna(), "").map(lambda value: str(value).strip())
        else:
            normalized[column] = ""

    return normalized


def score_to_priority(score: float) -> str:
    if score >= 0.7:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def priority_to_action(priority: str) -> str:
    if priority == "High":
        return "Contact now"
    if priority == "Medium":
        return "Nurture"
    return "Hold"


def score_leads(df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    prepared = normalize_input(df)
    probabilities = model.predict_proba(prepared[MODEL_FEATURES])[:, 1]
    scored = prepared.copy()
    scored["score"] = probabilities.round(3)
    scored["priority"] = scored["score"].apply(score_to_priority)
    scored["recommended_action"] = scored["priority"].apply(priority_to_action)
    return scored


def role_importance(role_title: str) -> str:
    title = role_title.lower()
    if "chief" in title or "vp" in title or "head" in title:
        return "senior"
    if "director" in title:
        return "director"
    return "manager"


def why_this_lead(row: pd.Series) -> str:
    role_tier = role_importance(row["role_title"])
    role_phrase = {
        "senior": "a senior decision-maker",
        "director": "a director-level stakeholder",
        "manager": "an active day-to-day operator",
    }[role_tier]
    intent_phrase = (
        "They requested a demo, which signals clear buying intent"
        if row["requested_demo"] == 1
        else "They have not requested a demo yet, so urgency depends more on engagement behavior"
    )
    engagement_phrase = (
        f"The engagement score is {int(row['engagement_score'])}, which is strong for this dataset"
        if row["engagement_score"] >= 75
        else f"The engagement score is {int(row['engagement_score'])}, which suggests moderate interest"
        if row["engagement_score"] >= 45
        else f"The engagement score is {int(row['engagement_score'])}, which suggests limited current momentum"
    )
    recency_phrase = (
        f"The lead was active {int(row['last_activity_days'])} day ago, so the signal is very fresh"
        if row["last_activity_days"] <= 1
        else f"The lead was active {int(row['last_activity_days'])} days ago, so the signal is still recent"
        if row["last_activity_days"] <= 7
        else f"The lead has been quiet for {int(row['last_activity_days'])} days, which lowers short-term urgency"
    )
    return (
        f"{row['role_title']} is {role_phrase} at {row['company_name']}. "
        f"{intent_phrase}. {engagement_phrase}. {recency_phrase}."
    )


def add_explanations(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["why_this_lead"] = enriched.apply(why_this_lead, axis=1)
    return enriched


def build_subject_line(row: pd.Series) -> str:
    if row["priority"] == "High":
        return f"Next step for {row['company_name']} after recent engagement"
    if row["priority"] == "Medium":
        return f"Useful follow-up for {row['role_title']} at {row['company_name']}"
    return f"Keeping in touch with {row['company_name']}"


def build_outreach_message(row: pd.Series) -> str:
    if row["priority"] == "High":
        return (
            f"Hi there, I noticed strong recent engagement from {row['company_name']}. "
            f"Because you're a {row['role_title']} in {row['industry']}, I thought it would be useful to share a short next-step overview tailored to your team."
        )
    if row["priority"] == "Medium":
        return (
            f"Hi there, thanks for the recent activity from {row['company_name']}. "
            f"We often help {row['industry']} teams evaluate options like this, and I’d be happy to send a concise summary relevant to your role."
        )
    return (
        f"Hi there, keeping in touch in case this becomes more relevant for {row['company_name']}. "
        "If priorities shift later, I can send a brief overview tailored to your team."
    )


def build_linkedin_message(row: pd.Series) -> str:
    return (
        f"Hi, reaching out because {row['company_name']} looks like a strong fit for a quick intro. "
        f"If helpful, I can share a short overview relevant to a {row['role_title']} in {row['industry']}."
    )


def add_mock_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["company_summary"] = enriched["company_name"].apply(fetch_company_summary)
    enriched["mock_email"] = enriched.apply(lambda row: fetch_contact_email(row["company_name"], row["role_title"]), axis=1)
    enriched["recent_signals"] = enriched["company_name"].apply(fetch_recent_signals)
    enriched["email_subject"] = enriched.apply(build_subject_line, axis=1)
    enriched["outreach_message"] = enriched.apply(build_outreach_message, axis=1)
    enriched["linkedin_message"] = enriched.apply(build_linkedin_message, axis=1)
    return enriched


def to_csv_download(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def render_top_lead_cards(df: pd.DataFrame, empty_message: str) -> None:
    if df.empty:
        st.info(empty_message)
        return
    if "lead_id" not in df.columns:
        df = df.copy()
        df["lead_id"] = df.index.astype(str)

    cards = list(df.head(6).iterrows())
    for start in range(0, len(cards), 2):
        row_cols = st.columns(2, gap="large")
        for col, (_, row) in zip(row_cols, cards[start:start + 2]):
            with col:
                lead_id = str(row.get("lead_id", row.name))
                is_selected = st.session_state.selected_lead_id == lead_id
                preview = str(row["why_this_lead"])
                if len(preview) > 150:
                    preview = preview[:147].rstrip() + "..."
                priority_class = {
                    "High": "badge-accent",
                    "Medium": "badge-warn",
                    "Low": "badge-muted",
                }.get(str(row["priority"]), "badge-primary")
                action_class = "badge-primary" if row["recommended_action"] == "Contact now" else "badge-muted"
                selected_text = "Selected lead\n" if is_selected else ""
                card_label = (
                    f"{selected_text}{row['company_name']}\n"
                    f"{row['role_title']}\n"
                    f"{row['industry']} · {row['country']}\n"
                    f"Score {row['score']:.3f} · {row['priority']} · {row['recommended_action']}\n"
                    f"{preview}"
                )
                if st.button(card_label, key=f"card_select_{lead_id}", width="stretch", type="tertiary"):
                    st.session_state.selected_lead_id = lead_id
                    st.session_state.show_copy_message_for_lead = None
                    st.rerun()


def render_metric_card(label: str, value: str, tone: str = "neutral") -> None:
    tone_map = {
        "neutral": ("#1A1208", "#FDFBF8", "#E2D9CE", "◦", "linear-gradient(135deg, rgba(124,58,237,0.05), rgba(37,99,235,0.04))"),
        "high": ("#1A1208", "#F5FBF8", "#B8E5D5", "↗", "linear-gradient(135deg, rgba(16,185,129,0.14), rgba(37,99,235,0.05))"),
        "medium": ("#1A1208", "#FFF9F1", "#F1D7A6", "◔", "linear-gradient(135deg, rgba(245,158,11,0.16), rgba(124,58,237,0.05))"),
        "low": ("#1A1208", "#FBF8F4", "#E2D9CE", "•", "linear-gradient(135deg, rgba(138,122,104,0.10), rgba(255,255,255,0.85))"),
        "accent": ("#1A1208", "#F7F8FE", "#BFD0FA", "✦", "linear-gradient(135deg, rgba(37,99,235,0.14), rgba(124,58,237,0.09))"),
    }
    text_color, bg_color, border_color, icon, glow = tone_map[tone]
    st.markdown(
        f"""
        <div class="metric-card-shell" style="background:{bg_color};border-color:{border_color};">
            <div class="metric-glow" style="background:{glow};"></div>
            <div style="display:flex;justify-content:space-between;align-items:flex-start;position:relative;">
                <div style="
                    color:#8A7A68;
                    font-size:0.76rem;
                    font-weight:700;
                    text-transform:uppercase;
                    letter-spacing:0.14em;
                    margin-bottom:0.55rem;
                    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace;
                ">{label}</div>
                <div class="metric-icon">{icon}</div>
            </div>
            <div style="
                color:{text_color};
                font-size:2.15rem;
                line-height:1.02;
                font-weight:800;
                position:relative;
                font-family: Georgia, Times New Roman, serif;
            ">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37,99,235,0.07), transparent 28%),
                radial-gradient(circle at top right, rgba(124,58,237,0.07), transparent 24%),
                #F7F3EE;
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.2rem;
            max-width: 1240px;
        }
        html, body, [class*="css"]  {
            color: #1A1208;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        div[data-testid="stDataFrame"] {
            background: #FDFBF8;
            border: 1px solid #E2D9CE;
            border-radius: 18px;
            padding: 0.4rem;
            box-shadow: 0 10px 28px rgba(26, 18, 8, 0.05);
        }
        div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {
            border-radius: 14px;
            overflow: hidden;
        }
        div[data-testid="stDataFrame"] [role="gridcell"],
        div[data-testid="stDataFrame"] [role="columnheader"] {
            color: #1A1208 !important;
            background: #FDFBF8 !important;
            border-color: #EFE6DB !important;
        }
        .hero-shell {
            background:
                linear-gradient(180deg, rgba(253,251,248,0.98), rgba(253,251,248,0.96)),
                #FDFBF8;
            border: 1px solid rgba(226,217,206,0.95);
            border-radius: 26px;
            padding: 1.7rem 1.8rem 1.45rem 1.8rem;
            box-shadow: 0 18px 46px rgba(26, 18, 8, 0.06);
            margin-bottom: 1.2rem;
            position: relative;
            overflow: hidden;
        }
        .hero-shell:before {
            content: "";
            position: absolute;
            inset: 0 auto auto 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, #2563EB, #7C3AED);
        }
        .filter-shell {
            background: #FDFBF8;
            border: 1px solid #E2D9CE;
            border-radius: 18px;
            padding: 1rem 1rem 0.65rem 1rem;
            margin-top: 0.75rem;
        }
        .detail-shell {
            background: #FDFBF8;
            border: 1px solid #E2D9CE;
            border-radius: 22px;
            padding: 1.1rem;
            box-shadow: 0 14px 32px rgba(26, 18, 8, 0.05);
        }
        .context-shell {
            background:#FDFBF8;
            border:1px solid #E2D9CE;
            border-radius:18px;
            padding:0.95rem 1rem;
            box-shadow:0 10px 26px rgba(26,18,8,0.04);
        }
        .badge {
            display:inline-block;
            padding:0.28rem 0.62rem;
            border-radius:999px;
            font-size:0.78rem;
            font-weight:700;
            margin-right:0.35rem;
            margin-bottom:0.35rem;
        }
        .badge-muted {
            background:#F3ECE3;
            color:#6E5F50;
        }
        .badge-primary {
            background:linear-gradient(135deg, #E8F0FF, #F0EAFE);
            color:#1D4ED8;
        }
        .badge-accent {
            background:linear-gradient(135deg, #DCFCEB, #F0FDF7);
            color:#047857;
        }
        .badge-warn {
            background:linear-gradient(135deg, #FCE7B8, #FFF7E5);
            color:#B45309;
        }
        .subtle-note {
            color: #8A7A68;
            font-size: 0.94rem;
        }
        .metric-card-shell {
            position: relative;
            overflow: hidden;
            border: 1px solid;
            border-radius: 22px;
            padding: 1.05rem 1rem 1rem 1rem;
            min-height: 118px;
            box-shadow: 0 12px 32px rgba(26, 18, 8, 0.05);
            transition: transform 180ms ease, box-shadow 180ms ease;
        }
        .metric-card-shell:hover {
            transform: translateY(-2px);
            box-shadow: 0 18px 40px rgba(26, 18, 8, 0.08);
        }
        .metric-glow {
            position: absolute;
            inset: 0;
            opacity: 1;
            pointer-events: none;
        }
        .metric-icon {
            width: 2rem;
            height: 2rem;
            display:flex;
            align-items:center;
            justify-content:center;
            border-radius:999px;
            background: rgba(253,251,248,0.78);
            color:#5B4C3D;
            font-size:0.95rem;
            font-weight:800;
            border: 1px solid rgba(226,217,206,0.95);
            position: relative;
        }
        .lead-card {
            background: linear-gradient(180deg, rgba(253,251,248,0.99), rgba(251,247,242,0.96));
            border: 1px solid #E2D9CE;
            border-radius: 22px;
            padding: 1rem 1rem 0.95rem 1rem;
            box-shadow: 0 12px 30px rgba(26, 18, 8, 0.05);
            margin-bottom: 0.85rem;
            transition: transform 180ms ease, box-shadow 180ms ease;
        }
        .lead-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 18px 38px rgba(26, 18, 8, 0.08);
        }
        .lead-card-selected {
            border-color: #7C3AED;
            box-shadow: 0 0 0 1px rgba(124,58,237,0.28), 0 18px 38px rgba(26, 18, 8, 0.08);
            background: linear-gradient(180deg, rgba(248,244,255,0.98), rgba(253,251,248,0.96));
        }
        div.stButton > button[kind="tertiary"] {
            background: linear-gradient(180deg, rgba(253,251,248,0.99), rgba(251,247,242,0.96)) !important;
            border: 1px solid #E2D9CE !important;
            border-radius: 22px !important;
            color: #1A1208 !important;
            text-align: left !important;
            white-space: pre-wrap !important;
            line-height: 1.52 !important;
            min-height: 210px !important;
            padding: 1rem 1rem 0.95rem 1rem !important;
            box-shadow: 0 12px 30px rgba(26, 18, 8, 0.05) !important;
        }
        div.stButton > button[kind="tertiary"]:hover {
            border-color: #CDBEAE !important;
            box-shadow: 0 18px 38px rgba(26, 18, 8, 0.08) !important;
            transform: translateY(-2px);
        }
        .detail-section {
            background: linear-gradient(180deg, rgba(248,243,238,0.72), rgba(253,251,248,1));
            border: 1px solid #E7DED4;
            border-radius: 14px;
            padding: 0.8rem 0.85rem;
            margin-top: 0.8rem;
        }
        .chart-shell {
            background: linear-gradient(180deg, rgba(253,251,248,0.99), rgba(251,247,242,0.96));
            border: 1px solid #E2D9CE;
            border-radius: 24px;
            padding: 1rem 1rem 0.45rem 1rem;
            box-shadow: 0 12px 30px rgba(26, 18, 8, 0.05);
        }
        .section-kicker {
            display:inline-block;
            padding:0.22rem 0.55rem;
            border-radius:999px;
            background:linear-gradient(90deg, rgba(37,99,235,0.12), rgba(124,58,237,0.10));
            color:#4F46E5;
            font-size:0.74rem;
            font-weight:700;
            letter-spacing:0.14em;
            text-transform:uppercase;
            margin-bottom:0.4rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace;
        }
        .stTextArea textarea {
            background: #FDFBF8 !important;
            color: #1A1208 !important;
            border: 1px solid #E2D9CE !important;
            border-radius: 14px !important;
            line-height: 1.55 !important;
            box-shadow: inset 0 1px 2px rgba(26, 18, 8, 0.03) !important;
        }
        .stTextArea label, .stSelectbox label, .stSegmentedControl label {
            color: #1A1208 !important;
            font-weight: 600 !important;
        }
        .stSegmentedControl [data-baseweb="button-group"] {
            flex-wrap: nowrap !important;
        }
        .stSegmentedControl [data-baseweb="button-group"] button {
            white-space: nowrap !important;
            min-width: 78px !important;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 999px !important;
            border: 1px solid #D9CCBD !important;
            background: #FDFBF8 !important;
            color: #1A1208 !important;
            box-shadow: 0 6px 18px rgba(26, 18, 8, 0.04) !important;
            writing-mode: horizontal-tb !important;
        }
        .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563EB, #7C3AED) !important;
            color: #FFFFFF !important;
            border: none !important;
        }
        .stSegmentedControl [data-baseweb="button-group"] button[aria-pressed="true"] {
            background: linear-gradient(135deg, #2563EB, #7C3AED) !important;
            color: #FFFFFF !important;
            border-color: transparent !important;
        }
        .table-hint {
            color: #8A7A68;
            font-size: 0.9rem;
            margin-top: -0.35rem;
            margin-bottom: 0.55rem;
        }
        .empty-state {
            background: #FDFBF8;
            border: 1px dashed #D9CCBD;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            color: #8A7A68;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div style="margin:0.35rem 0 1rem 0;">
            <div style="font-size:1.36rem;font-weight:800;color:#1A1208;margin-bottom:0.2rem;font-family:Georgia, Times New Roman, serif;">{title}</div>
            <div style="font-size:0.97rem;color:#8A7A68;max-width:840px;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ensure_state() -> None:
    if "lead_seed" not in st.session_state:
        st.session_state.lead_seed = 42
    if "last_refreshed" not in st.session_state:
        st.session_state.last_refreshed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "selected_priorities" not in st.session_state:
        st.session_state.selected_priorities = []
    if "priority_widget" not in st.session_state:
        st.session_state.priority_widget = "All"
    if "selected_industry" not in st.session_state:
        st.session_state.selected_industry = "Select industry"
    if "selected_source" not in st.session_state:
        st.session_state.selected_source = "Select source"
    if "selected_country" not in st.session_state:
        st.session_state.selected_country = "Select country"
    if "industry_widget" not in st.session_state:
        st.session_state.industry_widget = st.session_state.selected_industry
    if "source_widget" not in st.session_state:
        st.session_state.source_widget = st.session_state.selected_source
    if "country_widget" not in st.session_state:
        st.session_state.country_widget = st.session_state.selected_country
    if "selected_lead_id" not in st.session_state:
        st.session_state.selected_lead_id = None
    if "show_copy_message_for_lead" not in st.session_state:
        st.session_state.show_copy_message_for_lead = None
    if "selected_chart" not in st.session_state:
        st.session_state.selected_chart = "Priority"
    if "reset_filters_pending" not in st.session_state:
        st.session_state.reset_filters_pending = False
    if "lead_explorer_version" not in st.session_state:
        st.session_state.lead_explorer_version = 0
    if "segment_explorer_version" not in st.session_state:
        st.session_state.segment_explorer_version = 0


def request_reset_filters() -> None:
    st.session_state.reset_filters_pending = True
    st.session_state.selected_priorities = []
    st.session_state.selected_lead_id = None
    st.session_state.show_copy_message_for_lead = None
    st.session_state.selected_chart = "Priority"
    st.session_state.lead_explorer_version += 1
    st.session_state.segment_explorer_version += 1


def refresh_data() -> None:
    st.session_state.lead_seed += 1
    st.session_state.last_refreshed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.selected_lead_id = None
    st.session_state.show_copy_message_for_lead = None
    st.session_state.lead_explorer_version += 1
    st.session_state.segment_explorer_version += 1


def get_active_lead_table() -> pd.DataFrame:
    return synthetic_leads(st.session_state.lead_seed)


def apply_context_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    if st.session_state.selected_priorities:
        filtered = filtered[filtered["priority"].isin(st.session_state.selected_priorities)]
    if st.session_state.selected_industry != "Select industry":
        filtered = filtered[filtered["industry"] == st.session_state.selected_industry]
    if st.session_state.selected_source != "Select source":
        filtered = filtered[filtered["source"] == st.session_state.selected_source]
    if st.session_state.selected_country != "Select country":
        filtered = filtered[filtered["country"] == st.session_state.selected_country]
    return filtered


def toggle_priority(priority: str) -> None:
    current = list(st.session_state.selected_priorities)
    if priority in current:
        current.remove(priority)
    else:
        current.append(priority)
    st.session_state.selected_priorities = current


def render_priority_pill(priority: str, count: int, tone: str) -> None:
    active = priority in st.session_state.selected_priorities
    label = f"{priority} · {count}"
    if active:
        label = f"Selected: {label}"
    if st.button(label, key=f"priority_pill_{priority}", width="stretch"):
        toggle_priority(priority)


def chart_dataframe(filtered_df: pd.DataFrame, chart_name: str) -> tuple[pd.DataFrame, str, str]:
    if chart_name == "Priority":
        chart_df = (
            filtered_df["priority"]
            .value_counts()
            .reindex(["High", "Medium", "Low"], fill_value=0)
            .rename_axis("priority")
            .to_frame("leads")
        )
        return chart_df, "Priority", "#7C3AED"
    if chart_name == "Industry":
        chart_df = filtered_df["industry"].value_counts().rename_axis("industry").to_frame("leads").head(10)
        return chart_df, "Industry", "#10B981"
    if chart_name == "Source":
        chart_df = filtered_df["source"].value_counts().rename_axis("source").to_frame("leads").head(10)
        return chart_df, "Source", "#2563EB"
    if chart_name == "Country":
        chart_df = filtered_df["country"].value_counts().rename_axis("country").to_frame("leads").head(10)
        return chart_df, "Country", "#F59E0B"

    score_bins = pd.cut(
        filtered_df["score"],
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        include_lowest=True,
        labels=["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
    )
    chart_df = score_bins.value_counts().sort_index().rename_axis("score_range").to_frame("leads")
    return chart_df, "Score range", "#F59E0B"


def company_intelligence(row: pd.Series) -> dict[str, str]:
    company_type = (
        "Enterprise-scale account"
        if float(row["company_size"]) >= 250
        else "Mid-market account"
        if float(row["company_size"]) >= 75
        else "Emerging growth account"
    )
    engagement_signal = (
        "High-intent activity: demo requested and strong recent engagement."
        if int(row["requested_demo"]) == 1
        else "Warm activity: multiple engagement signals with recent interest."
        if int(row["clicked_link"]) == 1 or float(row["engagement_score"]) >= 65
        else "Lighter activity: lower engagement and less urgency right now."
    )
    urgency_level = (
        "High urgency: follow up quickly while intent is fresh."
        if row["priority"] == "High"
        else "Moderate urgency: good nurture candidate with meaningful upside."
        if row["priority"] == "Medium"
        else "Lower urgency: keep visible, but not first in queue."
    )
    return {
        "company_type": company_type,
        "engagement_signal": engagement_signal,
        "urgency_level": urgency_level,
    }


def render_detail_panel(detail_row: pd.Series, clear_button_key: str) -> None:
    lead_id = str(detail_row["lead_id"])
    intel = company_intelligence(detail_row)
    detail_header_cols = st.columns([1, 0.22])
    with detail_header_cols[1]:
        if st.button("Clear selected lead", key=clear_button_key, width="stretch"):
            clear_selected_lead()
            st.rerun()

    info_col, message_col = st.columns([1.05, 1.15], gap="large")
    with info_col:
        linkedin_query = f"{detail_row['company_name']} {detail_row['role_title']} LinkedIn"
        linkedin_url = "https://www.linkedin.com/search/results/all/?keywords=" + quote(linkedin_query)
        mailto_subject = quote(str(detail_row["email_subject"]))
        mailto_body = quote(str(detail_row["outreach_message"]))
        mailto_url = f"mailto:{detail_row['mock_email']}?subject={mailto_subject}&body={mailto_body}"
        st.markdown(
            f"""
            <div class="detail-shell">
                <div style="font-size:1.7rem;font-weight:800;color:#1A1208;line-height:1.05;font-family:Georgia, Times New Roman, serif;">{detail_row['company_name']}</div>
                <div style="color:#8A7A68;margin-top:0.25rem;font-size:1rem;">{detail_row['role_title']} · {detail_row['industry']} · {detail_row['country']}</div>
                <div style="margin-top:0.8rem;">
                    <span class="badge badge-primary">Score {detail_row['score']:.3f}</span>
                    <span class="badge badge-accent">{detail_row['priority']}</span>
                    <span class="badge badge-warn">{detail_row['recommended_action']}</span>
                </div>
                <div class="detail-section">
                    <div style="color:#1A1208;font-weight:700;">Company Snapshot</div>
                    <div class="subtle-note" style="margin-top:0.35rem;">Mock email: <strong>{detail_row['mock_email']}</strong></div>
                    <div class="subtle-note" style="margin-top:0.3rem;">Industry: <strong>{detail_row['industry']}</strong> · Country: <strong>{detail_row['country']}</strong></div>
                    <div class="subtle-note" style="margin-top:0.3rem;">Recommended action: <strong>{detail_row['recommended_action']}</strong></div>
                    <div class="subtle-note" style="margin-top:0.45rem;line-height:1.6;">{detail_row['company_summary']}</div>
                </div>
                <div class="detail-section">
                    <div style="color:#1A1208;font-weight:700;">Why this lead</div>
                    <div class="subtle-note" style="margin-top:0.35rem;line-height:1.6;">{detail_row['why_this_lead']}</div>
                </div>
                <div class="detail-section">
                    <div style="color:#1A1208;font-weight:700;">Company Intelligence</div>
                    <div class="subtle-note" style="margin-top:0.35rem;"><strong>Company type:</strong> {intel['company_type']}</div>
                    <div class="subtle-note" style="margin-top:0.3rem;"><strong>Engagement signal:</strong> {intel['engagement_signal']}</div>
                    <div class="subtle-note" style="margin-top:0.3rem;"><strong>Urgency level:</strong> {intel['urgency_level']}</div>
                </div>
                <div class="detail-section">
                    <div style="color:#1A1208;font-weight:700;">Recent signals</div>
                    <div class="subtle-note" style="margin-top:0.35rem;line-height:1.6;">{detail_row['recent_signals']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        action_cols = st.columns(2)
        with action_cols[0]:
            st.link_button("Open LinkedIn Search", linkedin_url, width="stretch")
        with action_cols[1]:
            st.link_button("Draft Email", mailto_url, width="stretch")
        if st.button("Copy Outreach Message", key=f"copy_outreach_{lead_id}", width="stretch"):
            st.session_state.show_copy_message_for_lead = lead_id
    with message_col:
        st.markdown(
            """
            <div class="detail-shell">
                <div style="font-size:1rem;font-weight:800;color:#1A1208;font-family:Georgia, Times New Roman, serif;">Outreach Drafts</div>
                <div class="subtle-note" style="margin-top:0.25rem;">Drafted from the current lead context to support fast follow-up. These actions are safe links only.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.text_area(
            "Outreach email subject",
            value=detail_row["email_subject"],
            height=70,
            key=f"email_subject_{clear_button_key}_{lead_id}",
        )
        st.text_area(
            "Outreach message",
            value=detail_row["outreach_message"],
            height=170,
            key=f"outreach_message_{clear_button_key}_{lead_id}",
        )
        st.text_area(
            "LinkedIn message",
            value=detail_row["linkedin_message"],
            height=140,
            key=f"linkedin_message_{clear_button_key}_{lead_id}",
        )
        if st.session_state.show_copy_message_for_lead == lead_id:
            st.text_area(
                "Copy-ready outreach message",
                value=detail_row["outreach_message"],
                height=170,
                key=f"copy_ready_outreach_{clear_button_key}_{lead_id}",
            )


def clear_selected_lead() -> None:
    st.session_state.selected_lead_id = None
    st.session_state.show_copy_message_for_lead = None
    st.session_state.lead_explorer_version += 1
    st.session_state.segment_explorer_version += 1


def apply_filter_state_defaults(
    industry_options: list[str], source_options: list[str], country_options: list[str]
) -> None:
    placeholder_industry = "Select industry"
    placeholder_source = "Select source"
    placeholder_country = "Select country"

    if st.session_state.reset_filters_pending:
        st.session_state.selected_priorities = []
        st.session_state.priority_widget = "All"
        st.session_state.selected_industry = placeholder_industry
        st.session_state.selected_source = placeholder_source
        st.session_state.selected_country = placeholder_country
        st.session_state.industry_widget = placeholder_industry
        st.session_state.source_widget = placeholder_source
        st.session_state.country_widget = placeholder_country
        st.session_state.selected_chart = "Priority"
        st.session_state.reset_filters_pending = False

    current_priority = st.session_state.get("priority_widget", "All")
    current_industry = st.session_state.get("industry_widget", st.session_state.selected_industry)
    current_source = st.session_state.get("source_widget", st.session_state.selected_source)
    current_country = st.session_state.get("country_widget", st.session_state.selected_country)

    if current_priority not in ["All", "High", "Medium", "Low"]:
        current_priority = "All"
        st.session_state.priority_widget = current_priority

    if current_industry not in industry_options:
        current_industry = placeholder_industry
        st.session_state.industry_widget = current_industry
    if current_source not in source_options:
        current_source = placeholder_source
        st.session_state.source_widget = current_source
    if current_country not in country_options:
        current_country = placeholder_country
        st.session_state.country_widget = current_country

    st.session_state.selected_priorities = [] if current_priority == "All" else [current_priority]
    st.session_state.selected_industry = current_industry
    st.session_state.selected_source = current_source
    st.session_state.selected_country = current_country


def main() -> None:
    ensure_state()
    inject_styles()

    input_df = get_active_lead_table()
    model = train_model()
    scored_df = score_leads(input_df, model)
    scored_df = add_explanations(scored_df)
    scored_df = add_mock_enrichment(scored_df)

    industry_options = ["Select industry"] + sorted(scored_df["industry"].dropna().unique().tolist())
    source_options = ["Select source"] + sorted(scored_df["source"].dropna().unique().tolist())
    country_options = ["Select country"] + sorted(scored_df["country"].dropna().unique().tolist())
    apply_filter_state_defaults(industry_options, source_options, country_options)

    filtered_df = apply_context_filters(scored_df)
    header_cols = st.columns([1.15, 0.34], gap="large")
    with header_cols[0]:
        st.markdown(
            f"""
            <div class="hero-shell">
                <div style="display:inline-block;padding:0.28rem 0.68rem;border-radius:999px;background:linear-gradient(90deg, rgba(37,99,235,0.12), rgba(124,58,237,0.10));color:#4F46E5;font-size:0.8rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.8rem;font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace;">
                    SignalDeck AI
                </div>
                <div style="font-size:2.8rem;font-weight:800;line-height:0.98;color:#1A1208;margin-bottom:0.2rem;font-family:Georgia, Times New Roman, serif;">
                    Find who to talk to next
                </div>
                <div style="font-size:1rem;line-height:1.6;color:#8A7A68;max-width:760px;margin-top:0.55rem;">
                    AI-powered GTM lead intelligence dashboard
                </div>
                <div style="width:120px;height:4px;border-radius:999px;background:linear-gradient(135deg, #2563EB, #7C3AED);margin:0.95rem 0 0.85rem 0;"></div>
                <div class="subtle-note" style="max-width:860px;">
                    Built on a simulated lead dataset for fast exploration. Filters, KPIs, the main chart, top leads, detail context, and downloads all respect the same active view.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_cols[1]:
        st.markdown('<div class="detail-shell" style="padding:1rem 1rem 0.9rem 1rem;">', unsafe_allow_html=True)
        st.download_button(
            label="Download current view",
            data=to_csv_download(filtered_df),
            file_name="current_view_leads.csv",
            mime="text/csv",
            width="stretch",
        )
        st.markdown(
            f"""
            <div class="subtle-note" style="margin-top:0.8rem;">
                Last refreshed: <strong>{st.session_state.last_refreshed}</strong><br>
                Active rows: <strong>{len(filtered_df)}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    render_section_header(
        "Filter Bar",
        "Refine the active lead pool. Every selection updates the full dashboard immediately.",
    )
    st.markdown('<div class="filter-shell">', unsafe_allow_html=True)
    filter_cols = st.columns([1.35, 1, 1, 1, 0.65], gap="medium")
    with filter_cols[0]:
        st.segmented_control(
            "Priority",
            options=["All", "High", "Medium", "Low"],
            selection_mode="single",
            key="priority_widget",
        )
    with filter_cols[1]:
        st.selectbox("Industry", options=industry_options, key="industry_widget")
    with filter_cols[2]:
        st.selectbox("Source", options=source_options, key="source_widget")
    with filter_cols[3]:
        st.selectbox("Country", options=country_options, key="country_widget")
    with filter_cols[4]:
        st.markdown("<div style='height: 1.95rem;'></div>", unsafe_allow_html=True)
        if st.button("Reset filters", key="reset_view_main", width="stretch"):
            request_reset_filters()
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    priority_choice = st.session_state.priority_widget
    if priority_choice in (None, "All"):
        st.session_state.selected_priorities = []
    else:
        st.session_state.selected_priorities = [str(priority_choice)]
    st.session_state.selected_industry = st.session_state.industry_widget
    st.session_state.selected_source = st.session_state.source_widget
    st.session_state.selected_country = st.session_state.country_widget

    filtered_df = apply_context_filters(scored_df)

    total_leads = len(filtered_df)
    high_priority = int((filtered_df["priority"] == "High").sum())
    average_score = float(filtered_df["score"].mean()) if total_leads else 0.0
    contact_now_count = int((filtered_df["recommended_action"] == "Contact now").sum())
    high_priority_share = (high_priority / total_leads * 100.0) if total_leads else 0.0

    current_view_bits = []
    if st.session_state.selected_priorities:
        current_view_bits.append(f"Priority: {', '.join(st.session_state.selected_priorities)}")
    if st.session_state.selected_industry != "Select industry":
        current_view_bits.append(f"Industry: {st.session_state.selected_industry}")
    if st.session_state.selected_source != "Select source":
        current_view_bits.append(f"Source: {st.session_state.selected_source}")
    if st.session_state.selected_country != "Select country":
        current_view_bits.append(f"Country: {st.session_state.selected_country}")
    current_view_text = " · ".join(current_view_bits) if current_view_bits else "Full dataset"

    st.markdown(
        f"""
        <div class="context-shell" style="margin:0.2rem 0 1rem 0;">
            <div style="font-size:0.76rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:#8A7A68;margin-bottom:0.32rem;font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace;">Current View</div>
            <div style="font-size:0.98rem;color:#1A1208;font-weight:600;">{current_view_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_header(
        "Dashboard Summary",
        "A fast read on the current GTM slice you are looking at.",
    )
    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Total leads", str(total_leads), tone="accent")
    with metric_cols[1]:
        render_metric_card("Contact now", str(contact_now_count), tone="high")
    with metric_cols[2]:
        render_metric_card("Average score", f"{average_score:.3f}", tone="neutral")
    with metric_cols[3]:
        render_metric_card("High priority %", f"{high_priority_share:.0f}%", tone="medium")

    if filtered_df.empty:
        st.warning("No leads match the current filters. Use Reset filters to return to the full dataset.")
        if st.button("Reset filters", key="reset_view_empty", width="content"):
            request_reset_filters()
            st.rerun()
        return

    render_section_header(
        "Main Chart",
        "One flexible view for understanding how the filtered lead pool is distributed.",
    )
    st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
    chart_name = st.selectbox(
        "View data by",
        options=[
            "Priority",
            "Industry",
            "Source",
            "Country",
            "Score distribution",
        ],
        key="selected_chart",
    )
    chart_df, x_label, chart_color = chart_dataframe(filtered_df, chart_name)
    st.bar_chart(chart_df, x_label=x_label, y_label="Leads", color=chart_color)
    st.markdown('</div>', unsafe_allow_html=True)

    render_section_header(
        "Top Leads Today",
        "Your action queue for the current view — highest-value leads and immediate outreach opportunities.",
    )
    top_leads = (
        filtered_df.sort_values(["score", "engagement_score"], ascending=[False, False])
        .head(6)[["lead_id", "company_name", "role_title", "industry", "country", "score", "priority", "recommended_action", "why_this_lead"]]
        .reset_index(drop=True)
    )
    render_top_lead_cards(top_leads, "No top leads are available in this view.")

    if st.session_state.selected_lead_id not in filtered_df["lead_id"].astype(str).tolist():
        st.session_state.selected_lead_id = None
        st.session_state.show_copy_message_for_lead = None

    render_section_header(
        "Lead Detail Panel",
        "A single focused company profile that updates as soon as you choose a lead from the cards above.",
    )
    if st.session_state.selected_lead_id is not None:
        selected_row = filtered_df.loc[filtered_df["lead_id"].astype(str) == str(st.session_state.selected_lead_id)].iloc[0]
        render_detail_panel(selected_row, "clear_selected_lead_main")
    else:
        st.markdown(
            """
            <div class="empty-state" style="margin-top:0.15rem;">
                Choose a lead from Top Leads Today to open company context, recent signals, and outreach drafts here.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
