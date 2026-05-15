"""
flask_app.py — Crop Recommendation System Web Application (Flask)

Run:
    python app/flask_app.py

Visit: http://127.0.0.1:5000
"""

import os
import smtplib
import json
from datetime import datetime, timedelta
from email.message import EmailMessage
import secrets
import re
from threading import Lock
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify, abort, send_from_directory
from flask import redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, top_k_accuracy_score
from werkzeug.security import check_password_hash, generate_password_hash

# Conditional import to handle both local and production environments
try:
    from app.chatbot import generate_chatbot_reply
except ImportError:
    from chatbot import generate_chatbot_reply

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TMPL_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
DATA_PATH  = os.path.join(BASE_DIR, "data", "Crop_recommendation.csv")
SCHEMES_DATA_PATH = os.path.join(BASE_DIR, "data", "agriculture_schemes_multilingual.json")

if load_dotenv is not None:
    load_dotenv(os.path.join(BASE_DIR, ".env"))

app = Flask(__name__, template_folder=TMPL_DIR)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")
SQLITE_DB_PATH = os.path.join(BASE_DIR, "app", "auth.db").replace("\\", "/")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{SQLITE_DB_PATH}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Add SSL options for Postgres connections (required by Render)
engine_options = {}
if "postgresql" in (os.environ.get("DATABASE_URL") or ""):
    engine_options = {
        "connect_args": {
            "sslmode": "require",
            "connect_timeout": 10,
        }
    }

db = SQLAlchemy(app, engine_options=engine_options)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to continue."

# ── Load model once at startup ────────────────────────────────────────────────
model = None
le = None
MODEL_LOAD_ERROR = None
INSIGHT_METRICS_CACHE = None
try:
    model = joblib.load(os.path.join(MODELS_DIR, "crop_model.pkl"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
except Exception as exc:
    MODEL_LOAD_ERROR = str(exc)
    print(f"[MODEL LOAD ERROR] {MODEL_LOAD_ERROR}")

# ── Crop details ──────────────────────────────────────────────────────────────
CROP_INFO = {
    "apple":       {"emoji": "🍎", "color": "#e53935",
                    "desc": "Apple grows best in cool climates with well-drained, loamy soil and requires cold winters for dormancy."},
    "banana":      {"emoji": "🍌", "color": "#f9a825",
                    "desc": "Banana thrives in tropical climates with high humidity and consistent rainfall throughout the year."},
    "blackgram":   {"emoji": "🫘", "color": "#4e342e",
                    "desc": "Blackgram is a legume suited to warm, semi-arid conditions and fixes nitrogen in soil."},
    "chickpea":    {"emoji": "🟡", "color": "#f57f17",
                    "desc": "Chickpea grows in dry, cool conditions and is highly drought-tolerant, ideal for Rabi season."},
    "coconut":     {"emoji": "🥥", "color": "#6d4c41",
                    "desc": "Coconut palm requires a tropical climate with sandy, well-drained soil and coastal humid winds."},
    "coffee":      {"emoji": "☕", "color": "#4e342e",
                    "desc": "Coffee needs a subtropical climate, high humidity, moderate rainfall, and well-drained acidic soil."},
    "cotton":      {"emoji": "🌿", "color": "#c8e6c9",
                    "desc": "Cotton requires long warm summers, moderate rainfall, deep fertile soil, and full sunshine."},
    "grapes":      {"emoji": "🍇", "color": "#6a1b9a",
                    "desc": "Grapes thrive in Mediterranean climate with well-drained sandy soil and hot, dry summers."},
    "jute":        {"emoji": "🌾", "color": "#8d6e63",
                    "desc": "Jute grows best in warm, humid climates with loamy, alluvial soil and high monsoonal rainfall."},
    "kidneybeans": {"emoji": "🫘", "color": "#b71c1c",
                    "desc": "Kidney beans prefer well-drained, fertile soil with moderate temperatures between 15–25°C."},
    "lentil":      {"emoji": "🟤", "color": "#795548",
                    "desc": "Lentil is a cool-season pulse crop highly tolerant to drought, suited to semi-arid regions."},
    "maize":       {"emoji": "🌽", "color": "#f9a825",
                    "desc": "Maize requires warm weather, fertile soil with good drainage, and moderate to high rainfall."},
    "mango":       {"emoji": "🥭", "color": "#ff8f00",
                    "desc": "Mango needs a tropical climate, deep well-drained soil, and seasonal dry spells for flowering."},
    "mothbeans":   {"emoji": "🫘", "color": "#827717",
                    "desc": "Moth beans are highly drought-resistant and perfectly suited to the hot arid zones of India."},
    "mungbean":    {"emoji": "🟢", "color": "#2e7d32",
                    "desc": "Mung bean is a short-season legume suited to warm, humid regions and improves soil fertility."},
    "muskmelon":   {"emoji": "🍈", "color": "#f9a825",
                    "desc": "Muskmelon needs warm temperatures, dry weather during fruiting, and light sandy, well-drained soil."},
    "orange":      {"emoji": "🍊", "color": "#e65100",
                    "desc": "Orange grows in subtropical climates with well-drained fertile soil and warm, sunny conditions."},
    "papaya":      {"emoji": "🍑", "color": "#ff6f00",
                    "desc": "Papaya is a fast-growing tropical fruit requiring rich, moist, loamy soil and full sun."},
    "pigeonpeas":  {"emoji": "🫘", "color": "#33691e",
                    "desc": "Pigeon pea is drought-tolerant, nitrogen-fixing, and suits the semi-arid tropical regions."},
    "pomegranate": {"emoji": "🍎", "color": "#c62828",
                    "desc": "Pomegranate thrives in semi-arid to arid regions and tolerates alkaline, well-drained soils."},
    "rice":        {"emoji": "🍚", "color": "#1565c0",
                    "desc": "Rice requires high humidity, standing water during growth, clayey or loamy soil, and warm temperatures."},
    "watermelon":  {"emoji": "🍉", "color": "#388e3c",
                    "desc": "Watermelon needs hot, sunny weather, a long frost-free season, and sandy well-drained soil."},
}

CROP_LIBRARY = {
    "rice": {"soil": "Clayey or loamy", "ph": "5.5 - 7.0", "rain": "150 - 300 mm", "temp": "20 - 30 C"},
    "maize": {"soil": "Well-drained loam", "ph": "5.5 - 7.5", "rain": "60 - 120 mm", "temp": "18 - 27 C"},
    "cotton": {"soil": "Deep black soil", "ph": "5.8 - 8.0", "rain": "50 - 100 mm", "temp": "21 - 30 C"},
    "mango": {"soil": "Deep loamy soil", "ph": "5.5 - 7.5", "rain": "75 - 250 mm", "temp": "24 - 30 C"},
    "coffee": {"soil": "Acidic loam", "ph": "5.0 - 6.5", "rain": "150 - 250 mm", "temp": "18 - 24 C"},
}

OTP_EXP_MINUTES = 10
OTP_MAX_ATTEMPTS = 5
OTP_RESEND_COOLDOWN_SECONDS = 30
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
PHONE_RE = re.compile(r"^[0-9+\-\s]{10,20}$")

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "gu": "Gujarati",
    "mr": "Marathi",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "bn": "Bengali",
    "or": "Odia",
    "as": "Assamese",
    "brx": "Bodo",
    "doi": "Dogri",
    "gom": "Konkani",
    "ks": "Kashmiri",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "ne": "Nepali",
    "sa": "Sanskrit",
    "sat": "Santali",
    "sd": "Sindhi",
    "ur": "Urdu",
}

# Some regional language codes are not directly supported by GoogleTranslator.
TRANSLATOR_LANG_MAP = {
    "en": "en",
    "hi": "hi",
    "gu": "gu",
    "mr": "mr",
    "pa": "pa",
    "ta": "ta",
    "te": "te",
    "kn": "kn",
    "bn": "bn",
    "or": "or",
    "as": "as",
    "brx": "hi",
    "doi": "hi",
    "gom": "mr",
    "ks": "ur",
    "mai": "hi",
    "ml": "ml",
    "mni": "bn",
    "ne": "ne",
    "sa": "hi",
    "sat": "hi",
    "sd": "sd",
    "ur": "ur",
}

INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands",
    "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir", "Ladakh",
    "Lakshadweep", "Puducherry", "All India",
]

_TRANSLATION_CACHE = {}
_TRANSLATION_LOCK = Lock()
_SCHEMES_CACHE = None
_SCHEMES_OPTIONS_CACHE = None
_UI_TRANSLATIONS = None


def _load_ui_translations() -> dict:
    """Load pre-built UI translations from JSON file."""
    global _UI_TRANSLATIONS
    if _UI_TRANSLATIONS is not None:
        return _UI_TRANSLATIONS
    
    try:
        ui_trans_path = os.path.join(BASE_DIR, "data", "ui_translations.json")
        if os.path.exists(ui_trans_path):
            with open(ui_trans_path, 'r', encoding='utf-8') as f:
                _UI_TRANSLATIONS = json.load(f)
        else:
            _UI_TRANSLATIONS = {"en": {}}  # Fallback
    except Exception:
        _UI_TRANSLATIONS = {"en": {}}
    
    return _UI_TRANSLATIONS


def _current_lang() -> str:
    lang = str(session.get("lang", "en")).split("-")[0].lower()
    return lang if lang in SUPPORTED_LANGUAGES else "en"


def _translate_text(text: str, lang: str) -> str:
    if not text:
        return ""
    if lang == "en":
        return text

    # First try pre-built UI translations
    ui_trans = _load_ui_translations()
    if lang in ui_trans and text in ui_trans[lang]:
        return ui_trans[lang][text]

    # Fall back to runtime translation only if key not in UI translations
    cache_key = (lang, text)
    with _TRANSLATION_LOCK:
        cached = _TRANSLATION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    translated = text
    translation_ok = False
    try:
        from deep_translator import GoogleTranslator  # type: ignore

        target = TRANSLATOR_LANG_MAP.get(lang, "en")
        translated = GoogleTranslator(source="auto", target=target).translate(text)
        translation_ok = True
    except Exception:
        translated = text

    # Cache only successful translations (or explicit English passthrough),
    # so transient translator errors do not permanently pin labels to English.
    if translation_ok or lang == "en":
        with _TRANSLATION_LOCK:
            _TRANSLATION_CACHE[cache_key] = translated
    return translated


def tr(text: str) -> str:
    return _translate_text(text, _current_lang())


# Register tr as a Jinja2 global so it's available in all templates
app.jinja_env.globals.update(tr=tr)


def _load_schemes() -> list:
    global _SCHEMES_CACHE
    if _SCHEMES_CACHE is not None:
        return _SCHEMES_CACHE

    if not os.path.exists(SCHEMES_DATA_PATH):
        _SCHEMES_CACHE = [
            {
                "en": {
                    "scheme_name": "PM-KISAN",
                    "description": "Income support of Rs 6000 per year for eligible farmer families.",
                    "benefits": "Direct benefit transfer in three installments.",
                    "eligibility": "Small and marginal farmer families.",
                    "category": "Income Support",
                    "farmer_type": "small",
                    "income_level": "low",
                    "land_size": "small",
                    "states": ["All India"],
                },
                "url": "https://pmkisan.gov.in/",
            },
            {
                "en": {
                    "scheme_name": "PMFBY",
                    "description": "Crop insurance scheme for yield loss due to natural calamities.",
                    "benefits": "Insurance coverage and financial support in case of crop failure.",
                    "eligibility": "All farmers including tenant farmers.",
                    "category": "Insurance",
                    "farmer_type": "all",
                    "income_level": "all",
                    "land_size": "all",
                    "states": ["All India"],
                },
                "url": "https://pmfby.gov.in/",
            },
        ]
        return _SCHEMES_CACHE

    try:
        with open(SCHEMES_DATA_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            _SCHEMES_CACHE = data if isinstance(data, list) else []
    except Exception:
        _SCHEMES_CACHE = []
    return _SCHEMES_CACHE


def _scheme_lang_block(scheme: dict, lang: str) -> dict:
    block = scheme.get(lang)
    if isinstance(block, dict):
        return block
    return scheme.get("en", {}) if isinstance(scheme.get("en", {}), dict) else {}


def _scheme_text(block: dict, *keys: str) -> str:
    for key in keys:
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _scheme_states_from_text(text: str) -> list:
    text_l = text.lower()
    states = [state for state in INDIAN_STATES if state != "All India" and state.lower() in text_l]
    return states or ["All India"]


def _infer_farmer_type(text: str) -> str:
    text_l = (text or "").lower()
    if any(k in text_l for k in ["women", "woman", "female"]):
        return "Women Farmers"
    if any(k in text_l for k in ["small and marginal", "small farmers", "marginal farmers"]):
        return "Small & Marginal Farmers"
    if any(k in text_l for k in ["tenant", "sharecropper", "lessee"]):
        return "Tenant Farmers"
    if any(k in text_l for k in ["sc", "st", "scheduled caste", "scheduled tribe", "tribal"]):
        return "SC/ST Farmers"
    if any(k in text_l for k in ["youth", "young", "startup"]):
        return "Young Farmers"
    if any(k in text_l for k in ["fpo", "producer organization", "self help group", "shg", "cooperative"]):
        return "FPO / SHG"
    return "All Farmers"


def _infer_income_level(text: str) -> str:
    text_l = (text or "").lower()
    if any(k in text_l for k in ["bpl", "below poverty", "low income", "weaker section"]):
        return "Low Income"
    if any(k in text_l for k in ["middle income", "annual income", "income up to", "income less than"]):
        return "Medium Income"
    if any(k in text_l for k in ["all farmers", "all eligible", "all categories"]):
        return "All Income Levels"
    return "All Income Levels"


def _infer_land_size(text: str) -> str:
    text_l = (text or "").lower()
    if any(k in text_l for k in ["landless"]):
        return "Landless"
    if any(k in text_l for k in ["small and marginal", "up to 2 hectare", "upto 2 hectare", "below 2 hectare", "< 2 hectare"]):
        return "Small Holdings"
    if any(k in text_l for k in ["2 to 10 hectare", "2-10 hectare", "medium holdings"]):
        return "Medium Holdings"
    if any(k in text_l for k in ["> 10 hectare", "above 10 hectare", "large holdings"]):
        return "Large Holdings"
    return "All Land Sizes"


def _build_scheme_options(schemes: list) -> dict:
    global _SCHEMES_OPTIONS_CACHE
    if _SCHEMES_OPTIONS_CACHE is not None:
        return _SCHEMES_OPTIONS_CACHE

    states = {"All India"}
    categories = set()
    farmer_types = set()
    income_levels = set()
    land_sizes = set()

    for scheme in schemes:
        en = _scheme_lang_block(scheme, "en")
        name = _scheme_text(en, "scheme_name", "name")
        desc = _scheme_text(en, "description", "short_description")
        benefits = _scheme_text(en, "benefits")
        eligibility = _scheme_text(en, "eligibility")
        cat = _scheme_text(en, "category")
        farmer_type = _scheme_text(en, "farmer_type")
        income_level = _scheme_text(en, "income_level")
        land_size = _scheme_text(en, "land_size")
        profile_text = f"{name} {desc} {benefits} {eligibility}"

        if cat:
            for item in cat.split(","):
                cleaned = item.strip()
                if cleaned:
                    categories.add(cleaned)
        farmer_types.add(farmer_type.strip() if farmer_type else _infer_farmer_type(profile_text))
        income_levels.add(income_level.strip() if income_level else _infer_income_level(profile_text))
        land_sizes.add(land_size.strip() if land_size else _infer_land_size(profile_text))

        merged = f"{name} {desc}"
        for state in _scheme_states_from_text(merged):
            states.add(state)

    _SCHEMES_OPTIONS_CACHE = {
        "states": sorted(states, key=lambda s: (s != "All India", s)),
        "categories": sorted(categories),
        "farmer_types": sorted(farmer_types),
        "income_levels": sorted(income_levels),
        "land_sizes": sorted(land_sizes),
    }
    return _SCHEMES_OPTIONS_CACHE


def _filtered_schemes(state: str, category: str, keyword: str, farmer_type: str, income_level: str, land_size: str, lang: str, page: int, per_page: int) -> dict:
    schemes = _load_schemes()
    state_l = state.lower().strip()
    category_l = category.lower().strip()
    keyword_l = keyword.lower().strip()
    farmer_type_l = farmer_type.lower().strip()
    income_level_l = income_level.lower().strip()
    land_size_l = land_size.lower().strip()

    out = []
    for scheme in schemes:
        en = _scheme_lang_block(scheme, "en")
        localized = _scheme_lang_block(scheme, lang)

        en_name = _scheme_text(en, "scheme_name", "name")
        en_desc = _scheme_text(en, "description", "short_description")
        en_benefits = _scheme_text(en, "benefits")
        en_eligibility = _scheme_text(en, "eligibility")
        en_category = _scheme_text(en, "category")
        profile_text = f"{en_name} {en_desc} {en_benefits} {en_eligibility}"
        en_farmer_type = _scheme_text(en, "farmer_type") or _infer_farmer_type(profile_text)
        en_income_level = _scheme_text(en, "income_level") or _infer_income_level(profile_text)
        en_land_size = _scheme_text(en, "land_size") or _infer_land_size(profile_text)

        all_text = f"{en_name} {en_desc} {en_benefits} {en_eligibility} {en_category}".lower()
        inferred_states = _scheme_states_from_text(all_text)

        if state_l and state_l != "all india" and state_l not in " ".join(s.lower() for s in inferred_states):
            continue
        if category_l and category_l not in en_category.lower():
            continue
        if keyword_l and keyword_l not in all_text:
            continue
        if farmer_type_l and farmer_type_l not in en_farmer_type.lower():
            continue
        if income_level_l and income_level_l not in en_income_level.lower():
            continue
        if land_size_l and land_size_l not in en_land_size.lower():
            continue

        name = _scheme_text(localized, "scheme_name", "name") or _translate_text(en_name, lang)
        desc = _scheme_text(localized, "description", "short_description") or _translate_text(en_desc, lang)
        benefits = _scheme_text(localized, "benefits") or _translate_text(en_benefits, lang)
        eligibility = _scheme_text(localized, "eligibility") or _translate_text(en_eligibility, lang)
        category_value = _scheme_text(localized, "category") or _translate_text(en_category, lang)

        out.append(
            {
                "scheme_name": name,
                "description": desc,
                "short_description": desc[:170] + "..." if len(desc) > 170 else desc,
                "benefits": benefits,
                "eligibility": eligibility,
                "categories": [item.strip() for item in category_value.split(",") if item.strip()] or [tr("Agriculture")],
                "states": inferred_states,
                "url": str(scheme.get("url") or ""),
            }
        )

    total = len(out)
    start = (page - 1) * per_page
    end = start + per_page
    return {
        "results": out[start:end],
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page if per_page else 1,
    }


def _smtp_configured() -> bool:
    return bool(
        os.environ.get("SMTP_HOST")
        and os.environ.get("SMTP_USER")
        and os.environ.get("SMTP_PASSWORD")
    )


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    farmer_name = db.Column(db.String(120), nullable=True)
    land_acres = db.Column(db.Float, nullable=True)
    years_farming = db.Column(db.Integer, nullable=True)
    phone_number = db.Column(db.String(24), nullable=True)
    is_verified = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.utcnow())
    last_login_at = db.Column(db.DateTime, nullable=True)


class OTPChallenge(db.Model):
    __tablename__ = "otp_challenges"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    purpose = db.Column(db.String(32), nullable=False, index=True)
    otp_hash = db.Column(db.String(255), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False, index=True)
    attempts = db.Column(db.Integer, nullable=False, default=0)
    is_used = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.utcnow())


@login_manager.user_loader
def load_user(user_id: str):
    try:
        return db.session.get(User, int(user_id))
    except Exception:
        return None


def _now_utc() -> datetime:
    # Keep timestamps naive UTC because SQLite DateTime values are returned as naive.
    return datetime.utcnow()


def _password_valid(password: str) -> tuple:
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not any(ch.isupper() for ch in password):
        return False, "Password must contain at least one uppercase letter."
    if not any(ch.islower() for ch in password):
        return False, "Password must contain at least one lowercase letter."
    if not any(ch.isdigit() for ch in password):
        return False, "Password must contain at least one digit."
    if not any(ch in "!@#$%^&*()-_=+[]{};:,.?/" for ch in password):
        return False, "Password must contain at least one special character."
    return True, ""


def _email_valid(email: str) -> bool:
    return bool(EMAIL_RE.match(email or ""))


def _phone_valid(phone_number: str) -> bool:
    if not phone_number:
        return False
    compact = phone_number.replace(" ", "")
    digit_count = sum(ch.isdigit() for ch in compact)
    return bool(PHONE_RE.match(phone_number)) and 10 <= digit_count <= 15


def _ensure_users_table_columns() -> None:
    """Ensure all required columns exist on the users table.
    
    This is a best-effort function that works with SQLite but gracefully
    skips on Postgres since Postgres will auto-create columns via db.create_all().
    """
    required_columns = {
        "farmer_name": "VARCHAR(120)",
        "land_acres": "FLOAT",
        "years_farming": "INTEGER",
        "phone_number": "VARCHAR(24)",
    }

    try:
        # Try SQLite approach first
        existing = db.session.execute(text("PRAGMA table_info(users)")).mappings().all()
        existing_names = {row.get("name") for row in existing}

        for col_name, col_type in required_columns.items():
            if col_name not in existing_names:
                db.session.execute(text(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}"))

        db.session.commit()
    except Exception:
        # If PRAGMA doesn't work (Postgres, etc.), skip this check.
        # The ORM schema should already be set up via db.create_all().
        pass


def _generate_otp_code() -> str:
    return "".join(secrets.choice("0123456789") for _ in range(6))


def _send_otp_email(recipient: str, otp_code: str, purpose: str) -> bool:
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASSWORD")
    smtp_from = os.environ.get("SMTP_FROM", smtp_user or "no-reply@example.com")

    subject = "Your OTP for Precision Crop Recommender"
    purpose_text = "account verification" if purpose in {"signup_verify", "email_verify"} else "login verification"
    body = (
        f"Your OTP for {purpose_text} is: {otp_code}\n"
        f"This OTP expires in {OTP_EXP_MINUTES} minutes."
    )

    if not _smtp_configured():
        print(f"[DEV OTP] {recipient} -> {otp_code} ({purpose})")
        return True

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = recipient
    msg.set_content(body)

    try:
        # Use a 10 second timeout to prevent hanging
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.starttls(timeout=10)
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True
    except Exception as exc:
        print(f"[OTP EMAIL ERROR] {exc}")
        # Return True anyway so signup doesn't fail - email is non-critical
        return True


def _store_dev_otp(otp_code: str) -> None:
    if not _smtp_configured():
        session["dev_otp_code"] = otp_code
    else:
        session.pop("dev_otp_code", None)


def _create_otp_challenge(user_id: int, purpose: str) -> str:
    # Invalidate previous active OTPs for the same purpose.
    OTPChallenge.query.filter_by(user_id=user_id, purpose=purpose, is_used=False).update({"is_used": True})
    otp_code = _generate_otp_code()
    challenge = OTPChallenge(
        user_id=user_id,
        purpose=purpose,
        otp_hash=generate_password_hash(otp_code),
        expires_at=_now_utc() + timedelta(minutes=OTP_EXP_MINUTES),
        attempts=0,
        is_used=False,
    )
    db.session.add(challenge)
    db.session.commit()
    return otp_code


def _verify_otp(user_id: int, purpose: str, otp_code: str) -> tuple:
    # DEV MODE: Accept any 6-digit number as a valid OTP
    if otp_code and len(otp_code) == 6 and otp_code.isdigit():
        challenge = (
            OTPChallenge.query
            .filter_by(user_id=user_id, purpose=purpose, is_used=False)
            .order_by(OTPChallenge.created_at.desc())
            .first()
        )
        if challenge:
            challenge.is_used = True
            db.session.commit()
        return True, "OTP verified successfully."
    return False, "Please enter a valid 6-digit OTP."


with app.app_context():
    db.create_all()
    _ensure_users_table_columns()


def _insight_images() -> list:
    """Return available insight image filenames from models directory."""
    wanted = [
        "feature_importance.png",
        "model_comparison.png",
        "confusion_matrix.png",
        "confusion_matrix_hybrid.png",
        "shap_feature_importance.png",
        "shap_summary_plot.png",
        "correlation_heatmap.png",
        "crop_distribution.png",
    ]
    return [name for name in wanted if os.path.exists(os.path.join(MODELS_DIR, name))]


def _insight_caption(filename: str) -> str:
    if filename == "feature_importance.png":
        return "Ranks raw input features by average importance contribution in the baseline tree model."
    if filename == "model_comparison.png":
        return "Compares baseline model test accuracies to help justify the selected production model."
    if filename == "confusion_matrix.png":
        return "Shows class-by-class prediction correctness for the best baseline model."
    if filename == "confusion_matrix_hybrid.png":
        return "Shows class-by-class prediction correctness for the hybrid stacked ensemble."
    if filename == "shap_feature_importance.png":
        return "Global SHAP impact summary using mean absolute SHAP values across features."
    if filename == "shap_summary_plot.png":
        return "Beeswarm SHAP plot showing direction and magnitude of feature influence per sample."
    if filename == "correlation_heatmap.png":
        return "Highlights linear relationships between numeric agronomic features in the dataset."
    if filename == "crop_distribution.png":
        return "Shows label balance across all crop classes in the training dataset."
    return "Supporting analytical visualization from the training workflow."


def _insight_artifacts() -> list:
    artifacts = []
    for filename in _insight_images():
        artifacts.append(
            {
                "filename": filename,
                "title": filename.replace("_", " ").replace(".png", "").title(),
                "caption": _insight_caption(filename),
            }
        )
    return artifacts


def _insight_metrics() -> tuple:
    global INSIGHT_METRICS_CACHE

    if INSIGHT_METRICS_CACHE is not None:
        return INSIGHT_METRICS_CACHE

    if model is None or le is None:
        return [], MODEL_LOAD_ERROR or "Model artifacts are unavailable on this server."

    if not os.path.exists(DATA_PATH):
        return [], "Dataset file not found for metrics calculation."

    try:
        df = pd.read_csv(DATA_PATH)
        features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        for col in features + ["label"]:
            if col not in df.columns:
                return [], f"Missing required column in dataset: {col}"

        X = df[features].to_numpy(dtype=float)
        y_true = le.transform(df["label"].astype(str).to_numpy())
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)

        metric_items = [
            {"label": "Accuracy", "value": f"{accuracy_score(y_true, y_pred) * 100:.2f}%", "tone": "good"},
            {"label": "Macro Precision", "value": f"{precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}", "tone": "good"},
            {"label": "Macro Recall", "value": f"{recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}", "tone": "good"},
            {"label": "Macro F1", "value": f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}", "tone": "good"},
            {"label": "Top-3 Accuracy", "value": f"{top_k_accuracy_score(y_true, y_proba, k=3, labels=np.arange(len(le.classes_))) * 100:.2f}%", "tone": "info"},
            {"label": "Log Loss", "value": f"{log_loss(y_true, y_proba, labels=np.arange(len(le.classes_))):.4f}", "tone": "warn"},
            {"label": "Samples", "value": f"{len(df):,}", "tone": "info"},
            {"label": "Classes", "value": f"{len(le.classes_)}", "tone": "info"},
        ]
        INSIGHT_METRICS_CACHE = (metric_items, "")
        return INSIGHT_METRICS_CACHE
    except Exception as exc:
        return [], f"Could not compute metrics: {exc}"


@app.context_processor
def inject_globals():
    return {
        "app_name": "Precision Crop Recommender",
        "year": 2026,
        "tr": tr,
        "supported_languages": SUPPORTED_LANGUAGES,
        "current_lang": _current_lang(),
        "chatbot_force_open": bool(session.pop("chatbot_force_open_once", False)),
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
@login_required
def index():
    return render_template("home.html")

@app.route("/recommend")
@login_required
def recommend_page():
    return render_template("predict.html")


@app.route("/crops")
@login_required
def crops_page():
    merged = []
    for crop_name, info in sorted(CROP_INFO.items(), key=lambda x: x[0]):
        defaults = {"soil": "Loamy", "ph": "5.5 - 7.5", "rain": "60 - 200 mm", "temp": "18 - 32 C"}
        defaults.update(CROP_LIBRARY.get(crop_name, {}))
        merged.append({"name": crop_name, **info, **defaults})
    return render_template("crops.html", crops=merged)


@app.route("/insights")
@login_required
def insights_page():
    metrics, metrics_error = _insight_metrics()
    return render_template("insights.html", artifacts=_insight_artifacts(), metrics=metrics, metrics_error=metrics_error)


@app.route("/about")
@login_required
def about_page():
    return render_template("about.html")


@app.route("/schemes")
@login_required
def schemes_page():
    return render_template("schemes.html")


@app.route("/api/schemes/options", methods=["GET"])
@login_required
def schemes_options():
    return jsonify({"success": True, **_build_scheme_options(_load_schemes())})


@app.route("/api/schemes", methods=["GET"])
@login_required
def schemes_api():
    language = str(request.args.get("language") or _current_lang()).split("-")[0].lower()
    if language not in SUPPORTED_LANGUAGES:
        language = "en"

    state = str(request.args.get("state") or "")
    category = str(request.args.get("category") or "")
    keyword = str(request.args.get("keyword") or "")
    farmer_type = str(request.args.get("farmer_type") or "")
    income_level = str(request.args.get("income_level") or "")
    land_size = str(request.args.get("land_size") or "")
    page = max(1, int(request.args.get("page", 1)))
    per_page = min(1000, max(1, int(request.args.get("per_page", 100))))

    return jsonify(
        {
            "success": True,
            **_filtered_schemes(
                state=state,
                category=category,
                keyword=keyword,
                farmer_type=farmer_type,
                income_level=income_level,
                land_size=land_size,
                lang=language,
                page=page,
                per_page=per_page,
            ),
        }
    )


@app.route("/set-language", methods=["POST"])
def set_language():
    payload = request.get_json(silent=True) or {}
    lang = str(payload.get("lang") or request.form.get("lang") or "en").split("-")[0].lower()
    if lang not in SUPPORTED_LANGUAGES:
        return jsonify({"success": False, "error": "Unsupported language"}), 400
    session["lang"] = lang
    with _TRANSLATION_LOCK:
        _TRANSLATION_CACHE.clear()
    return jsonify({"success": True, "lang": lang})


@app.route("/register", methods=["GET", "POST"])
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("recommend_page"))

    if request.method == "POST":
        farmer_name = (request.form.get("farmer_name") or "").strip()
        land_acres_raw = (request.form.get("land_acres") or "").strip()
        years_farming_raw = (request.form.get("years_farming") or "").strip()
        phone_number = (request.form.get("phone_number") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        if (
            not farmer_name
            or not land_acres_raw
            or not years_farming_raw
            or not phone_number
            or not email
            or not password
            or not confirm_password
        ):
            flash("All fields are required.", "error")
            return render_template("signup.html")

        if len(farmer_name) < 2:
            flash("Farmer name must be at least 2 characters.", "error")
            return render_template("signup.html")

        try:
            land_acres = float(land_acres_raw)
        except ValueError:
            flash("Number of acres must be a valid number.", "error")
            return render_template("signup.html")

        if land_acres <= 0 or land_acres > 100000:
            flash("Number of acres must be between 0 and 100000.", "error")
            return render_template("signup.html")

        if not years_farming_raw.isdigit():
            flash("Years of farming must be a whole number.", "error")
            return render_template("signup.html")

        years_farming = int(years_farming_raw)
        if years_farming < 0 or years_farming > 80:
            flash("Years of farming must be between 0 and 80.", "error")
            return render_template("signup.html")

        if not _phone_valid(phone_number):
            flash("Please enter a valid phone number (10-15 digits).", "error")
            return render_template("signup.html")

        if not _email_valid(email):
            flash("Please enter a valid email address.", "error")
            return render_template("signup.html")

        if password != confirm_password:
            flash("Password and confirm password do not match.", "error")
            return render_template("signup.html")

        valid, msg = _password_valid(password)
        if not valid:
            flash(msg, "error")
            return render_template("signup.html")

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            if existing_user.is_verified:
                flash("Account already exists. Please log in.", "error")
                return redirect(url_for("login"))

            existing_user.farmer_name = farmer_name
            existing_user.land_acres = land_acres
            existing_user.years_farming = years_farming
            existing_user.phone_number = phone_number
            existing_user.password_hash = generate_password_hash(password)
            db.session.commit()

            otp_code = _create_otp_challenge(existing_user.id, "signup_verify")
            _store_dev_otp(otp_code)
            sent = _send_otp_email(existing_user.email, otp_code, "signup_verify")
            if not sent:
                flash("Account exists but OTP could not be sent. Please try again.", "error")
                return render_template("signup.html")

            session["pending_user_id"] = existing_user.id
            session["pending_purpose"] = "signup_verify"
            session["last_otp_sent_at"] = int(_now_utc().timestamp())
            flash("Account exists but is not verified. OTP sent again.", "success")
            return redirect(url_for("verify_otp"))

        user = User(
            farmer_name=farmer_name,
            land_acres=land_acres,
            years_farming=years_farming,
            phone_number=phone_number,
            email=email,
            password_hash=generate_password_hash(password),
            is_verified=False,
        )
        db.session.add(user)
        db.session.commit()

        otp_code = _create_otp_challenge(user.id, "signup_verify")
        _store_dev_otp(otp_code)
        sent = _send_otp_email(user.email, otp_code, "signup_verify")
        if not sent:
            OTPChallenge.query.filter_by(user_id=user.id).delete()
            db.session.delete(user)
            db.session.commit()
            flash("Could not send OTP email right now. Please try again.", "error")
            return render_template("signup.html")

        session["pending_user_id"] = user.id
        session["pending_purpose"] = "signup_verify"
        session["last_otp_sent_at"] = int(_now_utc().timestamp())
        flash("OTP sent to your email. Enter it to activate your account.", "success")
        return redirect(url_for("verify_otp"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("recommend_page"))

    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        # Developer convenience: allow default accounts for quick access
        ADMIN_EMAILS = {"admin@gmail.com", "test@example.com", "jitendrakumaryadav2003@gmail.com"}
        if email in ADMIN_EMAILS:
            user = User.query.filter_by(email=email).first()
            if user is None:
                user = User(
                    farmer_name="Admin User",
                    land_acres=1.0,
                    years_farming=0,
                    phone_number="0000000000",
                    email=email,
                    password_hash=generate_password_hash("admin"),
                    is_verified=True,
                )
                db.session.add(user)
                db.session.commit()

            user.is_verified = True
            user.last_login_at = _now_utc()
            db.session.commit()
            login_user(user)
            session["chatbot_force_open_once"] = True
            flash("Login successful.", "success")
            next_url = request.args.get("next")
            if next_url:
                return redirect(next_url)
            return redirect(url_for("recommend_page"))

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid email or password.", "error")
            return render_template("login.html")

        if user.is_verified:
            user.last_login_at = _now_utc()
            db.session.commit()
            login_user(user)
            session["chatbot_force_open_once"] = True
            flash("Login successful.", "success")
            next_url = request.args.get("next")
            if next_url:
                return redirect(next_url)
            return redirect(url_for("recommend_page"))

        # Unverified users must complete email OTP verification.
        purpose = "signup_verify"
        otp_code = _create_otp_challenge(user.id, purpose)
        _store_dev_otp(otp_code)
        sent = _send_otp_email(user.email, otp_code, purpose)
        if not sent:
            flash("Could not send OTP email right now. Please try again.", "error")
            return render_template("login.html")

        session["pending_user_id"] = user.id
        session["pending_purpose"] = purpose
        session["last_otp_sent_at"] = int(_now_utc().timestamp())
        flash("Your account is not verified yet. OTP sent to complete verification.", "success")
        return redirect(url_for("verify_otp"))

    return render_template("login.html")


@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():
    pending_user_id = session.get("pending_user_id")
    pending_purpose = session.get("pending_purpose")
    dev_otp_code = session.get("dev_otp_code") if not _smtp_configured() else None

    if not pending_user_id or not pending_purpose:
        flash("No OTP verification in progress.", "error")
        return redirect(url_for("login"))

    user = db.session.get(User, int(pending_user_id))
    if user is None:
        session.pop("pending_user_id", None)
        session.pop("pending_purpose", None)
        session.pop("last_otp_sent_at", None)
        flash("Account not found. Please sign up again.", "error")
        return redirect(url_for("signup"))

    if request.method == "POST":
        otp_code = (request.form.get("otp") or "").strip()
        # Allow a fixed developer OTP override as requested
        if otp_code == "875621":
            ok = True
            msg = "Developer OTP accepted."
        else:
            if not otp_code.isdigit() or len(otp_code) != 6:
                flash("OTP must be a 6-digit number.", "error")
                return render_template("verify_otp.html", email=user.email, dev_otp_code=dev_otp_code)

            ok, msg = _verify_otp(user.id, pending_purpose, otp_code)
        if not ok:
            flash(msg, "error")
            return render_template("verify_otp.html", email=user.email, dev_otp_code=dev_otp_code)

        if pending_purpose in {"signup_verify", "email_verify"}:
            user.is_verified = True

        user.last_login_at = _now_utc()
        db.session.commit()

        session.pop("pending_user_id", None)
        session.pop("pending_purpose", None)
        session.pop("last_otp_sent_at", None)
        session.pop("dev_otp_code", None)

        login_user(user)
        session["chatbot_force_open_once"] = True
        flash("Verification successful. Welcome back!", "success")
        return redirect(url_for("recommend_page"))

    return render_template("verify_otp.html", email=user.email, dev_otp_code=dev_otp_code)


@app.route("/resend-otp", methods=["POST"])
def resend_otp():
    pending_user_id = session.get("pending_user_id")
    pending_purpose = session.get("pending_purpose")
    if not pending_user_id or not pending_purpose:
        flash("No OTP verification in progress.", "error")
        return redirect(url_for("login"))

    now_ts = int(_now_utc().timestamp())
    last_sent_ts = int(session.get("last_otp_sent_at", 0))
    if now_ts - last_sent_ts < OTP_RESEND_COOLDOWN_SECONDS:
        wait_for = OTP_RESEND_COOLDOWN_SECONDS - (now_ts - last_sent_ts)
        flash(f"Please wait {wait_for} seconds before requesting another OTP.", "error")
        return redirect(url_for("verify_otp"))

    user = db.session.get(User, int(pending_user_id))
    if user is None:
        flash("Account not found.", "error")
        return redirect(url_for("signup"))

    otp_code = _create_otp_challenge(user.id, pending_purpose)
    sent = _send_otp_email(user.email, otp_code, pending_purpose)
    if not sent:
        flash("Could not resend OTP email right now. Please try again.", "error")
        return redirect(url_for("verify_otp"))
    session["last_otp_sent_at"] = now_ts
    flash("A new OTP has been sent.", "success")
    return redirect(url_for("verify_otp"))


@app.route("/logout", methods=["GET", "POST"])
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


@app.route("/artifacts/<path:filename>")
@login_required
def model_artifact(filename: str):
    if not filename.endswith(".png"):
        abort(404)
    full_path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(full_path):
        abort(404)
    return send_from_directory(MODELS_DIR, filename)


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        if model is None or le is None:
            err = MODEL_LOAD_ERROR or "Model artifacts are unavailable on this server."
            return jsonify({"success": False, "error": err}), 503

        data = request.get_json()
        features = [
            float(data["N"]),
            float(data["P"]),
            float(data["K"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["ph"]),
            float(data["rainfall"]),
        ]
        arr  = np.array([features])
        pred = model.predict(arr)[0]
        crop = le.inverse_transform([pred])[0]

        # Top-5 probabilities
        proba = model.predict_proba(arr)[0]
        top5  = sorted(
            [{"crop": le.classes_[i], "prob": round(float(p) * 100, 2)} for i, p in enumerate(proba)],
            key=lambda x: -x["prob"]
        )[:5]

        info = CROP_INFO.get(crop, {"emoji": "🌱", "color": "#2e7d32", "desc": "A great crop for your conditions."})

        return jsonify({
            "success": True,
            "crop":    crop,
            "emoji":   info["emoji"],
            "color":   info["color"],
            "desc":    info["desc"],
            "top5":    top5,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/chatbot/message", methods=["POST"])
def chatbot_message():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    lang = str(payload.get("lang") or _current_lang()).split("-")[0].lower()

    if len(message) > 500:
        return jsonify({"success": False, "error": "Message is too long. Use 500 characters or less."}), 400

    bot_response = generate_chatbot_reply(message, CROP_INFO, CROP_LIBRARY, lang=lang)
    return jsonify({"success": True, **bot_response})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    smtp_state = "configured" if _smtp_configured() else "NOT configured (DEV OTP mode)"
    print("=" * 55)
    print("  Crop Recommendation System")
    print(f"  SMTP: {smtp_state}")
    print(f"  Open browser: http://127.0.0.1:{port}")
    print("=" * 55)
    app.run(debug=True, host="127.0.0.1", port=port)
