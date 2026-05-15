import json
import os
import re
from typing import Any, Dict, List


DEFAULT_SUGGESTIONS = [
    "How do I predict a crop?",
    "Explain model insights",
    "Tell me about rice",
]


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


# ── Fallback Q&A knowledge base ─────────────────────────────────────────────
_FALLBACK_QA: List[Dict[str, str]] = []


def _load_fallback_qa() -> List[Dict[str, str]]:
    """Load pre-built Q&A pairs from JSON file (cached after first load)."""
    global _FALLBACK_QA
    if _FALLBACK_QA:
        return _FALLBACK_QA

    qa_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fallback_qa.json")
    try:
        with open(qa_path, "r", encoding="utf-8") as f:
            _FALLBACK_QA = json.load(f)
    except Exception:
        _FALLBACK_QA = []
    return _FALLBACK_QA


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for matching."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _word_set(text: str) -> set:
    """Return set of meaningful words (length > 1) from normalized text."""
    stop = {"a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
            "i", "me", "my", "we", "you", "it", "in", "on", "of", "to", "for",
            "and", "or", "can", "will", "what", "how", "which", "this", "that",
            "be", "has", "have", "had", "not", "with", "from", "about", "please"}
    return {w for w in _normalize(text).split() if len(w) > 1 and w not in stop}


def _match_fallback(user_message: str) -> str:
    """Find the best matching Q&A pair using keyword overlap scoring."""
    qa_pairs = _load_fallback_qa()
    if not qa_pairs:
        return ""

    user_norm = _normalize(user_message)
    user_words = _word_set(user_message)

    best_score = 0.0
    best_answer = ""

    for pair in qa_pairs:
        q = pair.get("q", "")
        a = pair.get("a", "")
        q_norm = _normalize(q)
        q_words = _word_set(q)

        # Exact match → highest priority
        if user_norm == q_norm:
            return a

        # Substring match only for multi-word questions (avoid "hi" matching "high humidity")
        if len(q_norm.split()) >= 3 and q_norm in user_norm:
            return a
        if len(user_norm.split()) >= 3 and user_norm in q_norm:
            return a

        # Keyword overlap score
        if not q_words:
            continue
        overlap = user_words & q_words
        if not overlap:
            continue

        # Score = overlap relative to the question's keyword count
        score = len(overlap) / len(q_words)
        # Bonus if overlap covers most of user's words too
        user_coverage = len(overlap) / max(len(user_words), 1)
        combined = score * 0.6 + user_coverage * 0.4

        if combined > best_score:
            best_score = combined
            best_answer = a

    # Require at least 40% match confidence
    if best_score >= 0.4:
        return best_answer
    return ""


def _translate_text(text: str, target_lang: str) -> str:
    if not text:
        return ""
    target_lang = (target_lang or "en").split("-")[0].lower()
    if target_lang == "en":
        return text

    try:
        from deep_translator import GoogleTranslator  # type: ignore

        target = TRANSLATOR_LANG_MAP.get(target_lang, "en")
        return GoogleTranslator(source="auto", target=target).translate(text)
    except Exception:
        return text


def _build_project_context(crop_info: Dict[str, Dict[str, str]], crop_library: Dict[str, Dict[str, str]]) -> str:
    crop_lines = []
    for crop_name in sorted(crop_info.keys()):
        info = crop_info.get(crop_name, {})
        lib = crop_library.get(crop_name, {})
        crop_lines.append(
            (
                f"- {crop_name}: desc={info.get('desc', 'NA')}; "
                f"soil={lib.get('soil', 'NA')}; ph={lib.get('ph', 'NA')}; "
                f"rainfall={lib.get('rain', 'NA')}; temp={lib.get('temp', 'NA')}"
            )
        )

    return (
        "Project: Precision Crop Recommender Flask app.\n"
        "Core features:\n"
        "- Predict Crop page: user enters N, P, K, temperature, humidity, pH, rainfall.\n"
        "- /predict endpoint returns top crop + top-5 confidence.\n"
        "- Crop Library page contains crop descriptions and ranges for some crops.\n"
        "- Model Insights page includes metrics (accuracy, precision, recall, F1, top-3, log loss).\n"
        "- This assistant should answer only project and agriculture context questions.\n"
        "Known crop entries:\n"
        + "\n".join(crop_lines)
    )


def _build_prompt_template(project_context: str, user_question: str) -> str:
    return f"""
You are the in-app Farm Assistant for this project.

Follow these rules strictly:
1. Answer based only on the project context below and user's question.
2. If exact data is unavailable, say clearly: "I don't have that exact value in this project data".
3. Keep answers practical, concise, and farmer-friendly.
4. Prefer plain text, max 6 sentences.
5. Do not invent API endpoints, metrics, or crop ranges.

PROJECT CONTEXT:
{project_context}

USER QUESTION:
{user_question}
""".strip()


def _load_gemini_client(api_key: str):
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Gemini SDK not installed. Install dependency: google-generativeai"
        ) from exc

    genai.configure(api_key=api_key)
    return genai


def _generate_with_gemini(prompt: str) -> str:
    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    model_name = (os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash").strip()

    if not api_key:
        raise RuntimeError("Gemini API key not configured.")

    try:
        genai = _load_gemini_client(api_key)
        model = genai.GenerativeModel(model_name=model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 350,
            },
        )

        text = (getattr(response, "text", "") or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty response.")
        return text
    except Exception as e:
        raise RuntimeError(f"AI service temporarily unavailable: {str(e)[:200]}")


def generate_chatbot_reply(message: str, crop_info: Dict[str, Dict[str, str]], crop_library: Dict[str, Dict[str, str]], lang: str = "en") -> Dict[str, Any]:
    user_message = (message or "").strip()
    if not user_message:
        return {
            "reply": _translate_text("Please type a question so I can help.", lang),
            "suggestions": [_translate_text(item, lang) for item in DEFAULT_SUGGESTIONS],
        }

    user_message_en = _translate_text(user_message, "en") if lang != "en" else user_message

    # 1) Try Gemini API first
    project_context = _build_project_context(crop_info, crop_library)
    prompt = _build_prompt_template(project_context, user_message_en)

    try:
        answer = _generate_with_gemini(prompt)
        final_answer = _translate_text(answer, lang) if lang != "en" else answer
        return {
            "reply": final_answer,
            "suggestions": [_translate_text(item, lang) for item in DEFAULT_SUGGESTIONS],
        }
    except Exception:
        pass  # Fall through to offline fallback

    # 2) Fallback: match against pre-built Q&A knowledge base
    fallback_answer = _match_fallback(user_message_en)
    if fallback_answer:
        final_answer = _translate_text(fallback_answer, lang) if lang != "en" else fallback_answer
        return {
            "reply": final_answer,
            "suggestions": [_translate_text(item, lang) for item in DEFAULT_SUGGESTIONS],
        }

    # 3) Generic fallback if nothing matched
    generic = (
        "I'm sorry, I couldn't find an answer for that. "
        "Try asking about a specific crop (e.g., 'tell me about rice'), "
        "soil requirements, or how to use the prediction tool."
    )
    return {
        "reply": _translate_text(generic, lang),
        "suggestions": [
            _translate_text("How do I predict a crop?", lang),
            _translate_text("What crops are supported?", lang),
            _translate_text("Tell me about rice", lang),
        ],
    }
