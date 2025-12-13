import os, io, zipfile, base64, tempfile, json, datetime, re
from pathlib import Path
from .logging_config import get_logger
from openai import OpenAI

logger = get_logger(__name__)

# ---------------------------
# Guardrail helpers (defined before UI usage)
# ---------------------------
def moderate_text(text: str, model: str) -> tuple:
    """
    Run a moderation check on text. Returns (flagged: bool, reasons: List[str]).
    Robustly normalizes 'categories' from dict-like or attribute-based (pydantic) responses.
    """
    try:
        if not text or not text.strip():
            logger.warning("moderate_text called with empty text")
            return False, []

        logger.info("Running moderation check on text (length: %d)", len(text))
        # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        client = OpenAI(api_key="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")
        resp = client.moderations.create(model="omni-moderation-latest", input=text)
        
        logger.debug("Moderation API response received")
        # normalize to the first result object/dict
        if isinstance(resp, dict):
            res0 = resp.get("results", [resp])[0]
        else:
            res0 = getattr(resp, "results", [resp])[0]

        # extract flagged and raw categories
        if isinstance(res0, dict):
            flagged = bool(res0.get("flagged", False))
            categories_raw = res0.get("categories", {}) or {}
        else:
            flagged = bool(getattr(res0, "flagged", False))
            categories_raw = getattr(res0, "categories", {}) or {}

        # normalize categories into a plain dict of booleans
        def normalize_bool_map(obj):
            if obj is None:
                return {}
            if isinstance(obj, dict):
                return {k: bool(v) for k, v in obj.items()}
            if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
                try:
                    return {k: bool(v) for k, v in obj.dict().items()}
                except Exception:
                    pass
            if hasattr(obj, "__dict__"):
                try:
                    return {k: bool(v) for k, v in vars(obj).items()}
                except Exception:
                    pass
            out = {}
            for k in dir(obj):
                if k.startswith("_"):
                    continue
                try:
                    v = getattr(obj, k)
                except Exception:
                    continue
                if isinstance(v, bool):
                    out[k] = v
            return out

        categories = normalize_bool_map(categories_raw)
        reasons = [k for k, v in categories.items() if v]
        
        if reasons:
            logger.warning("Text flagged by moderation. Reasons: %s", reasons)
        else:
            logger.info("Text passed moderation check")

        return flagged, reasons

    except Exception as e:
        logger.error(f"Moderation check failed: {e}", exc_info=True)
        return False, []
    
# ---------------------------
# Additional Guardrail helpers (PII, validation, redaction, audit)
# ---------------------------
PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{3}[-.\s]?){2}\d{4}\b")
}

def validate_user_query(query: str, max_len: int = 1500):
    if not query or not query.strip():
        return False, "Empty query."
    if len(query) > max_len:
        return False, f"Query too long (>{max_len} chars)."
    banned = ["make a bomb", "how to kill", "bypass security"]
    lowered = query.lower()
    for b in banned:
        if b in lowered:
            return False, "Query contains disallowed content."
    return True, ""

def check_pii_in_text(text: str):
    hits = {}
    if not text:
        return hits
    for k, pat in PII_PATTERNS.items():
        found = pat.findall(text)
        if found:
            hits[k] = found
    return hits

def redact_pii(text: str):
    if not text:
        return text
    out = text
    for pat in PII_PATTERNS.values():
        out = pat.sub("[REDACTED]", out)
    return out

def enforce_output_schema(text: str, max_len: int = 4000):
    if not text:
        return False, "Empty model output."
    if len(text) > max_len:
        return False, "Model output too long."
    refuse_keywords = ["how to make a bomb", "instructions to kill"]
    lower = text.lower()
    for k in refuse_keywords:
        if k in lower:
            return False, "Model output violated safety rules."
    return True, ""

AUDIT_LOG = Path("./audit.log")
def log_audit_entry(event_type: str, payload: dict):
    payload = dict(payload)
    for k, v in payload.items():
        if isinstance(v, str):
            payload[k] = redact_pii(v)
    entry = {"ts": datetime.datetime.utcnow().isoformat() + "Z", "event": event_type, "payload": payload}
    try:
        if AUDIT_LOG.exists():
            AUDIT_LOG.write_text(AUDIT_LOG.read_text() + json.dumps(entry) + "\n")
        else:
            AUDIT_LOG.write_text(json.dumps(entry) + "\n")
    except Exception:
        with AUDIT_LOG.open("a", encoding="utf8") as f:
            f.write(json.dumps(entry) + "\n")

def display_safe_text(text: str):
    safe = redact_pii(text)
    if len(safe) > 1000:
        safe = safe[:1000] + "\n\n...[truncated]"
    f"out_{hash(safe)%10000}"