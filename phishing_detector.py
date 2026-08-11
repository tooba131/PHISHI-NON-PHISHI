import json
import os
import pickle
import re
import urllib.parse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "saved models" / "resnet1d_se_20260724_062248.keras"
TOKENIZER_PATH = ROOT / "saved models" / "tokenizer_20260724_062248.pkl"
CONFIG_PATH = ROOT / "saved models" / "config_20260724_062248.json"

MAX_LEN = 250
DEFAULT_THRESHOLD = 0.5
OPTIMAL_THRESHOLD = 0.3082025349140167

_SCHEME_RE = re.compile(r"^(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)")
_PROTOCOL_RE = re.compile(r"^(?:https?://|http://)")
_WWW_RE = re.compile(r"^www\.", re.IGNORECASE)
_DOMAIN_PATH_RE = re.compile(r"^(?P<domain>[^/]+)(?P<path>/.*)?$")
_SHORTENER_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "adf.ly",
    "bit.do",
    "cutt.ly",
    "v.gd",
    "j.mp",
}

_TAG_RE = re.compile(r"<([A-Z0-9_]+):(\d+)>")
_TAG_CHAR_MAP = {
    "IPV4": "α",
    "IPV6": "β",
    "MAC_ADDRESS": "γ",
    "DATE": "δ",
    "NUMERIC_ID": "ε",
    "HEX_ID": "ζ",
    "UUID_FORMAT": "η",
    "JWT_FORMAT": "θ",
    "HASH_FORMAT": "λ",
    "ETH_WALLET": "μ",
    "BTC_WALLET": "π",
    "EMAIL_MATCH": "ρ",
    "EMAIL_MISMATCH": "σ",
    "BASE64_EXTERNAL": "τ",
    "BASE64_FORMAT": "φ",
    "SESSION_ID": "ψ",
    "TOKEN": "ω",
    "OTP_CODE": "Δ",
    "OAUTH_PARAM": "Θ",
    "OAUTH_CLIENT": "Λ",
    "FILE_EXEC": "Ξ",
    "FILE_MACRO": "Π",
    "FILE_ID": "Σ",
    "FILE_NAME": "Φ",
    "REF_EXTERNAL": "Ψ",
    "REF_ENCODED": "Ω",
    "REF_INTERNAL": "∇",
    "REF_DOMAIN": "ℵ",
    "REF_OTHER": "ℐ",
    "TIMESTAMP": "ℜ",
    "IP_PARAM": "℘",
    "ID": "℮",
    "EMPTY": "∅",
    "DEFAULT": "∞",
}
_FIX_LEN_TAGS = frozenset({"IPV4", "IPV6", "MAC_ADDRESS", "DATE", "TIMESTAMP", "ETH_WALLET", "BTC_WALLET"})
_MAX_TAG_REPEAT = 512

_HOST_RE = re.compile(r"^(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://|//)?(?P<host>\[[^\]]+\]|[^/?#:]+)(?P<rest>.*)$")
_OBF_DEC_RE = re.compile(r"^\d{8,10}$")
_OBF_HEX_RE = re.compile(r"^0[xX][0-9a-fA-F]{6,8}$")
_OBF_OCT_RE = re.compile(r"^(?:0[0-7]+\.){3}0[0-7]+$")
_SEG_END = r"(?=[&#/?]|$)"
_QS_END = r"(?=[&#]|$)"
_RANDOM_VALUE = (
    r"("
    r"\d{4,}"
    r"|(?=[^\s&#]*\d)[a-fA-F0-9]{8,}"
    r"|(?=[^\s&#]*[a-zA-Z])(?=[^\s&#]*\d)[a-zA-Z0-9\-]{8,}"
    r")"
)
_IPV4_RE = r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
_IPV6_CANDIDATE_RE = re.compile(r"(?<![A-Za-z0-9])\[?((?:[0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4})\]?(?![A-Za-z0-9])")
_VALUE_BASED_RULES = {
    "<IPV4>": re.compile(rf"\b{_IPV4_RE}\b"),
    "<MAC_ADDRESS>": re.compile(r"(\b)((?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2})(?=\b)"),
    "<DATE>": re.compile(r"(\b|/)((?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01]))(?=\b|/|$)"),
    "<NUMERIC_ID>": re.compile(r"(/)(\d{6,})(?=/|[?#]|$)"),
    "<HEX_ID>": re.compile(r"(/)([\da-fA-F]{15,})(?=/|[?#]|$)"),
    "<ETH_WALLET>": re.compile(r"(\b)(0x[\da-fA-F]{40})(?=\b)"),
    "<BTC_WALLET>": re.compile(r"([?&/=])(bc1[a-zA-HJ-NP-Z0-9]{25,39}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})" + _SEG_END),
    "<UUID_FORMAT>": re.compile(r"([?&][^=]+=|/)([\da-fA-F]{8}-[\da-fA-F]{4}-[\da-fA-F]{4}-[\da-fA-F]{4}-[\da-fA-F]{12})" + _SEG_END),
    "<JWT_FORMAT>": re.compile(r"([?&][^=]+=|/)(eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)" + _SEG_END),
    "<HASH_FORMAT>": re.compile(r"([?&][^=]+=|/)([\da-fA-F]{32}|[\da-fA-F]{40}|[\da-fA-F]{64})" + _SEG_END),
    "<EMAIL>": re.compile(r"([?&][^=]+=)([a-zA-Z0-9_.%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})" + _QS_END),
    "<BASE64_FORMAT>": re.compile(r"([?&][^=]+=)([a-zA-Z0-9_+/\-]{20,}={0,2})" + _QS_END),
}
_NAME_BASED_RULES = {
    "<SESSION_ID>": re.compile(r"([?&](?:session|sid|sessionid|PHPSESSID|JSESSIONID)=)" + _RANDOM_VALUE + _QS_END, re.IGNORECASE),
    "<TOKEN>": re.compile(r"([?&](?:token|access_token|auth|api_key|key|secret|bearer)=)" + _RANDOM_VALUE + _QS_END, re.IGNORECASE),
    "<OTP_CODE>": re.compile(r"([?&](?:otp|pin|passcode|verification_code|confirm_code|verif_code|2fa|mfa|one_time_password|auth_code)=)(\d{4,8})" + _QS_END, re.IGNORECASE),
    "<OAUTH_PARAM>": re.compile(r"([?&](?:code|state|nonce|sig|signature)=)" + _RANDOM_VALUE + _QS_END, re.IGNORECASE),
    "<OAUTH_CLIENT>": re.compile(r"([?&](?:client_id|app_id|consumer_key)=)" + _RANDOM_VALUE + _QS_END, re.IGNORECASE),
    "<TIMESTAMP>": re.compile(r"([?&](?:timestamp|ts|time|date|t|_t)=)(\d{8,15}|(?:19|20)\d{2}[-/.T](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])(?:[T ]\d{2}:\d{2}(?::\d{2})?)?)" + _QS_END, re.IGNORECASE),
    "<IP_PARAM>": re.compile(r"([?&](?:ip|ip_address|remote_addr|client_ip)=)(localhost|" + _IPV4_RE + r"|(?:[a-fA-F0-9]{1,4}:){2,7}[a-fA-F0-9]{0,4}|\d{7,10})" + _QS_END, re.IGNORECASE),
    "<ID>": re.compile(r"([?&](?:id|uid|[a-z_]*_id)=)(\d{4,}|(?=[^\s&#]*\d)[a-fA-F0-9]{10,})" + _QS_END, re.IGNORECASE),
    "<REF>": re.compile(r"([?&](?:ref|reference|referrer|returnUrl|next|redirect|redirect_uri|url|return|goto|dest|destination)=)([^&#]*)", re.IGNORECASE),
    "<FILE_PARAM>": re.compile(r"([?&](?:file|doc|document|download|dl|attachment|attach|asset)=)([^&#]*)", re.IGNORECASE),
}
_VALUE_ORDER = [
    "<MAC_ADDRESS>",
    "<ETH_WALLET>",
    "<BTC_WALLET>",
    "<UUID_FORMAT>",
    "<JWT_FORMAT>",
    "<HASH_FORMAT>",
    "<BASE64_FORMAT>",
    "<EMAIL>",
    "<HEX_ID>",
    "<NUMERIC_ID>",
]
_EXTERNAL_SCHEME_RE = re.compile(r"^(?:https?://|//|/\\|\\\\|javascript:|data:|vbscript:)", re.IGNORECASE)
_ENCODED_EXT_RE = re.compile(r"(?:https?%3A|%2F%2F|javascript%3A|data%3A|vbscript%3A)", re.IGNORECASE)
_DOUBLE_ENCODED_RE = re.compile(r"(?:https?%253A|%252F%252F|javascript%253A|data%253A|vbscript%253A)", re.IGNORECASE)
_BARE_DOMAIN_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9\-]*(?:\.[a-zA-Z0-9][a-zA-Z0-9\-]*)+\.[a-zA-Z]{2,}(?:[:/]|$)", re.IGNORECASE)
_EXEC_EXTS = frozenset({".exe", ".bat", ".cmd", ".ps1", ".vbs", ".wsf", ".hta", ".jar", ".class", ".sh", ".bash", ".py", ".rb", ".php", ".dll", ".scr", ".pif", ".msi", ".dmg", ".pkg", ".deb", ".rpm", ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".iso", ".img", ".jse", ".vbe"})
_MACRO_EXTS = frozenset({".doc", ".docm", ".xls", ".xlsm", ".ppt", ".pptm"})
_EMAIL_ECOSYSTEMS = {
    "google": frozenset({"gmail", "google", "googlemail"}),
    "gmail": frozenset({"gmail", "google", "googlemail"}),
    "microsoft": frozenset({"microsoft", "outlook", "live", "msn", "hotmail"}),
    "outlook": frozenset({"microsoft", "outlook", "live", "msn", "hotmail"}),
    "live": frozenset({"microsoft", "outlook", "live", "msn", "hotmail"}),
    "office": frozenset({"microsoft", "outlook", "live", "msn", "hotmail"}),
    "yahoo": frozenset({"yahoo"}),
    "apple": frozenset({"icloud", "apple"}),
    "icloud": frozenset({"icloud", "apple"}),
}


def _normalize_hostname(host: str) -> str:
    return (host or "").strip().rstrip(".").lower()


def _strip_hostname_trailing_dot(domain: str) -> str:
    return (domain or "").rstrip(".")


def _strip_default_port(domain: str, scheme: str) -> str:
    default_ports = {"http": "80", "https": "443"}
    default_port = default_ports.get(scheme)
    if not default_port:
        return domain
    hostpart, colon, port = domain.rpartition(":")
    if colon and port == default_port:
        return hostpart
    return domain


def _strip_empty_url_markers(rest: str) -> str:
    if rest.startswith("/?") or rest.startswith("/#"):
        rest = rest[1:]
    if rest in {"/", "?", "#", "?#"}:
        return ""
    if rest.startswith("?#"):
        rest = rest[1:]
    if rest.endswith("?") or rest.endswith("#"):
        rest = rest[:-1]
    return rest


def clean_url(url: str) -> str:
    url = str(url).strip()
    scheme_match = _SCHEME_RE.match(url)
    scheme = scheme_match.group("scheme").lower() if scheme_match else ""
    url = _PROTOCOL_RE.sub("", url)
    url = _WWW_RE.sub("", url)
    match = _DOMAIN_PATH_RE.match(url)
    if match:
        domain = _strip_hostname_trailing_dot(match.group("domain").lower())
        domain = _strip_default_port(domain, scheme)
        rest = match.group("path") or ""
        if rest:
            rest = rest.lower()
            rest = _strip_empty_url_markers(rest)
        return domain + rest
    return url.lower()


def is_short(url: str) -> bool:
    try:
        value = str(url).strip()
        if not value.startswith(("http://", "https://")):
            value = "http://" + value
        parsed = urllib.parse.urlsplit(value)
        domain = _normalize_hostname(parsed.hostname or "")
        return domain in _SHORTENER_DOMAINS
    except Exception:
        return False


def _unquote_twice(value: str) -> str:
    decoded = urllib.parse.unquote(value).strip()
    return urllib.parse.unquote(decoded).strip() if "%" in decoded else decoded


def _decode_base64_text(value: str) -> str:
    padded = value + "=" * (-len(value) % 4)
    for altchars in (b"-_", None):
        try:
            raw = __import__("base64").b64decode(padded.encode("ascii"), altchars=altchars, validate=True)
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            continue
    return ""


def _file_extension(value: str) -> str:
    decoded = _unquote_twice(value).lower()
    filename = decoded.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    filename = re.sub(r"[?#].*", "", filename)
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1]


def _is_existing_tag(value: str) -> bool:
    return bool(_TAG_RE.fullmatch(value))


def _extract_host(url: str) -> str:
    if not isinstance(url, str):
        return ""
    candidate = url.strip()
    if not candidate:
        return ""
    try:
        parsed = urllib.parse.urlsplit(candidate if "://" in candidate or candidate.startswith("//") else f"//{candidate}")
        host = parsed.hostname
        if host:
            return _normalize_hostname(host)
    except ValueError:
        pass
    fallback = candidate.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    if fallback.startswith("[") and "]" in fallback:
        return fallback[1 : fallback.index("]")].lower()
    return _normalize_hostname(fallback.split(":", 1)[0])


def _int_to_ipv4(value: int) -> str | None:
    if 0 <= value <= 0xFFFFFFFF:
        return ".".join(str((value >> shift) & 255) for shift in (24, 16, 8, 0))
    return None


def _decode_obfuscated_ipv4_host(host: str) -> str | None:
    raw = host.strip("[]")
    try:
        if _OBF_DEC_RE.fullmatch(raw):
            return _int_to_ipv4(int(raw, 10))
        if _OBF_HEX_RE.fullmatch(raw):
            return _int_to_ipv4(int(raw, 16))
        if _OBF_OCT_RE.fullmatch(raw):
            octets = [int(part, 8) for part in raw.split(".")]
            if all(0 <= octet <= 255 for octet in octets):
                return ".".join(str(octet) for octet in octets)
    except (ValueError, OverflowError):
        return None
    return None


def _normalize_obfuscated_ip(url: str) -> str:
    match = _HOST_RE.match(url)
    if not match:
        return url
    decoded = _decode_obfuscated_ipv4_host(match.group("host"))
    if not decoded:
        return url
    scheme = match.group("scheme") or ""
    return f"{scheme}{decoded}{match.group('rest')}"


def _get_replacer(mask_tag: str, url_domain: str = ""):
    base_tag = mask_tag[1:-1]
    is_fixed = base_tag in _FIX_LEN_TAGS
    domain = (url_domain or "").lower()

    def replacer(match: re.Match):
        prefix = match.group(1)
        target = match.group(2)
        if not target:
            return f"{prefix}<EMPTY:1>"
        if _is_existing_tag(target):
            return match.group(0)
        tag_name = base_tag
        multiplier = 1 if is_fixed else len(target)
        if mask_tag == "<REF>":
            decoded = _unquote_twice(target)
            decoded_b64 = _decode_base64_text(target).strip()
            multiplier = len(decoded) if decoded else len(target)
            if _DOUBLE_ENCODED_RE.search(target) or _ENCODED_EXT_RE.search(target) or _EXTERNAL_SCHEME_RE.match(decoded_b64):
                tag_name = "REF_ENCODED"
            elif _EXTERNAL_SCHEME_RE.match(decoded):
                tag_name = "REF_EXTERNAL"
            elif decoded.startswith("/"):
                tag_name = "REF_INTERNAL"
            elif _BARE_DOMAIN_RE.match(decoded):
                tag_name = "REF_DOMAIN"
            else:
                tag_name = "REF_OTHER"
        elif mask_tag == "<BASE64_FORMAT>":
            decoded_b64 = _decode_base64_text(target)
            if _EXTERNAL_SCHEME_RE.match(decoded_b64.strip()):
                tag_name = "BASE64_EXTERNAL"
        elif mask_tag == "<EMAIL>":
            email_domain = target.split("@")[-1].lower() if "@" in target else ""
            url_main = ".".join(domain.split(".")[-2:]) if domain else ""
            email_main = ".".join(email_domain.split(".")[-2:])
            is_match = bool(url_main and email_main == url_main)
            if not is_match:
                for url_kw, email_kws in _EMAIL_ECOSYSTEMS.items():
                    if url_kw in domain and any(email_kw in email_domain for email_kw in email_kws):
                        is_match = True
                        break
            tag_name = "EMAIL_MATCH" if is_match else "EMAIL_MISMATCH"
        elif mask_tag == "<FILE_PARAM>":
            ext = _file_extension(target)
            if ext in _EXEC_EXTS:
                tag_name = "FILE_EXEC"
            elif ext in _MACRO_EXTS:
                tag_name = "FILE_MACRO"
            elif not ext and re.fullmatch(r"[a-zA-Z0-9_\-]{8,}", target):
                tag_name = "FILE_ID"
            else:
                tag_name = "FILE_NAME"
        return f"{prefix}<{tag_name}:{multiplier}>"

    return replacer


_STATIC_REPLACERS = {"<DATE>": _get_replacer("<DATE>")}
for _tag in _VALUE_ORDER:
    if _tag != "<EMAIL>":
        _STATIC_REPLACERS[_tag] = _get_replacer(_tag)
for _tag in _NAME_BASED_RULES:
    _STATIC_REPLACERS[_tag] = _get_replacer(_tag)


def _mask_ipv6_candidates(text: str) -> str:
    def replacer(match: re.Match):
        candidate = match.group(1)
        try:
            __import__("ipaddress").IPv6Address(candidate)
        except ValueError:
            return match.group(0)
        return "<IPV6:1>"

    return _IPV6_CANDIDATE_RE.sub(replacer, text)


def sanitize_url(url: str) -> str:
    if not isinstance(url, str) or len(url) < 4:
        return url
    masked = _normalize_obfuscated_ip(url.strip())
    url_domain = _extract_host(masked)
    masked = _VALUE_BASED_RULES["<IPV4>"].sub("<IPV4:1>", masked)
    masked = _mask_ipv6_candidates(masked)
    masked = _VALUE_BASED_RULES["<DATE>"].sub(_STATIC_REPLACERS["<DATE>"], masked)
    for mask_tag, pattern in _NAME_BASED_RULES.items():
        masked = pattern.sub(_STATIC_REPLACERS[mask_tag], masked)
    for mask_tag in _VALUE_ORDER:
        replacer = _get_replacer(mask_tag, url_domain) if mask_tag == "<EMAIL>" else _STATIC_REPLACERS[mask_tag]
        masked = _VALUE_BASED_RULES[mask_tag].sub(replacer, masked)
    return masked


def to_char_stream(url: str) -> str:
    intermediate_url = sanitize_url(url)
    if not isinstance(intermediate_url, str):
        return url

    def char_mapper(match: re.Match):
        tag_name = match.group(1)
        multiplier = min(int(match.group(2)), _MAX_TAG_REPEAT)
        char = _TAG_CHAR_MAP.get(tag_name, _TAG_CHAR_MAP["DEFAULT"])
        return char * multiplier

    return _TAG_RE.sub(char_mapper, intermediate_url)


MODEL = None
TOKENIZER = None
CONFIG = None


def _load_model_artifacts():
    global MODEL, TOKENIZER, CONFIG
    if MODEL is None:
        CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        with TOKENIZER_PATH.open("rb") as handle:
            TOKENIZER = pickle.load(handle)
        MODEL = tf.keras.models.load_model(MODEL_PATH)
    return MODEL, TOKENIZER, CONFIG


def _prepare_sequence(url: str) -> np.ndarray:
    cleaned = clean_url(url)
    char_stream = to_char_stream(cleaned)
    seq = TOKENIZER.texts_to_sequences([char_stream])
    return pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")


def predict_url(url: str, threshold: float | None = None) -> dict:
    model, tokenizer, config = _load_model_artifacts()
    cleaned = clean_url(url)
    char_stream = to_char_stream(cleaned)
    seq = TOKENIZER.texts_to_sequences([char_stream])
    seq = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob = float(model.predict(seq, verbose=0)[0][0])
    if threshold is None:
        threshold = config.get("OPTIMAL_THRESHOLD", OPTIMAL_THRESHOLD)
    is_phishing = prob >= threshold
    return {
        "url": url,
        "probability": prob,
        "threshold": threshold,
        "label": "phishing" if is_phishing else "legitimate",
        "is_phishing": is_phishing,
        "clean_url": cleaned,
        "char_stream": char_stream,
    }


if __name__ == "__main__":
    print(predict_url("https://example.com"))
