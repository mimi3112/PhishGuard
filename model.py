import os
import re
import json
import time
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import joblib

# Load trained pipelines and metadata
URL_MODEL = joblib.load("models/url_model.pkl")
TEXT_MODEL = joblib.load("models/text_model.pkl")

with open("models/metadata.json", "r") as f:
    META = json.load(f)

URL_FEATURE_COLS = META.get("url_features", [
    "url_length", "dot_count", "hyphen_count", "at_symbol", "ip_address",
    "https", "subdomain_count", "special_char_count", "suspicious_word_count", "domain_length"
])

def extract_url_features_dict(url):
    raw = url.strip()
    u = raw if re.match(r"^[a-zA-Z]+://", raw) else "http://" + raw
    p = urlparse(u)
    host = p.netloc.split("@")[-1].split(":")[0]
    full = raw
    words = ["login", "verify", "account", "secure", "update", "bank", "password", "confirm", "signin", "wallet"]
    
    return {
        "url_length": len(full),
        "dot_count": full.count("."),
        "hyphen_count": full.count("-"),
        "at_symbol": int("@" in full),
        "ip_address": int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host))),
        "https": int(u.lower().startswith("https://")),
        "subdomain_count": max(0, len(host.split(".")) - 2),
        "special_char_count": sum(not ch.isalnum() for ch in full),
        "suspicious_word_count": sum(w in full.lower() for w in words),
        "domain_length": len(host)
    }

def extract_url_dataframe(url):
    feats = extract_url_features_dict(url)
    return pd.DataFrame([feats], columns=URL_FEATURE_COLS)

def predict_url(url):
    if not url.strip():
        return None
    df = extract_url_dataframe(url)
    prob = float(URL_MODEL.predict_proba(df)[0, 1])
    return prob

def predict_text(text):
    if not text.strip():
        return None
    prob = float(TEXT_MODEL.predict_proba([text.strip()])[0, 1])
    return prob

def analyze(url="", message=""):
    start_time = time.time()
    url_cleaned = url.strip() if url else ""
    msg_cleaned = message.strip() if message else ""

    url_risk = predict_url(url_cleaned)
    text_risk = predict_text(msg_cleaned)

    vals = [v for v in [url_risk, text_risk] if v is not None]
    if not vals:
        return {
            "error": "Please provide either a website URL, message content, or both for analysis."
        }

    # Combined weighted risk score
    if len(vals) == 1:
        final_risk = vals[0]
    else:
        # 55% URL risk weight, 45% Text NLP risk weight
        final_risk = (0.55 * url_risk) + (0.45 * text_risk)

    reasons = []
    # Explainable signals for URL
    if url_cleaned:
        f = extract_url_features_dict(url_cleaned)
        if f["ip_address"] == 1:
            reasons.append("Host is an IP address instead of a recognized domain name.")
        if f["https"] == 0:
            reasons.append("Insecure connection (HTTP): URL does not use SSL/HTTPS encryption.")
        if f["suspicious_word_count"] >= 2:
            reasons.append("Contains multiple high-risk credential keywords (login, verify, bank, secure).")
        elif f["suspicious_word_count"] == 1:
            reasons.append("Contains sensitive authentication keywords.")
        if f["at_symbol"] == 1:
            reasons.append("Contains '@' delimiter, commonly used to conceal the actual destination.")
        if f["hyphen_count"] >= 3:
            reasons.append("Excessive hyphens in host indicating potential brand typosquatting.")
        if f["subdomain_count"] >= 3:
            reasons.append("Suspiciously deep subdomain structure designed to impersonate legitimate brands.")
        if f["url_length"] > 85:
            reasons.append("Unusually long URL structure with obfuscated query strings.")

    # Explainable signals for Text
    if msg_cleaned:
        low = msg_cleaned.lower()
        urgent_keywords = ["urgent", "immediately", "24 hours", "suspended", "frozen", "deactivated", "lockout", "action required"]
        if any(w in low for w in urgent_keywords):
            reasons.append("High psychological urgency and threat of account deactivation detected.")
        if any(w in low for w in ["password", "credentials", "seed phrase", "otp", "pin", "wire transfer", "routing number"]):
            reasons.append("Direct solicitation for sensitive credentials, OTP, or financial disbursement.")
        if any(w in low for w in ["tax refund", "prize winner", "gift card", "reward voucher", "$499", "geek squad"]):
            reasons.append("Lucrative lure or invoice dispute bait identified.")
        if ("http://" in low or "https://" in low) and not url_cleaned:
            reasons.append("Embedded external hyperlinks detected within message body.")

    # Heuristic security boosting for critical structural signals
    heuristic_penalty = 0.0
    if url_cleaned:
        f = extract_url_features_dict(url_cleaned)
        if f["ip_address"] == 1:
            heuristic_penalty += 0.40
        if f["https"] == 0 and f["suspicious_word_count"] >= 1:
            heuristic_penalty += 0.25
        if f["at_symbol"] == 1:
            heuristic_penalty += 0.25
    if msg_cleaned:
        low = msg_cleaned.lower()
        if any(w in low for w in ["password", "credentials", "seed phrase", "routing number"]):
            heuristic_penalty += 0.30
        if any(w in low for w in ["urgent", "immediately", "suspended", "frozen", "deactivated"]):
            heuristic_penalty += 0.20

    if heuristic_penalty > 0:
        final_risk = min(1.0, max(final_risk, (final_risk * 0.4) + (heuristic_penalty * 0.6)))

    # Determine risk tier
    final_pct = round(final_risk * 100, 1)
    if final_pct >= 68.0:
        verdict = "PHISHING"
        level = "HIGH"
        recommendation = "DANGER: High probability of phishing or credential theft. Do NOT click any links, enter credentials, or send funds."
    elif final_pct >= 38.0:
        verdict = "SUSPICIOUS"
        level = "MEDIUM"
        recommendation = "CAUTION: This contains suspicious indicators. Cross-check sender identity via official channels before interacting."
    else:
        verdict = "LEGITIMATE"
        level = "LOW"
        recommendation = "SAFE: No malicious phishing indicators detected. Standard caution is still recommended."

    latency_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "verdict": verdict,
        "level": level,
        "risk": final_pct,
        "url_risk": None if url_risk is None else round(url_risk * 100, 1),
        "text_risk": None if text_risk is None else round(text_risk * 100, 1),
        "reasons": reasons if reasons else ["No anomalous signals detected; URL and text exhibit standard legitimate attributes."],
        "recommendation": recommendation,
        "latency_ms": latency_ms,
        "models_used": {
            "url_model": META.get("best_url_model", "Decision Tree"),
            "text_model": META.get("best_text_model", "Naive Bayes")
        }
    }

if __name__ == "__main__":
    test_res = analyze(
        url="http://192.168.1.1/secure-chase-banking/verify.php",
        message="URGENT: Your bank account is locked. Verify password immediately."
    )
    print("Test Analysis Result:")
    print(json.dumps(test_res, indent=2))
