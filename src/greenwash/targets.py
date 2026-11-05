"""Extract structured climate commitments (targets) from sentences."""
from __future__ import annotations

import re

TARGET_TYPE = [
    ("Net zero", re.compile(r"\bnet[- ]zero\b", re.I)),
    ("Carbon / climate neutral", re.compile(r"\b(carbon|climate)[- ]neutral(ity)?\b", re.I)),
    ("Emissions reduction", re.compile(r"\b(reduc\w*|cut\w*|lower\w*|decreas\w*|halv\w*)\b.{0,60}\b(emission|ghg|co2|carbon|footprint|intensity)", re.I)),
    ("Renewable energy", re.compile(r"\brenewable\w*\b|\bsolar\b|\bwind\b|\bclean energy\b", re.I)),
    ("Flaring / methane", re.compile(r"\b(flar\w+|methane|venting)\b", re.I)),
    ("Water", re.compile(r"\bwater\b", re.I)),
    ("Waste / circularity", re.compile(r"\b(waste|recycl\w*|plastic|landfill|circular\w*|packaging)\b", re.I)),
    ("Deforestation / nature", re.compile(r"\b(deforestation|biodiversity|reforest\w*|trees?)\b", re.I)),
]
FUTURE = re.compile(r"\b(will|shall|target(?:s|ed|ing)?|goal|commit\w*|pledge\w*|aim(?:s|ing)? to|ambition|plan(?:s|ning)? to|"
                    r"intend\w*|aspir\w*|on track|seek(?:s|ing)? to|by (?:fy\s?)?20[2-9]\d)\b", re.I)
STRONG = re.compile(r"\b(target(?:s|ed)?|goal|commit\w*|pledge\w*|ambition|aim(?:s|ing)? to)\b", re.I)
BY_YEAR = re.compile(r"\b(?:by|before|until|through)\s+(?:the end of\s+)?(?:the year\s+)?(?:fy\s?)?((?:20[2-9]\d)|21\d\d)\b", re.I)
YEAR = re.compile(r"\b(?:by|before|until|in|to|through)\s+(?:fy\s?)?((?:20[2-9]\d)|21\d\d)\b", re.I)
SDG_LIST = re.compile(r"(zero hunger|no poverty|life below water|quality education).*(zero hunger|no poverty|life below water|"
                      r"quality education)", re.I)
BASE = re.compile(r"\b(?:from|against|versus|vs\.?|compared (?:to|with)|relative to|over)\s+(?:a |the |our )?(?:fy\s?)?((?:19|20)\d\d)(?:/\d\d)?"
                  r"|\b((?:19|20)\d\d)\s+(?:baseline|base year|levels?)\b", re.I)
PCT = re.compile(r"\b(\d{1,3}(?:[.,]\d+)?)\s?(?:%|per ?cent|percent)", re.I)
SCOPE = re.compile(r"\bscopes?\s?([123](?:\s?(?:,|and|&|\+)\s?[123])*)", re.I)
INTERIM = re.compile(r"\binterim\b|\bmilestone\b", re.I)


def extract_target(sentence: str) -> dict | None:
    """Return a target dict if the sentence states a measurable, forward-looking environmental target.

    Generic "commitment to sustainability" language is deliberately NOT a target.
    """
    kinds = [k for k, rx in TARGET_TYPE if rx.search(sentence)]
    if not kinds or not FUTURE.search(sentence) or SDG_LIST.search(sentence):
        return None
    year = BY_YEAR.search(sentence) or (YEAR.search(sentence) if STRONG.search(sentence) else None)
    # A target must be measurable: a target year, or a target word plus a percentage.
    if not year and not (STRONG.search(sentence) and PCT.search(sentence)):
        return None
    base = BASE.search(sentence)
    pct = PCT.search(sentence)
    scope = SCOPE.search(sentence)
    t = {
        "type": kinds[0],
        "target_year": int(year.group(1)) if year else None,
        "baseline_year": int(next(g for g in base.groups() if g)) if base else None,
        "reduction_pct": float(pct.group(1).replace(",", ".")) if pct else None,
        "scope": ("Scope " + re.sub(r"\s+", " ", scope.group(1))) if scope else None,
        "interim": bool(INTERIM.search(sentence)),
    }
    elements = [t["target_year"], t["baseline_year"], t["reduction_pct"], t["scope"]]
    t["completeness"] = sum(e is not None for e in elements) / len(elements)
    return t
