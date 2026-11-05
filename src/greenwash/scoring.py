"""Transparent, rule-based specificity & verifiability scoring of environmental claims.

Each claim starts at 50. Red flags (vague, absolute, superlative, offsets, ...) subtract points and
evidence signals (numbers, target year, baseline, scope, verification, ...) add points. Each signal counts
once per claim. The score is clipped to 0-100. Signals and weights live in data/reference/red_flags.csv.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from .paths import LEXICON

BASE = 50


@dataclass(frozen=True)
class Signal:
    name: str
    kind: str  # "red" or "green"
    weight: int
    regex: re.Pattern
    label: str
    rationale: str
    source: str


@lru_cache(maxsize=1)
def load_signals() -> tuple[Signal, ...]:
    df = pd.read_csv(LEXICON)
    return tuple(Signal(r.signal, r.kind, int(r.weight), re.compile(r.pattern, re.I), r.label, r.rationale, r.source)
                 for r in df.itertuples())


SIGNAL_NAMES = [s.name for s in load_signals()]


def detect(text: str) -> dict[str, list[str]]:
    """Return {signal_name: [matched phrases]} for the signals present in the text."""
    out = {}
    for s in load_signals():
        hits = [m.group(0) for m in s.regex.finditer(text)]
        if hits:
            out[s.name] = sorted(set(h.strip() for h in hits), key=str.lower)
    return out


def specificity_score(text: str) -> tuple[int, dict[str, list[str]]]:
    """0 (vague, unverifiable) to 100 (specific, verifiable), plus the signals that drove it."""
    found = detect(text)
    w = {s.name: s.weight for s in load_signals()}
    score = BASE + sum(w[k] for k in found)
    return int(max(0, min(100, score))), found


def grade(score: float) -> str:
    """Letter grade for a specificity score."""
    return "A" if score >= 75 else "B" if score >= 60 else "C" if score >= 45 else "D" if score >= 30 else "E"


def features(texts) -> pd.DataFrame:
    """Binary signal-presence features (one column per signal) for modelling."""
    sig = load_signals()
    return pd.DataFrame([[int(bool(s.regex.search(t))) for s in sig] for t in texts], columns=[s.name for s in sig])


def explain(found: dict[str, list[str]]) -> list[dict]:
    """Human-readable rows for each triggered signal."""
    rows = []
    for s in load_signals():
        if s.name in found:
            rows.append({"signal": s.label, "kind": s.kind, "points": s.weight, "evidence": ", ".join(found[s.name][:4]),
                         "why": s.rationale, "reference": s.source})
    return rows


def signal_matrix(texts):
    """Picklable feature function used inside the saved risk pipeline."""
    return features(texts).values
