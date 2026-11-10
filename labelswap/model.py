"""Dependency-free multinomial Naive Bayes text classifier with a confidence gate."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

FORMAT = "labelswap/1"
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    words = _TOKEN.findall(text.lower())
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


def in_holdout(text: str, fraction: float) -> bool:
    """Deterministic split: the same text always lands on the same side."""
    if fraction <= 0:
        return False
    h = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    return (h / 2**64) < fraction


@dataclass
class Model:
    alpha: float = 1.0
    threshold: float = 1.01  # above any probability: defer everything until tuned
    doc_counts: Counter = field(default_factory=Counter)
    token_counts: dict = field(default_factory=lambda: defaultdict(Counter))
    token_totals: Counter = field(default_factory=Counter)
    vocab: set = field(default_factory=set)

    @property
    def labels(self) -> list[str]:
        return sorted(self.doc_counts)

    def fit(self, rows: Iterable[tuple[str, str]]) -> "Model":
        for text, label in rows:
            toks = tokenize(text)
            self.doc_counts[label] += 1
            self.token_counts[label].update(toks)
            self.token_totals[label] += len(toks)
            self.vocab.update(toks)
        if not self.doc_counts:
            raise ValueError("no training rows")
        return self

    def proba(self, text: str) -> dict[str, float]:
        if not self.doc_counts:
            raise ValueError("model is empty")
        toks = [t for t in tokenize(text) if t in self.vocab]
        n_docs = sum(self.doc_counts.values())
        v = len(self.vocab) or 1
        logs = {}
        for label in self.labels:
            lp = math.log(self.doc_counts[label] / n_docs)
            denom = self.token_totals[label] + self.alpha * v
            counts = self.token_counts[label]
            for t in toks:
                lp += math.log((counts.get(t, 0) + self.alpha) / denom)
            logs[label] = lp
        top = max(logs.values())
        exp = {k: math.exp(val - top) for k, val in logs.items()}
        z = sum(exp.values())
        return {k: val / z for k, val in exp.items()}

    def decide(self, text: str) -> dict:
        probs = self.proba(text)
        label = max(sorted(probs), key=lambda k: probs[k])
        conf = probs[label]
        return {
            "label": label,
            "confidence": round(conf, 6),
            "action": "local" if conf >= self.threshold else "defer",
        }

    def to_json(self) -> dict:
        return {
            "format": FORMAT,
            "alpha": self.alpha,
            "threshold": self.threshold,
            "doc_counts": dict(self.doc_counts),
            "token_counts": {k: dict(v) for k, v in self.token_counts.items()},
        }

    @classmethod
    def from_json(cls, data: dict) -> "Model":
        if data.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} model file")
        m = cls(alpha=float(data["alpha"]), threshold=float(data["threshold"]))
        m.doc_counts = Counter(data["doc_counts"])
        for label, counts in data["token_counts"].items():
            m.token_counts[label] = Counter(counts)
            m.token_totals[label] = sum(counts.values())
            m.vocab.update(counts)
        return m

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "Model":
        with open(path, encoding="utf-8") as f:
            return cls.from_json(json.load(f))


def shadow(model: Model, rows: list[tuple[str, str]], threshold: float | None = None) -> dict:
    """Compare local decisions to reference (LLM) labels, like running both side by side."""
    t = model.threshold if threshold is None else threshold
    total = len(rows)
    accepted = agree = overall_agree = 0
    confusion: Counter = Counter()
    for text, ref in rows:
        d = model.decide(text)
        if d["label"] == ref:
            overall_agree += 1
        if d["confidence"] >= t:
            accepted += 1
            if d["label"] == ref:
                agree += 1
            else:
                confusion[(ref, d["label"])] += 1
    return {
        "rows": total,
        "threshold": t,
        "coverage": accepted / total if total else 0.0,
        "agreement_on_local": agree / accepted if accepted else None,
        "agreement_if_all_local": overall_agree / total if total else None,
        "local_calls": accepted,
        "deferred_calls": total - accepted,
        "top_mistakes": [
            {"reference": r, "local": l, "count": c} for (r, l), c in confusion.most_common(5)
        ],
    }


def tune(model: Model, rows: list[tuple[str, str]], target: float) -> dict:
    """Pick the lowest confidence gate whose local answers agree with the reference at >= target."""
    if not rows:
        raise ValueError("no rows to tune on")
    scored = []
    for text, ref in rows:
        d = model.decide(text)
        scored.append((d["confidence"], d["label"] == ref))
    scored.sort(key=lambda x: -x[0])
    best = None
    agree = 0
    for i, (conf, ok) in enumerate(scored, 1):
        agree += ok
        # Only cut between distinct confidences so ties are never split.
        if i < len(scored) and scored[i][0] == conf:
            continue
        if agree / i >= target:
            best = conf
    model.threshold = best if best is not None else 1.01
    report = shadow(model, rows)
    report["target"] = target
    report["met"] = best is not None
    return report
