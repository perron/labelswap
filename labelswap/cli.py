"""Command line for labelswap."""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .model import Model, in_holdout, shadow, tune


def read_rows(path: str, text_field: str, label_field: str, need_label: bool = True):
    rows = []
    fh = sys.stdin if path == "-" else open(path, encoding="utf-8")
    try:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{n}: bad JSON ({e.msg})")
            if not isinstance(obj, dict):
                raise SystemExit(f"{path}:{n}: expected a JSON object")
            if text_field not in obj:
                raise SystemExit(f"{path}:{n}: missing field '{text_field}'")
            label = obj.get(label_field)
            if need_label and label is None:
                raise SystemExit(f"{path}:{n}: missing field '{label_field}'")
            rows.append((str(obj[text_field]), None if label is None else str(label)))
    finally:
        if fh is not sys.stdin:
            fh.close()
    return rows


def pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def print_report(r: dict, cost: float | None) -> None:
    print(f"rows checked:          {r['rows']}")
    print(f"confidence gate:       {r['threshold']:.4f}" if r["threshold"] <= 1 else "confidence gate:       none (everything defers)")
    print(f"handled locally:       {r['local_calls']} ({pct(r['coverage'])})")
    print(f"deferred to the LLM:   {r['deferred_calls']}")
    print(f"agreement when local:  {pct(r['agreement_on_local'])}")
    print(f"agreement if all local: {pct(r['agreement_if_all_local'])}")
    if "target" in r:
        print(f"target agreement:      {pct(r['target'])} ({'met' if r['met'] else 'NOT met, every call defers'})")
    if cost is not None:
        print(f"LLM spend avoided:     ${r['local_calls'] * cost:,.2f} on these {r['rows']} calls")
    for m in r["top_mistakes"]:
        print(f"  mistake: LLM said {m['reference']!r}, local said {m['local']!r} x{m['count']}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="labelswap", description="Replace repetitive LLM labeling calls with a tiny local model that defers when unsure.")
    p.add_argument("--version", action="version", version=f"labelswap {__version__}")
    p.add_argument("--text-field", default="text")
    p.add_argument("--label-field", default="label")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="learn from logged LLM labels, then tune the gate on a held-out slice")
    t.add_argument("data")
    t.add_argument("-o", "--out", default="labelswap.json")
    t.add_argument("--holdout", type=float, default=0.2)
    t.add_argument("--target", type=float, default=0.97, help="required agreement with the LLM on calls handled locally")
    t.add_argument("--alpha", type=float, default=1.0)
    t.add_argument("--cost", type=float, help="LLM cost per call in dollars, for the savings line")
    t.add_argument("--json", action="store_true")

    s = sub.add_parser("shadow", help="run the model next to fresh LLM labels and report agreement")
    s.add_argument("model")
    s.add_argument("data")
    s.add_argument("--cost", type=float)
    s.add_argument("--json", action="store_true")

    d = sub.add_parser("decide", help="label text locally or say defer; reads JSONL from a file or - for stdin")
    d.add_argument("model")
    d.add_argument("data", nargs="?", default="-")
    d.add_argument("--text", help="decide a single string instead of JSONL")

    a = p.parse_args(argv)
    if a.cmd == "train":
        if not 0 <= a.holdout < 1:
            p.error("--holdout must be in [0, 1)")
        if not 0 <= a.target <= 1:
            p.error("--target must be in [0, 1]")
        rows = read_rows(a.data, a.text_field, a.label_field)
        train = [r for r in rows if not in_holdout(r[0], a.holdout)]
        hold = [r for r in rows if in_holdout(r[0], a.holdout)]
        if not train:
            raise SystemExit("no training rows after the holdout split")
        model = Model(alpha=a.alpha).fit(train)
        if hold:
            rep = tune(model, hold, a.target)
        else:
            rep = None
        model.save(a.out)
        if a.json:
            print(json.dumps({"model": a.out, "train_rows": len(train), "holdout_rows": len(hold), "labels": model.labels, "report": rep}, indent=2))
        else:
            print(f"trained on {len(train)} rows, {len(model.labels)} labels, saved {a.out}")
            if rep:
                print_report(rep, a.cost)
            else:
                print("no holdout rows, so no gate was tuned and every call defers")
        return 0
    try:
        model = Model.load(a.model)
    except FileNotFoundError:
        raise SystemExit(f"model file not found: {a.model}")
    except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError) as e:
        raise SystemExit(f"cannot load model {a.model}: {e}")
    if a.cmd == "shadow":
        rep = shadow(model, read_rows(a.data, a.text_field, a.label_field))
        print(json.dumps(rep, indent=2) if a.json else "", end="\n" if a.json else "")
        if not a.json:
            print_report(rep, a.cost)
        return 0
    if a.text is not None:
        print(json.dumps(model.decide(a.text)))
        return 0
    for text, _ in read_rows(a.data, a.text_field, a.label_field, need_label=False):
        print(json.dumps(model.decide(text)))
    return 0
