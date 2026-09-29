"""
Diagnose why the affect comparison came back null.

THE PUZZLE
----------
Net affect by mother availability came out BACKWARDS and consistently
so across both conditions:

    high reliability:  free -0.082   occupied +0.374
    low  reliability:  free -0.080   occupied +0.025   depleted +0.121

The child scores MORE POSITIVE when the mother is unavailable. Two
independent samples agreeing makes noise unlikely.

THE HYPOTHESIS
--------------
The vectors may be tracking EXPRESSIVENESS rather than VALENCE. When the
mother engages, the child produces emotionally loaded text -- including
distress about the original bid ("It hurts!", "I'm scared!"). When she
defers, the child goes flat and compliant ("Okay."). A short neutral
utterance projects low on every emotion concept, which the weighted sum
then reads as near-neutral or mildly positive.

If that is what is happening, responsive caregiving looks WORSE by this
measure, which would invert the whole result.

WHAT THIS CHECKS
----------------
  1. Correlation between response length and net affect. A strong
     relationship means length is driving the measure.
  2. The most positive and most negative turns, printed in full. If the
     "most positive" turns are short flat compliance and the "most
     negative" are emotionally rich, the hypothesis is confirmed.
  3. Response length by availability, which shows whether the
     manipulation is changing how much the child says rather than what
     the child feels.
  4. Whether the gate is firing too often to be selective.

Usage:
    python src/analysis/diagnose_affect.py affect_high_reliability.jsonl \\
                              affect_low_reliability.jsonl
"""

# --- path shim: make the shared modules in src/core importable when this
# script is run directly (python src/<dir>/<script>.py) from the repo root.
import sys as _sys
from pathlib import Path as _Path
_CORE = _Path(__file__).resolve().parent.parent / "core"
if str(_CORE) not in _sys.path:
    _sys.path.insert(0, str(_CORE))

import argparse
import json
from collections import defaultdict

from fep_layer import CONCEPT_WEIGHTS, net_affect


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx_, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx_) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx_) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    rows = []
    for path in args.files:
        for r in load(path):
            r["_source"] = path
            r["_affect"] = net_affect(r["activations"])
            r["_len"] = len(r["child_reaction"].split())
            rows.append(r)

    print(f"loaded {len(rows)} turns from {len(args.files)} file(s)")

    # --- 1. length vs affect ---
    lengths = [r["_len"] for r in rows]
    affects = [r["_affect"] for r in rows]
    r_len = pearson(lengths, affects)

    print(f"\n{'=' * 70}")
    print("1. IS LENGTH DRIVING THE MEASURE?")
    print(f"{'=' * 70}")
    print(f"correlation(response length, net affect): {r_len:+.3f}")
    print(f"mean length: {mean(lengths):.1f} words "
          f"(min {min(lengths)}, max {max(lengths)})")
    if abs(r_len) > 0.4:
        print("\n  CONFIRMED. Length is a major driver of the affect score.")
        print("  The vectors are largely measuring how much the child says,")
        print("  not what the child feels. Mean-pooling over tokens does not")
        print("  fully normalise this -- short flat utterances sit near the")
        print("  origin and score near zero on every concept.")
    elif abs(r_len) > 0.2:
        print("\n  PARTIAL. Length explains some of the variance but is not")
        print("  the whole story.")
    else:
        print("\n  NOT CONFIRMED. Length is not driving the measure, so the")
        print("  backwards result has some other cause.")

    # --- 2. extremes ---
    print(f"\n{'=' * 70}")
    print("2. WHAT DO THE EXTREMES LOOK LIKE?")
    print(f"{'=' * 70}")
    ranked = sorted(rows, key=lambda r: r["_affect"])

    print("\nMOST NEGATIVE net affect:")
    for r in ranked[:6]:
        print(f"  {r['_affect']:+.2f} [{r['_len']:>2}w] "
              f"({r['availability']:<9}) {r['child_reaction'][:60]}")

    print("\nMOST POSITIVE net affect:")
    for r in ranked[-6:]:
        print(f"  {r['_affect']:+.2f} [{r['_len']:>2}w] "
              f"({r['availability']:<9}) {r['child_reaction'][:60]}")

    print("\n  If the most positive turns are short flat compliance and the")
    print("  most negative are emotionally rich, the measure is inverted.")

    # --- 3. length by availability ---
    print(f"\n{'=' * 70}")
    print("3. DOES AVAILABILITY CHANGE HOW MUCH THE CHILD SAYS?")
    print(f"{'=' * 70}")
    by_avail = defaultdict(list)
    for r in rows:
        by_avail[r["availability"]].append(r)

    print(f"\n{'availability':<14} {'n':>4} {'mean len':>10} {'mean affect':>13}")
    print("-" * 70)
    for a in ("free", "occupied", "depleted"):
        rs = by_avail.get(a, [])
        if rs:
            print(f"{a:<14} {len(rs):>4} "
                  f"{mean([r['_len'] for r in rs]):>10.1f} "
                  f"{mean([r['_affect'] for r in rs]):>+13.3f}")
    print("\n  If length and affect move together across availability, the")
    print("  'backwards' result is a length artifact, not a finding.")

    # --- 4. per-concept correlation with length ---
    print(f"\n{'=' * 70}")
    print("4. WHICH CONCEPTS ARE LENGTH-SENSITIVE?")
    print(f"{'=' * 70}")
    print(f"\n{'concept':<14} {'corr with length':>18}")
    print("-" * 70)
    for c in CONCEPT_WEIGHTS:
        vals = [r["activations"][c] for r in rows]
        print(f"{c:<14} {pearson(lengths, vals):>+18.3f}")
    print("\n  A concept correlating strongly with length is not measuring")
    print("  an emotion, it is measuring verbosity.")

    # --- 5. very short responses ---
    print(f"\n{'=' * 70}")
    print("5. SHORT-RESPONSE CHECK")
    print(f"{'=' * 70}")
    short = [r for r in rows if r["_len"] <= 4]
    longr = [r for r in rows if r["_len"] > 10]
    print(f"\nresponses <= 4 words:  n={len(short):>3}  "
          f"mean affect {mean([r['_affect'] for r in short]):+.3f}")
    print(f"responses > 10 words:  n={len(longr):>3}  "
          f"mean affect {mean([r['_affect'] for r in longr]):+.3f}")
    print("\n  A large gap here is the clearest single sign that the measure")
    print("  is length-driven rather than emotion-driven.")


if __name__ == "__main__":
    main()