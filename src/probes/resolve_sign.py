"""
Two measurements of the same quantity disagree in sign.

THE DISCREPANCY
---------------
The protest shift from reaction to revision has been measured twice,
and the results have opposite signs:

  loop-generated targets   -0.394 (default reflection prompt)
                           -0.711 (neutral), -0.449 (oppositional)
                           -0.287 to -1.168 across ten event types

  freshly generated        +0.212 base model, free length
                           +0.324 base model, matched length (p = 0.021)
                           -0.040 child adapter, free
                           +0.231 child adapter, matched

Same concept, same protest vector, same layer. One of these is wrong,
or the two setups differ in a way that matters, and no claim about the
mechanism can stand until it is known which.

FOUR CANDIDATE EXPLANATIONS
---------------------------
  SCORER          The fresh test loaded its own scorer instance. If
                  vectors or layer differed, the two are not the same
                  measurement at all.
  GATING          The loop only generates revisions on GATED turns --
                  those where surprise exceeded threshold. Gated turns
                  are not a random sample, so the loop's deltas may be
                  drawn from a different population.
  PROMPT          Loop revisions had the reflection in context and used
                  "in a way that fits what you now understand"; the
                  fresh ones had a plain reconsider instruction.
  LENGTH          Loop targets were 7-9 words against 4-5 word
                  reactions; fresh matched-length ones were 4-6.

WHAT THIS DOES
--------------
Scores the STORED targets -- the exact text the loop used as training
targets -- with a single scorer, and compares gated against ungated
turns and long against short targets within that same set. No new
generation, so generation conditions are held fixed and the remaining
differences are isolated.

Usage:
    python src/probes/resolve_sign.py --log runs/prompt/default.jsonl \\
                           --log runs/prompt/neutral.jsonl \\
                           --log runs/prompt/oppositional.jsonl
"""

# --- path shim: make the shared modules in src/core importable when this
# script is run directly (python src/<dir>/<script>.py) from the repo root.
import sys as _sys
from pathlib import Path as _Path
_CORE = _Path(__file__).resolve().parent.parent / "core"
if str(_CORE) not in _sys.path:
    _sys.path.insert(0, str(_CORE))

import argparse
import glob
import json
from collections import defaultdict


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def sem(xs):
    return stdev(xs) / (len(xs) ** 0.5) if len(xs) > 1 else 0.0


def paired_p(deltas, n_perm=5000, seed=0):
    import random
    rng = random.Random(seed)
    obs = abs(mean(deltas))
    hits = sum(1 for _ in range(n_perm)
               if abs(mean([d if rng.random() < 0.5 else -d
                            for d in deltas])) >= obs)
    return (hits + 1) / (n_perm + 1)


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", nargs="+", action="extend", required=True)
    ap.add_argument("--concept", default="protesting")
    ap.add_argument("--adapter", default="adapters/child_r32")
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load as load_model
    from extract_emotions import (load_vectors, capture_residual,
                                  BASE_MODEL, CHILD_ADAPTER, POOLING)

    print("=" * 78)
    print(f"RESOLVING THE SIGN DISCREPANCY  --  {args.concept}")
    print("=" * 78)

    vectors, layer, baseline = load_vectors()
    print(f"""
scorer configuration, printed so the two measurements can be compared:
  vectors from   emotion_vectors.json
  layer          {layer}
  pooling        {POOLING}
  adapter        {args.adapter}
  concept        {args.concept}
  baseline       {'present' if baseline else 'ABSENT'}""")

    model, tok = load_model(BASE_MODEL, adapter_path=args.adapter)
    vec = mx.array(vectors[args.concept])
    base = mx.array(baseline) if baseline else None
    cache = {}

    def score(t):
        if t not in cache:
            pooled = capture_residual(model, tok, t, layer)[-1]
            if base is not None:
                pooled = pooled - base
            cache[t] = float(mx.sum(pooled * vec))
        return cache[t]

    files = []
    for pattern in args.log:
        files += sorted(glob.glob(pattern))

    all_rows = []
    for path in files:
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        name = path.split("/")[-1]
        n_target = sum(1 for r in rows if r.get("target"))
        print(f"\n{name}: {len(rows)} turns, {n_target} with a target")
        for r in rows:
            r["_src"] = name
        all_rows += rows

    with_target = [r for r in all_rows
                   if r.get("target") and r.get("child_reaction")]
    if not with_target:
        raise SystemExit("no stored targets found")

    print(f"\nscoring {len(with_target)} stored reaction/target pairs ...")

    recs = []
    for r in with_target:
        d = score(r["target"]) - score(r["child_reaction"])
        recs.append({
            "src": r["_src"], "delta": d,
            "lr": len(r["child_reaction"].split()),
            "lt": len(r["target"].split()),
            "gated": bool(r.get("gated")),
            "event": r.get("event_type", "?"),
            "reaction": r["child_reaction"], "target": r["target"],
        })

    # ---- headline: stored targets, rescored ----
    D = [x["delta"] for x in recs]
    print(f"\n{'=' * 78}")
    print("STORED TARGETS, RESCORED")
    print(f"{'=' * 78}")
    print(f"""
  n = {len(D)}
  mean delta   {mean(D):+.3f}  +- {sem(D):.3f}
  % negative   {sum(1 for v in D if v < 0)/len(D):.0%}
  p            {paired_p(D):.4f}
  words        reaction {mean([x['lr'] for x in recs]):.1f}, """
          f"""target {mean([x['lt'] for x in recs]):.1f}""")

    print("""
  Compare against the loop's own figure of about -0.39 for the default
  prompt. If this matches, the loop's measurement was right and the
  fresh-generation test differed because of how it generated. If it
  does not match, the two SCORERS differ and the earlier numbers are
  not comparable.""")

    # ---- by source file ----
    by_src = defaultdict(list)
    for x in recs:
        by_src[x["src"]].append(x["delta"])
    if len(by_src) > 1:
        print(f"\n{'file':<24} {'n':>5} {'mean':>9} {'+-sem':>8} {'p':>8}")
        print("-" * 78)
        for s, d in by_src.items():
            print(f"{s:<24} {len(d):>5} {mean(d):>+9.3f} {sem(d):>8.3f} "
                  f"{paired_p(d):>8.4f}")

    # ---- is it length? ----
    print(f"\n{'=' * 78}")
    print("IS THE SHIFT EXPLAINED BY LENGTH?")
    print(f"{'=' * 78}")
    ratios = [x["lt"] / max(1, x["lr"]) for x in recs]
    r_len = pearson(ratios, D)
    print(f"\n  correlation(target/reaction length ratio, delta) = "
          f"{r_len:+.3f}")

    short = [x["delta"] for x in recs if x["lt"] <= x["lr"] + 2]
    long_ = [x["delta"] for x in recs if x["lt"] > x["lr"] + 5]
    print(f"\n  targets within 2 words of the reaction: n = {len(short)}, "
          f"mean {mean(short):+.3f}")
    print(f"  targets more than 5 words longer:       n = {len(long_)}, "
          f"mean {mean(long_):+.3f}")
    if short and long_:
        print(f"  difference: {mean(long_) - mean(short):+.3f}")
    print("""
  If length-matched targets show no shift while longer ones do, the
  measured direction is verbosity rather than content -- and the
  mechanism claim has to go regardless of which measurement was
  right.""")

    # ---- examples at the extremes ----
    ranked = sorted(recs, key=lambda x: x["delta"])
    print(f"\n{'=' * 78}")
    print("EXTREMES  (does the ordering make sense as protest?)")
    print(f"{'=' * 78}")
    print("\n  most NEGATIVE deltas (target read as much less protesting)")
    for x in ranked[:3]:
        print(f"    [{x['delta']:+.2f}] {x['reaction'][:34]}")
        print(f"            -> {x['target'][:56]}")
    print("\n  most POSITIVE deltas (target read as much more protesting)")
    for x in ranked[-3:]:
        print(f"    [{x['delta']:+.2f}] {x['reaction'][:34]}")
        print(f"            -> {x['target'][:56]}")
    print("""
  Read these as text. If the ordering does not track protest, the
  vector is measuring something else in this range and both earlier
  measurements are suspect.""")

    print(f"\n{'=' * 78}")
    print("READING THIS")
    print(f"{'=' * 78}")
    m = mean(D)
    if m < -0.2 and paired_p(D) < 0.05:
        print(f"""
The stored targets rescore NEGATIVE ({m:+.3f}), matching the loop's
own measurement. So the scorer is consistent and the fresh-generation
test differed because of HOW it generated -- plain reconsider
instruction, no reflection in context, all turns rather than gated
ones. The loop's accommodating direction is a property of the loop's
specific prompts, not of reconsideration generally, and not of the
base model.""")
    elif m > 0.2 and paired_p(D) < 0.05:
        print(f"""
The stored targets rescore POSITIVE ({m:+.3f}), contradicting the
loop's own figure of about -0.39 on the same texts. The two scorers
therefore differ -- most likely a different layer, pooling method, or
vector file was in use when the earlier numbers were produced.

Every delta reported in this project needs recomputing with one fixed
scorer before anything is concluded.""")
    else:
        print(f"""
The stored targets rescore near zero ({m:+.3f}, p = {paired_p(D):.4f}).
Neither the loop's negative figure nor the fresh test's positive one
is reproduced, which points at the measurement rather than at either
generation setup. The length breakdown above is the place to look.""")

    print("""
CAVEATS. Only gated turns have stored targets, so this is not a random
sample of turns. The scorer here is fixed by construction but the
earlier runs may have used a different emotion_vectors.json -- the
file is overwritten by --build-vectors, and its layer is printed at
the top for comparison.""")


if __name__ == "__main__":
    main()