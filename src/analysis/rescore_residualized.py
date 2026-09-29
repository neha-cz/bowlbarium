"""
Re-analyse the multi-seed learning runs with length regressed out.

WHY
---
The replicated result was comforted +1.068 (mean/sd 3.42) and protesting
-0.829 (mean/sd 2.82). But two things about it invite a length
explanation:

  - the revision prompt now explicitly asks for two or three sentences,
    so the learning child is being trained toward longer output;
  - of the four concepts, comforted carries the strongest residual
    correlation with response length (+0.349 even after last-token
    pooling fixed the worst of it).

So "the learning child became more comforted" and "the learning child
started talking more" are not yet distinguishable.

This regresses response length out of each concept, pooled across the
learning and frozen files for a seed, and recomputes the drift
difference. No generation is rerun -- the affect files already contain
child_reaction, which is all the residualisation needs.

READING IT
----------
  comforted survives     -> the effect is not a length artifact
  comforted collapses    -> it was length; the finding is protest
                            reduction alone, which is still real
  protesting collapses   -> the whole result was length

Note the honest caveat from earlier in this project: a child that
learned to say more IS a behavioural change, and residualising removes
real signal along with the confound. The point is to see whether the
conclusion depends on the choice, not to declare one version correct.

Usage:
    python src/analysis/rescore_residualized.py --seeds 600 601 602
    python src/analysis/rescore_residualized.py --seeds 600 601 602 --workdir runs/learning
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
import os

from fep_layer import CONCEPT_WEIGHTS
from drift_analysis import quartile_drift
from extract_emotions import residualize_length


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx_, my = mean(xs), mean(ys)
    num = sum((x - mx_) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx_) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def drift_for_seed(workdir, seed, frac, residualize):
    on = os.path.join(workdir, f"L{seed}_on.jsonl")
    off = os.path.join(workdir, f"L{seed}_off.jsonl")
    rows_on, rows_off = load(on), load(off)

    if residualize:
        # Pool both files before residualising, matching how they were
        # z-scored together -- otherwise the two get different
        # length corrections and become incomparable.
        pooled = rows_on + rows_off
        residualize_length(pooled)
        rows_on = pooled[:len(rows_on)]
        rows_off = pooled[len(rows_on):]

    d_on = quartile_drift(rows_on, frac)
    d_off = quartile_drift(rows_off, frac)
    return ({c: d_on[c]["drift"] - d_off[c]["drift"] for c in CONCEPT_WEIGHTS},
            rows_on, rows_off)


def report(label, per_seed):
    concepts = list(CONCEPT_WEIGHTS)
    print(f"\n{'=' * 74}")
    print(label)
    print(f"{'=' * 74}")
    print(f"\n{'concept':<14} {'mean':>10} {'sd':>9} {'mean/sd':>9} "
          f"{'consistent':>11}  per-seed")
    print("-" * 74)
    out = {}
    for c in concepts:
        diffs = [d[c] for d in per_seed]
        sd = stdev(diffs)
        ratio = abs(mean(diffs)) / sd if sd > 0 else float("inf")
        signs = [1 if x > 0 else (-1 if x < 0 else 0) for x in diffs]
        ok = all(s == signs[0] for s in signs) and signs[0] != 0
        shown = " ".join(f"{x:+.2f}" for x in diffs)
        print(f"{c:<14} {mean(diffs):>+10.3f} {sd:>9.3f} {ratio:>9.2f} "
              f"{'YES' if ok else 'no':>11}  {shown}")
        out[c] = {"mean": mean(diffs), "sd": sd, "ratio": ratio,
                  "consistent": ok}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--workdir", default="runs/learning")
    ap.add_argument("--frac", type=float, default=0.25)
    args = ap.parse_args()

    raw_per_seed, res_per_seed = [], []
    for seed in args.seeds:
        raw, rows_on, rows_off = drift_for_seed(
            args.workdir, seed, args.frac, residualize=False)
        res, _, _ = drift_for_seed(
            args.workdir, seed, args.frac, residualize=True)
        raw_per_seed.append(raw)
        res_per_seed.append(res)

        if seed == args.seeds[0]:
            pooled = rows_on + rows_off
            lengths = [len(r["child_reaction"].split()) for r in pooled]
            print(f"{'=' * 74}")
            print("LENGTH CHECK  (seed "
                  f"{seed}, learning vs frozen)")
            print(f"{'=' * 74}")
            lo = [len(r["child_reaction"].split()) for r in rows_on]
            lf = [len(r["child_reaction"].split()) for r in rows_off]
            print(f"\nmean words -- learning {mean(lo):.1f}, "
                  f"frozen {mean(lf):.1f}")
            q = max(5, int(len(rows_on) * args.frac))
            print(f"learning: first {q} turns {mean(lo[:q]):.1f} words, "
                  f"last {q} {mean(lo[-q:]):.1f}")
            print(f"frozen:   first {q} turns {mean(lf[:q]):.1f} words, "
                  f"last {q} {mean(lf[-q:]):.1f}")
            print("\ncorrelation of each concept with length (pooled):")
            for c in CONCEPT_WEIGHTS:
                vals = [r["activations"][c] for r in pooled]
                print(f"    {c:<12} {pearson(lengths, vals):+.3f}")
            print("\nIf the learning child's responses lengthen across the")
            print("run while the frozen child's do not, length is a live")
            print("explanation for any drift in a length-correlated")
            print("concept.")

    raw = report("AS SCORED  (length left in)", raw_per_seed)
    res = report("RESIDUALISED  (length regressed out)", res_per_seed)

    print(f"\n{'=' * 74}")
    print("DOES THE CONCLUSION DEPEND ON THE CHOICE?")
    print(f"{'=' * 74}")
    print(f"\n{'concept':<14} {'raw mean':>11} {'resid mean':>12} "
          f"{'kept':>8} {'raw m/sd':>10} {'res m/sd':>10}")
    print("-" * 74)
    for c in CONCEPT_WEIGHTS:
        kept = (abs(res[c]["mean"]) / abs(raw[c]["mean"])
                if raw[c]["mean"] else 0.0)
        print(f"{c:<14} {raw[c]['mean']:>+11.3f} {res[c]['mean']:>+12.3f} "
              f"{kept:>7.0%} {raw[c]['ratio']:>10.2f} "
              f"{res[c]['ratio']:>10.2f}")

    survivors = [c for c in CONCEPT_WEIGHTS
                 if res[c]["consistent"] and res[c]["ratio"] > 1.5]
    print(f"\nsurvive residualisation (consistent, mean/sd > 1.5): "
          f"{survivors or 'none'}")

    print(f"\n{'=' * 74}")
    print("A LIMITATION OF THIS TEST -- read the LENGTH CHECK above first")
    print(f"{'=' * 74}")
    print("""
If the learning child's responses LENGTHEN monotonically across the run,
then length is collinear with time, and regressing out length regresses
out everything that drifts over time -- real effects included.

Verified on synthetic data: with a purely length-driven "comforted" and
a genuinely length-independent "protesting" drift, residualisation
correctly killed comforted (5% kept) AND wrongly killed protesting (9%
kept), because in the learning arm length rose with time by
construction.

So:
  learning-arm length FLAT across the run
      -> length is not collinear with time, and this test is
         informative: what survives is real.
  learning-arm length RISING across the run
      -> this test cannot separate the two, and a null result here is
         not evidence against the effect. Compare length-matched turns
         instead (early vs late turns of similar word count), or check
         whether the same drift appears in the frozen arm, which has the
         same length distribution but no gradient steps.""")


if __name__ == "__main__":
    main()