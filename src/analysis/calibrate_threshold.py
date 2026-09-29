"""
Calibrate the surprise threshold.

THE PROBLEM
-----------
With surprise_threshold = 1.0 the gate fired on 48% of turns in the
high-reliability condition and 70-73% in the low. A gate that fires on
half of all turns is not selecting anything -- on the synthetic stable
series in fep_layer.py it fired 0/8, which is the intended behaviour.
Ordinary interaction should not trigger a weight update.

THE SUBTLETY
------------
The threshold must be calibrated on the REFERENCE condition alone --
high reliability, the "ordinary" environment -- and then applied
unchanged to both. Calibrating on the pooled data would force both
conditions toward the same fire rate and erase exactly the difference
being measured. The reference sets what counts as normal; the other
condition is then free to deviate from it.

Usage:
    python src/analysis/calibrate_threshold.py affect_high_reliability.jsonl \\
                                  affect_low_reliability.jsonl
    python src/analysis/calibrate_threshold.py ... --target 0.15
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

from fep_layer import AffectTracker, PopulationBaseline, CONCEPT_WEIGHTS


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def surprises_for(rows, population, threshold=1.0):
    """Run the tracker and return the per-turn |surprise| values.

    The threshold does not affect the surprise values themselves, only
    the is_surprising flag, so any value works for collection.
    """
    tracker = AffectTracker(population=population,
                            surprise_threshold=threshold)
    vals = []
    current_episode = None
    for r in rows:
        if current_episode is not None and r["episode"] != current_episode:
            tracker.episode_boundary()
        current_episode = r["episode"]
        reading = tracker.observe(r["activations"], episode=r["episode"])
        vals.append(abs(reading.surprise))
    return vals


def percentile(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    idx = q * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def histogram(vals, bins=10, width=44):
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in vals:
        i = min(int((v - lo) / step), bins - 1)
        counts[i] += 1
    peak = max(counts) or 1
    for i, c in enumerate(counts):
        left = lo + i * step
        bar = "#" * int(width * c / peak)
        print(f"  {left:5.2f}-{left + step:5.2f} {c:>4} {bar}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", help="affect file for the REFERENCE "
                                      "condition (usually high reliability)")
    ap.add_argument("other", nargs="?", help="affect file for comparison")
    ap.add_argument("--target", type=float, default=0.15,
                    help="fraction of REFERENCE turns the gate should fire "
                         "on (default 0.15)")
    args = ap.parse_args()

    ref_rows = load(args.reference)
    other_rows = load(args.other) if args.other else None

    # Shared population baseline, same as analyze_affect.py builds.
    population = PopulationBaseline(min_n=30)
    all_rows = ref_rows + (other_rows or [])
    for r in all_rows:
        population.update(sum(CONCEPT_WEIGHTS.get(c, 0) * v
                              for c, v in r["activations"].items()))

    ref_vals = surprises_for(ref_rows, population)
    other_vals = surprises_for(other_rows, population) if other_rows else None

    print("=" * 66)
    print("SURPRISE DISTRIBUTION -- reference condition")
    print("=" * 66)
    print(f"file: {args.reference}   n={len(ref_vals)}")
    print(f"mean |surprise|: {sum(ref_vals) / len(ref_vals):.3f}   "
          f"max: {max(ref_vals):.3f}")
    print()
    histogram(ref_vals)

    srt = sorted(ref_vals)
    print(f"\n{'percentile':>12} {'|surprise|':>12}")
    print("-" * 66)
    for q in (0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99):
        print(f"{q:>12.0%} {percentile(srt, q):>12.3f}")

    recommended = percentile(srt, 1 - args.target)

    print(f"\n{'=' * 66}")
    print("RECOMMENDED THRESHOLD")
    print(f"{'=' * 66}")
    print(f"target fire rate on reference: {args.target:.0%}")
    print(f"threshold:                     {recommended:.3f}")

    print(f"\n{'threshold':>10} {'ref fire':>10} "
          f"{'other fire':>12} {'gap':>10}")
    print("-" * 66)
    for t in (1.0, 1.5, 2.0, 2.5, 3.0, recommended):
        ref_rate = sum(v >= t for v in ref_vals) / len(ref_vals)
        if other_vals:
            oth_rate = sum(v >= t for v in other_vals) / len(other_vals)
            tag = "  <- recommended" if abs(t - recommended) < 1e-9 else ""
            print(f"{t:>10.2f} {ref_rate:>10.0%} {oth_rate:>12.0%} "
                  f"{oth_rate - ref_rate:>+10.0%}{tag}")
        else:
            print(f"{t:>10.2f} {ref_rate:>10.0%}")

    print(f"\nSet surprise_threshold={recommended:.2f} when constructing")
    print("AffectTracker. Check the gap column: if it collapses at the")
    print("recommended threshold, the condition difference lives in")
    print("marginal surprises rather than large ones, which is worth")
    print("knowing before building a learning loop on top of it.")


if __name__ == "__main__":
    main()