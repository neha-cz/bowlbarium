"""
Length-matched drift analysis.

THE PROBLEM THIS SOLVES
-----------------------
In the learning arm, response length rose from 5.4 words in the first
quartile to 14.5 in the last, while the frozen arm went 5.0 -> 3.9. So
in the learning condition length is nearly collinear with time.

That breaks residualisation as a test. Regressing out length regresses
out everything that drifts over time, real effects included -- verified
on synthetic data, where a genuinely length-independent drift was cut to
9% of its magnitude alongside a purely length-driven one cut to 5%.
Residualisation over-kills under collinearity, so it cannot decide the
question.

WHAT THIS DOES INSTEAD
----------------------
Rather than regressing length out, it holds length CONSTANT. Turns are
bucketed by word count, and within each bucket the early turns are
compared with the late turns. A 10-word turn early is compared only
against a 10-word turn late.

If the child's affect drifts within a length bucket, that drift is not
length -- both sides of the comparison have the same length by
construction.

This is a stronger test than residualisation because it makes no
assumption that the relationship between length and activation is
linear, and it cannot silently remove a time-varying effect.

THE COST
--------
Matching discards turns with no counterpart in the other period. If the
learning child's early turns are all 3-6 words and its late turns all
12-18, the overlapping region may be thin, and the test then runs on a
small subsample. The script reports how many turns survive matching, and
that number has to be read before the result means anything.

Usage:
    python length_matched.py --seeds 600 601 602
    python length_matched.py --seeds 600 601 602 --bucket 3
"""

import argparse
import json
import os
from collections import defaultdict

from fep_layer import CONCEPT_WEIGHTS


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


def words(row):
    return len(row["child_reaction"].split())


def matched_drift(rows, frac, bucket, min_per_cell):
    """Drift per concept, comparing early vs late WITHIN length buckets.

    Returns (drift_by_concept, n_matched, bucket_detail).
    """
    q = max(5, int(len(rows) * frac))
    early, late = rows[:q], rows[-q:]

    def by_bucket(subset):
        out = defaultdict(list)
        for r in subset:
            out[words(r) // bucket].append(r)
        return out

    eb, lb = by_bucket(early), by_bucket(late)
    shared = sorted(set(eb) & set(lb))

    drift = {c: [] for c in CONCEPT_WEIGHTS}
    weights = []
    detail = []

    for b in shared:
        e, l = eb[b], lb[b]
        if len(e) < min_per_cell or len(l) < min_per_cell:
            continue
        # Weight each bucket by how much evidence it carries.
        w = min(len(e), len(l))
        weights.append(w)
        detail.append((b * bucket, (b + 1) * bucket - 1, len(e), len(l)))
        for c in CONCEPT_WEIGHTS:
            d = (mean([r["activations"][c] for r in l])
                 - mean([r["activations"][c] for r in e]))
            drift[c].append(d * w)

    total_w = sum(weights)
    if total_w == 0:
        return None, 0, detail

    return ({c: sum(v) / total_w for c, v in drift.items()},
            total_w, detail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--workdir", default="runs_learning")
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--bucket", type=int, default=3,
                    help="word-count bucket width; wider matches more "
                         "turns but controls length less tightly")
    ap.add_argument("--min-per-cell", type=int, default=3,
                    help="minimum turns on each side of a bucket for it "
                         "to count")
    args = ap.parse_args()

    per_seed = []
    for seed in args.seeds:
        on = os.path.join(args.workdir, f"L{seed}_on.jsonl")
        off = os.path.join(args.workdir, f"L{seed}_off.jsonl")
        rows_on, rows_off = load(on), load(off)

        d_on, n_on, det_on = matched_drift(
            rows_on, args.frac, args.bucket, args.min_per_cell)
        d_off, n_off, _ = matched_drift(
            rows_off, args.frac, args.bucket, args.min_per_cell)

        print(f"\n{'=' * 74}")
        print(f"SEED {seed}")
        print(f"{'=' * 74}")

        if d_on is None or d_off is None:
            print("  no length buckets with enough turns on both sides.")
            print("  The early and late length distributions do not "
                  "overlap enough")
            print("  to compare. Try --bucket 5 or --min-per-cell 2.")
            per_seed.append(None)
            continue

        print(f"\nmatched turns: learning {n_on}, frozen {n_off}")
        print(f"\n{'word range':>12} {'early n':>9} {'late n':>8}")
        print("-" * 74)
        for lo, hi, ne, nl in det_on:
            print(f"{f'{lo}-{hi}':>12} {ne:>9} {nl:>8}")

        diff = {c: d_on[c] - d_off[c] for c in CONCEPT_WEIGHTS}
        per_seed.append(diff)
        print(f"\n{'concept':<14} {'learning':>11} {'frozen':>10} "
              f"{'difference':>12}")
        print("-" * 74)
        for c in CONCEPT_WEIGHTS:
            print(f"{c:<14} {d_on[c]:>+11.3f} {d_off[c]:>+10.3f} "
                  f"{diff[c]:>+12.3f}")

    valid = [d for d in per_seed if d is not None]
    if len(valid) < 2:
        print("\nToo few usable seeds to aggregate.")
        return

    print(f"\n{'=' * 74}")
    print(f"AGGREGATE, LENGTH-MATCHED  ({len(valid)} seeds)")
    print(f"{'=' * 74}")
    print(f"\n{'concept':<14} {'mean':>10} {'sd':>9} {'mean/sd':>9} "
          f"{'consistent':>11}  per-seed")
    print("-" * 74)
    survivors = []
    for c in CONCEPT_WEIGHTS:
        diffs = [d[c] for d in valid]
        sd = stdev(diffs)
        ratio = abs(mean(diffs)) / sd if sd > 0 else float("inf")
        signs = [1 if x > 0 else (-1 if x < 0 else 0) for x in diffs]
        ok = all(s == signs[0] for s in signs) and signs[0] != 0
        if ok and ratio > 1.5:
            survivors.append(c)
        shown = " ".join(f"{x:+.2f}" for x in diffs)
        print(f"{c:<14} {mean(diffs):>+10.3f} {sd:>9.3f} {ratio:>9.2f} "
              f"{'YES' if ok else 'no':>11}  {shown}")

    print(f"\n{'=' * 74}")
    print("READING THIS")
    print(f"{'=' * 74}")
    print(f"\nsurvive length matching: {survivors or 'none'}")
    print("""
Within a length bucket both sides have the same word count, so drift
here is not verbosity. Matching breaks the length-time collinearity
rather than regressing through it, which residualisation cannot do.

BUT THIS TEST IS CONSERVATIVE, NOT DECISIVE. On synthetic data with a
purely length-driven effect and a genuinely length-independent one, it
correctly killed the artifact (-0.05 against a true 0) but recovered
only a third of the real effect (-0.24 against a true -0.75), with
mean/sd below 1. Matching discards non-overlapping turns and estimates
within thin buckets, so it attenuates and adds noise.

Read results accordingly:
  survives matching  -> strong evidence the effect is not length
  fails matching     -> WEAK evidence against; the test loses real
                        effects at this sample size, so a null here
                        does not settle the question

Check the matched-turn counts above. If matching kept only a small
fraction of the run, the seed-to-seed spread will be wide and neither
outcome carries much weight.""")


if __name__ == "__main__":
    main()