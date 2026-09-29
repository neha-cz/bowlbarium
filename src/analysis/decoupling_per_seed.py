"""
Per-seed decoupling.

WHY THIS CHECK
--------------
The decoupling result -- learning flattens the situation-to-objection
mapping, lift spread falling from 1.247 to 0.432 -- was computed on
POOLED data across seeds. That is the same footing on which four
earlier findings in this project sat before per-seed examination
killed them:

    gate rate          22-point single-seed effect, averaged to zero
    withdrawn          consistent across 3 seeds, reversed on 5 more
    distressed         replicated, but also appeared in the null control
    two-state occupancy  replicated per-seed, then retired for a
                         measurement reason

Pooling is particularly dangerous here. If seeds differ in their
overall objection level, the pooled distribution is a mixture, and a
band defined on the pooled data can look differently distributed
across situations for reasons that have nothing to do with learning.

WHAT THIS DOES
--------------
Computes the lift spread within each seed's learning and frozen arms
separately, using a band threshold defined within that arm, and
reports:

    spread frozen, spread learning, change     per seed
    sign consistency across seeds
    mean and mean/sd of the change

Decoupling holds only if the spread falls in every seed. A single seed
driving a pooled result is exactly the pattern this project has seen
repeatedly.

Usage:
    python src/analysis/decoupling_per_seed.py \\
        --pair 600 runs/learning/L600_on.jsonl runs/learning/L600_off.jsonl \\
        --pair 601 runs/learning/L601_on.jsonl runs/learning/L601_off.jsonl \\
        --pair 602 runs/learning/L602_on.jsonl runs/learning/L602_off.jsonl
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


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def quantile(v, q):
    s = sorted(v)
    i = q * (len(s) - 1)
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (i - lo) * (s[hi] - s[lo])


def load(path, concept):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            a = r.get("activations")
            if a and concept in a:
                r["_v"] = a[concept]
                rows.append(r)
    return rows


def lift_spread(rows, band_q, events, min_n):
    """SD of lift across situation types, within one arm.

    Lift = a situation's share of the high-objection band divided by
    its share of all turns. 1.0 means no relationship; spread near
    zero means the band is distributed in proportion to how often
    situations occur.
    """
    cut = quantile([r["_v"] for r in rows], band_q)
    tot, hit = defaultdict(int), defaultdict(int)
    for r in rows:
        e = r.get("event_type", "?")
        tot[e] += 1
        hit[e] += r["_v"] >= cut
    use = [e for e in events if tot[e] >= min_n]
    n_all = sum(tot[e] for e in use)
    n_band = sum(hit[e] for e in use)
    if n_band == 0 or len(use) < 4:
        return None, {}, len(use)
    lifts = {}
    for e in use:
        lifts[e] = ((hit[e] / n_band) / (tot[e] / n_all)
                    if tot[e] else 0.0)
    vals = list(lifts.values())
    return stdev(vals), lifts, len(use)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=3, action="append",
                    metavar=("SEED", "LEARNING", "FROZEN"), required=True)
    ap.add_argument("--concept", default="protesting")
    ap.add_argument("--band", type=float, default=0.85)
    ap.add_argument("--min-n", type=int, default=8,
                    help="minimum turns of a situation type within one "
                         "arm for it to be included")
    args = ap.parse_args()

    print("=" * 78)
    print(f"PER-SEED DECOUPLING  --  {args.concept}")
    print("=" * 78)
    print("""
Lift spread is computed WITHIN each arm, with the band threshold set
inside that arm. Pooling across seeds was how four earlier findings in
this project survived until they were checked.""")

    # situation types present everywhere
    all_events = None
    loaded = []
    for seed, lp, fp in args.pair:
        L, F = load(lp, args.concept), load(fp, args.concept)
        if len(L) < 60 or len(F) < 60:
            print(f"\nseed {seed}: too few turns ({len(L)}, {len(F)}) "
                  f"-- skipped")
            continue
        ev = set(r.get("event_type", "?") for r in L) & \
             set(r.get("event_type", "?") for r in F)
        all_events = ev if all_events is None else (all_events & ev)
        loaded.append((seed, L, F))
    if len(loaded) < 2:
        raise SystemExit("need at least two usable pairs")
    events = sorted(all_events)

    print(f"\n{len(loaded)} seed pair(s), {len(events)} shared situation "
          f"types")
    print(f"\n{'seed':>6} {'n L/F':>12} {'types':>6} {'spread F':>10} "
          f"{'spread L':>10} {'change':>9}")
    print("-" * 78)
    changes = []
    per_seed_lifts = []
    for seed, L, F in loaded:
        sf, lift_f, nf = lift_spread(F, args.band, events, args.min_n)
        sl, lift_l, nl = lift_spread(L, args.band, events, args.min_n)
        if sf is None or sl is None:
            print(f"{seed:>6} {'--':>12} {'too few types':>6}")
            continue
        changes.append(sl - sf)
        per_seed_lifts.append((seed, lift_f, lift_l))
        print(f"{seed:>6} {f'{len(L)}/{len(F)}':>12} {min(nf,nl):>6} "
              f"{sf:>10.3f} {sl:>10.3f} {sl-sf:>+9.3f}")

    if len(changes) < 2:
        raise SystemExit("too few usable pairs")

    signs = [1 if c > 0 else (-1 if c < 0 else 0) for c in changes]
    consistent = all(s == signs[0] for s in signs) and signs[0] != 0
    sd = stdev(changes)
    ratio = abs(mean(changes)) / sd if sd > 0 else float("inf")

    print("-" * 78)
    print(f"{'mean':>6} {'':>12} {'':>6} {'':>10} {'':>10} "
          f"{mean(changes):>+9.3f}")
    print(f"{'sd':>6} {'':>12} {'':>6} {'':>10} {'':>10} {sd:>9.3f}")

    # which situations move, per seed
    print(f"\n{'=' * 78}")
    print("PER-SITUATION LIFT CHANGE, BY SEED")
    print(f"{'=' * 78}")
    common = sorted(set.intersection(
        *[set(lf) & set(ll) for _, lf, ll in per_seed_lifts]))
    print(f"\n{'situation':<17}" + "".join(
        f"{s:>11}" for s, _, _ in per_seed_lifts))
    print("-" * 78)
    toward_one = defaultdict(list)
    for e in common:
        row = f"{e:<17}"
        for seed, lf, ll in per_seed_lifts:
            row += f"{ll[e]-lf[e]:>+11.2f}"
            # did it move toward 1.0?
            toward_one[e].append(abs(ll[e] - 1) < abs(lf[e] - 1))
        print(row)

    n_toward = sum(1 for e in common if all(toward_one[e]))
    print(f"""
  situations moving TOWARD lift 1.0 in every seed: {n_toward}/{len(common)}
  (decoupling predicts most of them, from whichever side they began)""")

    print(f"\n{'=' * 78}")
    print("READING THIS")
    print(f"{'=' * 78}")
    print(f"""
  spread change: mean {mean(changes):+.3f}, sd {sd:.3f}, mean/sd {ratio:.2f}
  sign consistent across {len(changes)} seeds: {'YES' if consistent else 'no'}
  pooled figure for comparison: 1.247 -> 0.432, change -0.815""")

    if consistent and signs[0] < 0 and ratio > 1.5:
        print(f"""
The lift spread falls in every seed, with a mean well outside the
between-seed spread. Decoupling replicates per-seed and is not an
artifact of pooling.

That matters for the framing: the claim is not that objection drops on
one situation type, but that the mapping from situation to objection
flattens. A level-shift account does not predict situations moving in
OPPOSITE directions, which is what the per-situation table shows.""")
    elif consistent and signs[0] < 0:
        print(f"""
The spread falls in every seed but the mean is not well separated from
the between-seed spread (mean/sd {ratio:.2f}). Directionally
consistent, statistically thin -- more seeds would settle it.""")
    elif consistent and signs[0] > 0:
        print("""
The spread RISES in every seed -- learning concentrated the band
further rather than flattening it. That is the opposite of the pooled
result, which means the pooled figure was an artifact of mixing seeds
with different objection levels.""")
    else:
        print("""
The sign flips between seeds. The pooled decoupling result does not
survive per-seed examination, and joins the earlier findings that
looked convincing until this check was applied.

Concretely: pooling seeds with different objection levels produces a
mixture, and a band defined on the pooled data can appear
redistributed across situations for reasons unrelated to learning.""")

    print("""
CAVEATS. The band is defined within each arm, so both arms contain the
same fraction of their own turns by construction and only the
DISTRIBUTION across situations can differ -- absolute objection level
plays no part in this measure and is reported separately. Situation
types with few turns in a single run are noisy; --min-n controls
which are included and the 'types' column shows how many survived.""")


if __name__ == "__main__":
    main()