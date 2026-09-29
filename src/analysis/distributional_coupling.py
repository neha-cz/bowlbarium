"""
Distributional coupling: no lags, no trajectories.

WHY THE PREVIOUS COUPLING TESTS WERE THE WRONG SHAPE
----------------------------------------------------
entropy_production.py established that this system has essentially no
turn-to-turn temporal structure: shuffling the turn order changed the
entropy production by less than a thousandth, in both arms, at two
resolutions. The child's own autocorrelation is +0.10 to +0.18.

Every effect that DID survive in this project came from comparing
DISTRIBUTIONS across conditions -- Result 1, Result 2, Result 3 -- not
from trajectories.

Then two coupling analyses were built on lagged temporal prediction
anyway, which is the one thing this data was already known not to
support. Both returned null, and at least one of them
(two_body.py's positive control) turned out to be mathematically empty:
r_eff = r0 + c * bid_rate exactly, so bid_rate and r_eff are perfectly
collinear and every partial correlation conditioning on both returns 0
by construction.

THE DISTRIBUTIONAL VERSION
--------------------------
With feedback on, r_eff varies across a run because the child's own
bidding moved it. So bin turns by the LEVEL of r_eff -- ignoring when
they occurred -- and ask whether the child's affect distribution
differs across bins.

No lags, no autocorrelation, no Markov assumption. This is the same
design as the reliability sweep, except availability is endogenous
rather than set externally, and the same design as the confound check
that produced the cleanest number in the project.

Reported per bin:
    mean and median protest
    P(top 15% of protest)      -- the hit-rate measure from Result 3
    event mix                  -- so composition shifts are visible

A LIMIT THAT CANNOT BE ANALYSED AWAY
------------------------------------
r_eff is CAUSED by the child's past bidding. A child that bids more
creates higher availability AND is a higher-protest child, so
covariation between them is guaranteed by the feedback rule and does
not establish that availability influenced the child.

Holding event type constant helps, and comparing within a single run
helps, but the direction of causation stays ambiguous. The honest
claim available here is covariation, not influence.

Usage:
    python src/analysis/distributional_coupling.py --runs "runs/fb/fb_*.jsonl"
    python src/analysis/distributional_coupling.py --runs "runs/fb/fb_*.jsonl" --bins 4
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
from collections import Counter, defaultdict

from fep_layer import CONCEPT_WEIGHTS


def load(patterns, concept):
    rows = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    a = r.get("activations")
                    if not a or concept not in a:
                        continue
                    if r.get("r_eff") is None:
                        continue
                    r["_v"] = a[concept]
                    r["_src"] = path.split("/")[-1]
                    rows.append(r)
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def median(xs):
    s = sorted(xs)
    n = len(s)
    return 0.0 if not n else (s[n // 2] if n % 2
                              else 0.5 * (s[n // 2 - 1] + s[n // 2]))


def quantile(vals, q):
    s = sorted(vals)
    i = q * (len(s) - 1)
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (i - lo) * (s[hi] - s[lo])


def cliffs_delta(xs, ys, cap=300, seed=0):
    import random
    rng = random.Random(seed)
    a = xs if len(xs) <= cap else [rng.choice(xs) for _ in range(cap)]
    b = ys if len(ys) <= cap else [rng.choice(ys) for _ in range(cap)]
    gt = lt = 0
    for p in a:
        for q in b:
            if p > q:
                gt += 1
            elif p < q:
                lt += 1
    n = len(a) * len(b)
    return (gt - lt) / n if n else 0.0


def perm_p(xs, ys, n_perm, seed=0):
    """Difference of means against a label-shuffled null."""
    import random
    rng = random.Random(seed)
    obs = abs(mean(xs) - mean(ys))
    pool = xs + ys
    k = len(xs)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(mean(pool[:k]) - mean(pool[k:])) >= obs:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--concept", default="protesting",
                    choices=list(CONCEPT_WEIGHTS))
    ap.add_argument("--bins", type=int, default=3)
    ap.add_argument("--band", type=float, default=0.85)
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--within-run", action="store_true",
                    help="centre protest within each run first, so "
                         "between-run differences cannot drive the result")
    args = ap.parse_args()

    rows = load(args.runs, args.concept)
    if len(rows) < 60:
        raise SystemExit(
            f"only {len(rows)} turns with r_eff -- this needs data from "
            "learning_loop.py --feedback")

    print("=" * 78)
    print(f"DISTRIBUTIONAL COUPLING  --  {args.concept}")
    print("=" * 78)
    print(f"""
No lags and no trajectory assumptions: turns are grouped by the LEVEL
of caregiver availability, regardless of when they occurred.

{len(rows)} turns from {len(set(r['_src'] for r in rows))} run(s)""")

    if args.within_run:
        by_run = defaultdict(list)
        for r in rows:
            by_run[r["_src"]].append(r)
        for rs in by_run.values():
            m = mean([r["_v"] for r in rs])
            for r in rs:
                r["_v"] -= m
        print("protest centred within run")

    # bin by r_eff level, using quantiles so bins are populated
    reffs = [r["r_eff"] for r in rows]
    edges = [quantile(reffs, i / args.bins) for i in range(1, args.bins)]
    print(f"r_eff range {min(reffs):.3f} to {max(reffs):.3f}, "
          f"bin edges {[round(e, 3) for e in edges]}")

    def which(v):
        for i, e in enumerate(edges):
            if v < e:
                return i
        return len(edges)

    groups = defaultdict(list)
    for r in rows:
        groups[which(r["r_eff"])].append(r)

    cut = quantile([r["_v"] for r in rows], args.band)

    print(f"\n{'bin':>5} {'r_eff':>14} {'n':>6} {'mean':>9} {'median':>9} "
          f"{f'P(top {1-args.band:.0%})':>12}")
    print("-" * 78)
    stats = []
    for b in sorted(groups):
        g = groups[b]
        vals = [r["_v"] for r in g]
        lo = min(r["r_eff"] for r in g)
        hi = max(r["r_eff"] for r in g)
        hit = sum(1 for v in vals if v >= cut) / len(vals)
        stats.append((b, g, vals, hit))
        print(f"{b:>5} {f'{lo:.2f}-{hi:.2f}':>14} {len(g):>6} "
              f"{mean(vals):>+9.3f} {median(vals):>+9.3f} {hit:>11.1%}")

    # lowest vs highest availability
    lo_b, lo_g, lo_v, lo_h = stats[0]
    hi_b, hi_g, hi_v, hi_h = stats[-1]
    d_mean = mean(hi_v) - mean(lo_v)
    d_med = median(hi_v) - median(lo_v)
    cd = cliffs_delta(hi_v, lo_v)
    p = perm_p(list(hi_v), list(lo_v), args.permutations)

    print(f"\n{'=' * 78}")
    print("LOWEST vs HIGHEST AVAILABILITY")
    print(f"{'=' * 78}")
    print(f"""
  difference of means    {d_mean:+.3f}
  difference of medians  {d_med:+.3f}
  Cliff's delta          {cd:+.3f}
  permutation p          {p:.4f}   ({args.permutations} shuffles)
  hit rate               {hi_h:.1%} at high availability against
                         {lo_h:.1%} at low""")

    # event mix, to see whether composition rather than level shifts
    print(f"\n{'=' * 78}")
    print("EVENT MIX BY AVAILABILITY BIN")
    print(f"{'=' * 78}")
    events = sorted({r.get("event_type", "?") for r in rows})
    print(f"\n{'event':<18}" + "".join(f"{f'bin {b}':>9}"
                                       for b, *_ in stats))
    print("-" * 78)
    for e in events:
        row = f"{e:<18}"
        for b, g, _, _ in stats:
            row += f"{sum(1 for r in g if r.get('event_type')==e)/len(g):>8.0%} "
        print(row)
    print("""
  If the event mix differs across bins, availability and situation are
  confounded -- the child may face different situations at different
  availability levels, and that alone would move protest.""")

    print(f"\n{'=' * 78}")
    print("READING THIS")
    print(f"{'=' * 78}")

    # is the event mix stable?
    max_shift = 0.0
    for e in events:
        shares = [sum(1 for r in g if r.get("event_type") == e) / len(g)
                  for _, g, _, _ in stats]
        max_shift = max(max_shift, max(shares) - min(shares))

    if p < 0.05 and abs(cd) > 0.15:
        print(f"""
The child's protest distribution differs across availability levels
(Cliff's delta {cd:+.3f}, p = {p:.4f}). Child affect and caregiver
availability covary at the distributional level, which the lagged
tests could not detect because this system has almost no turn-to-turn
temporal structure to carry a lagged signal.""")
    elif p < 0.05:
        print(f"""
The means differ (p = {p:.4f}) but Cliff's delta is small
({cd:+.3f}), so a randomly drawn turn from one availability level is
barely more likely to show higher protest than one from another. Real
at the aggregate, negligible per turn.""")
    else:
        print(f"""
No difference in the child's protest distribution across availability
levels (p = {p:.4f}, Cliff's delta {cd:+.3f}). Caregiver availability
and child affect do not covary, even without any temporal assumption.

Given that r_eff is CAUSED by the child's own bidding, some
covariation was almost guaranteed by the feedback rule -- so its
absence is a stronger null than it looks.""")

    if max_shift > 0.10:
        print(f"""
CAUTION: the event mix shifts by up to {max_shift:.0%} across bins, so
availability and situation are confounded here. Any difference above
may be the child facing different situations rather than responding to
availability. Re-run this within a single event type to separate
them.""")

    print("""
THE LIMIT THAT REMAINS. r_eff is a function of the child's own past
bidding, so a child that bids more both raises availability and is a
higher-protest child. Covariation therefore does not establish
influence in either direction, and no amount of binning fixes that --
it is a property of the feedback design, not of the analysis.""")


if __name__ == "__main__":
    main()