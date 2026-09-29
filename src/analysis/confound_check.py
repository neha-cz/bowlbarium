"""
Confound check for Result 3.

THE CLAIM AND THE ALTERNATIVE
-----------------------------
Result 3 states that learning changes WHICH SITUATIONS produce the
child's high-protest turns: conflict share of the top 15% of protest
falls from 47% (frozen) to 20% (learning), consistent across 6 of 6
seeds.

That reading assumes the child's REACTIVITY changed -- that a conflict
event no longer produces a high-protest response the way it used to.

There is a mundane alternative. If conflict EVENTS simply occur less
often in the learning condition, their share of the high-protest band
would fall for a reason that has nothing to do with reactivity. The
finding would then be about interaction dynamics -- what situations
arise -- rather than about how the child responds to them.

Both are real effects. They are different claims, and only one is
about the child.

WHAT DISTINGUISHES THEM
-----------------------
    base rate       how often conflict events occur at all
    hit rate        P(turn is in the top 15% of protest | conflict event)
    lift            hit rate relative to the base rate

If base rates match and hit rates differ, reactivity changed -- the
same events stop producing high protest. That supports Result 3.

If base rates differ, the composition shift is partly or wholly
explained by event frequency, and the reactivity reading does not
follow.

Note that the event sequence comes from event_generator.py and is
FIXED before either condition runs, so base rates should be identical
by construction. If they are not, something in the pipeline is
filtering turns unequally -- which would itself need explaining.

Usage:
    python src/analysis/confound_check.py \\
        --a learning "runs/learning/*_on.jsonl" \\
        --b frozen   "runs/learning/*_off.jsonl"
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
import re
from collections import Counter, defaultdict

from fep_layer import CONCEPT_WEIGHTS


def load_rows(pattern, concept):
    rows = []
    for path in sorted(glob.glob(pattern)):
        src = path.split("/")[-1]
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                a = r.get("activations")
                if not a or concept not in a:
                    continue
                r["_v"] = a[concept]
                r["_src"] = src
                rows.append(r)
    return rows


def seed_of(src):
    m = re.search(r"[LDA](\d{3})", src)
    return m.group(1) if m else src


def quantile(vals, q):
    s = sorted(vals)
    i = q * (len(s) - 1)
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (i - lo) * (s[hi] - s[lo])


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def analyse(rows, q, event="conflict"):
    """Base rate, hit rate and lift for one condition."""
    vals = [r["_v"] for r in rows]
    cut = quantile(vals, q)
    band = [r for r in rows if r["_v"] >= cut]

    n = len(rows)
    n_ev = sum(1 for r in rows if r.get("event_type") == event)
    n_band_ev = sum(1 for r in band if r.get("event_type") == event)

    base = n_ev / n if n else 0.0
    hit = n_band_ev / n_ev if n_ev else 0.0
    share = n_band_ev / len(band) if band else 0.0
    lift = share / base if base else float("nan")
    return {"n": n, "n_event": n_ev, "base": base, "hit": hit,
            "band_share": share, "lift": lift, "n_band": len(band)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs=2, metavar=("NAME", "GLOB"), required=True)
    ap.add_argument("--b", nargs=2, metavar=("NAME", "GLOB"), required=True)
    ap.add_argument("--concept", default="protesting",
                    choices=list(CONCEPT_WEIGHTS))
    ap.add_argument("--band", type=float, default=0.85)
    ap.add_argument("--event", default="conflict")
    args = ap.parse_args()

    na, pa = args.a
    nb, pb = args.b
    A = load_rows(pa, args.concept)
    B = load_rows(pb, args.concept)
    if not A or not B:
        raise SystemExit("no rows loaded")

    print("=" * 78)
    print(f"CONFOUND CHECK  --  did '{args.event}' events become rarer, or "
          f"less reactive?")
    print("=" * 78)

    ra = analyse(A, args.band, args.event)
    rb = analyse(B, args.band, args.event)

    print(f"""
  base rate   how often the event occurs at all
  hit rate    P(top {1-args.band:.0%} of protest | event occurred)
  band share  the event's share of the high-protest band
  lift        band share / base rate
""")
    print(f"{'':<14} {'turns':>8} {'events':>8} {'base':>9} {'hit':>9} "
          f"{'band share':>12} {'lift':>8}")
    print("-" * 78)
    for name, r in ((na, ra), (nb, rb)):
        print(f"{name:<14} {r['n']:>8} {r['n_event']:>8} {r['base']:>8.1%} "
              f"{r['hit']:>8.1%} {r['band_share']:>11.1%} {r['lift']:>8.2f}")

    # per seed
    print(f"\n{'=' * 78}")
    print("PER SEED")
    print(f"{'=' * 78}")
    sa, sb = defaultdict(list), defaultdict(list)
    for r in A:
        sa[seed_of(r["_src"])].append(r)
    for r in B:
        sb[seed_of(r["_src"])].append(r)
    shared = sorted(set(sa) & set(sb))

    print(f"\n{'seed':>6} {'base A':>9} {'base B':>9} {'d base':>8}   "
          f"{'hit A':>8} {'hit B':>8} {'d hit':>8}")
    print("-" * 78)
    d_base, d_hit = [], []
    for s in shared:
        x, y = analyse(sa[s], args.band, args.event), \
               analyse(sb[s], args.band, args.event)
        d_base.append(x["base"] - y["base"])
        d_hit.append(x["hit"] - y["hit"])
        print(f"{s:>6} {x['base']:>8.1%} {y['base']:>8.1%} "
              f"{x['base']-y['base']:>+7.1%}   {x['hit']:>7.1%} "
              f"{y['hit']:>7.1%} {x['hit']-y['hit']:>+7.1%}")

    print("-" * 78)
    print(f"{'mean':>6} {'':>9} {'':>9} {mean(d_base):>+7.1%}   "
          f"{'':>8} {'':>8} {mean(d_hit):>+7.1%}")

    # what replaced it, if anything
    print(f"\n{'=' * 78}")
    print("EVENT MIX OF THE HIGH-PROTEST BAND")
    print(f"{'=' * 78}")
    for name, rows in ((na, A), (nb, B)):
        vals = [r["_v"] for r in rows]
        cut = quantile(vals, args.band)
        band = [r for r in rows if r["_v"] >= cut]
        c = Counter(r.get("event_type", "?") for r in band)
        allc = Counter(r.get("event_type", "?") for r in rows)
        print(f"\n  {name}")
        print(f"    {'event':<18} {'in band':>9} {'overall':>9} {'lift':>8}")
        for ev, k in c.most_common(6):
            b = k / len(band)
            o = allc[ev] / len(rows)
            print(f"    {ev:<18} {b:>8.1%} {o:>8.1%} "
                  f"{b/o if o else float('nan'):>8.2f}")

    # verdict
    print(f"\n{'=' * 78}")
    print("READING THIS")
    print(f"{'=' * 78}")

    mb, mh = mean(d_base), mean(d_hit)
    base_signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in d_base]
    hit_signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in d_hit]
    hit_consistent = (all(s == hit_signs[0] for s in hit_signs)
                      and hit_signs[0] != 0)

    print(f"""
  base-rate difference  {mb:+.1%}
  hit-rate difference   {mh:+.1%}   (sign consistent: "
{'YES' if hit_consistent else 'no'})""")

    if abs(mb) < 0.02 and abs(mh) > 0.05 and hit_consistent:
        print(f"""
The base rates match -- '{args.event}' events occur equally often in
both conditions -- while the hit rate differs and does so consistently
across seeds. The same events stop producing high-protest responses
under learning.

That is a change in the child's REACTIVITY, which is what Result 3
claims. The confound is ruled out.""")
    elif abs(mb) >= 0.02:
        print(f"""
The base rates DIFFER by {mb:+.1%}. '{args.event}' events are not
equally frequent across conditions, so part of the composition shift
is explained by which situations arose rather than by how the child
responded.

Since the event sequence is fixed before either run, this should not
happen -- something is filtering turns unequally, and that needs
explaining before Result 3 can be stated as a reactivity claim.""")
    else:
        print(f"""
The hit-rate difference is small ({mh:+.1%}) or inconsistent across
seeds. The composition shift in the band is not accompanied by a clear
change in how often conflict events produce high protest, so the
reactivity reading is not supported by this check.""")

    print("""
CAVEATS. The band is defined by a within-condition quantile, so it
contains the same NUMBER of turns in each condition by construction --
composition can shift without any change in absolute protest levels.
The hit rate is the quantity that avoids this, since it conditions on
the event rather than on the band.""")


if __name__ == "__main__":
    main()