"""
What kind of protest got suppressed?

THE GAP THIS FILLS
------------------
The distributional analysis established WHERE the effect sits but not
WHAT it consists of:

    mode              -0.544 vs -0.615    essentially unchanged
    10th percentile   -0.163              barely moves
    90th percentile   -0.858              moves a lot
    99th percentile   +0.170              unchanged

and the mixture fit showed the activated-state occupancy falling from
0.662 to 0.386 with the component means holding steady. So
moderately-high protest turns became rarer while typical and extreme
turns did not change.

Every one of those statements is about numbers. None says what the
suppressed turns actually WERE. This prints them.

WHAT TO LOOK FOR
----------------
The 90th-percentile band is where the compression happened. Comparing
the turns in that band across conditions answers a question the
statistics cannot:

  - is the suppressed material a particular KIND of protest --
    demands, accusations, repeated bids, escalation after refusal?
  - does the learning child's band contain milder versions of the same
    thing, or different content entirely?
  - what sits at the extreme (99th) in both, given that the extreme
    did not change?

Also printed: the mother's line and the child's original bid for each
turn, since a protest turn is a response to something and reading it
alone loses the situation.

Usage:
    python inspect_band.py \\
        --a learning "runs_learning/*_on.jsonl" \\
        --b frozen   "runs_learning/*_off.jsonl"
    python inspect_band.py ... --low 0.85 --high 0.95 --n 12
"""

import argparse
import glob
import json

from fep_layer import CONCEPT_WEIGHTS


def load_rows(pattern, concept):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                a = r.get("activations")
                if a and concept in a:
                    r["_v"] = a[concept]
                    r["_src"] = path.split("/")[-1]
                    rows.append(r)
    return rows


def quantile(vals, q):
    s = sorted(vals)
    i = q * (len(s) - 1)
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (i - lo) * (s[hi] - s[lo])


def band(rows, lo_q, hi_q, concept):
    vals = [r["_v"] for r in rows]
    lo, hi = quantile(vals, lo_q), quantile(vals, hi_q)
    sel = [r for r in rows if lo <= r["_v"] <= hi]
    return sorted(sel, key=lambda r: -r["_v"]), lo, hi


def show(label, rows, n, concept, show_context):
    print(f"\n{'=' * 78}")
    print(label)
    print(f"{'=' * 78}")
    if not rows:
        print("  (no turns in this band)")
        return
    for r in rows[:n]:
        print(f"\n  [{concept} = {r['_v']:+.3f}]  {r['_src']}  "
              f"ep{r.get('episode')} idx{r.get('index')}  "
              f"({r.get('availability', '?')}, {r.get('event_type', '?')})")
        if show_context:
            if r.get("child_bid"):
                print(f"      child bid  > {r['child_bid']}")
            if r.get("mother_response"):
                print(f"      mother     > {r['mother_response']}")
        print(f"      CHILD      > {r.get('child_reaction', '(missing)')}")


def summarize_events(rows, label):
    from collections import Counter
    ev = Counter(r.get("event_type", "?") for r in rows)
    av = Counter(r.get("availability", "?") for r in rows)
    n = len(rows) or 1
    print(f"\n  {label} ({n} turns)")
    print("    event types:  " + ", ".join(
        f"{k} {v/n:.0%}" for k, v in ev.most_common(5)))
    print("    availability: " + ", ".join(
        f"{k} {v/n:.0%}" for k, v in av.most_common(3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs=2, metavar=("NAME", "GLOB"), required=True)
    ap.add_argument("--b", nargs=2, metavar=("NAME", "GLOB"), required=True)
    ap.add_argument("--concept", default="protesting",
                    choices=list(CONCEPT_WEIGHTS))
    ap.add_argument("--low", type=float, default=0.85)
    ap.add_argument("--high", type=float, default=0.95)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--extremes", action="store_true",
                    help="also show the top of the distribution, which did "
                         "NOT differ between conditions")
    args = ap.parse_args()

    na, pa = args.a
    nb, pb = args.b
    A = load_rows(pa, args.concept)
    B = load_rows(pb, args.concept)
    if not A or not B:
        raise SystemExit("no rows loaded -- check the globs")

    has_ctx = any("mother_response" in r for r in A + B)
    if not has_ctx:
        print("note: these files have no mother_response -- "
              "extract_emotions.py --score-all drops it, so only the "
              "child's reaction is shown.")

    bandA, loA, hiA = band(A, args.low, args.high, args.concept)
    bandB, loB, hiB = band(B, args.low, args.high, args.concept)

    print("=" * 78)
    print(f"THE {args.low:.0%}-{args.high:.0%} BAND  --  {args.concept}")
    print("=" * 78)
    print(f"""
This is where the compression happened: the 90th percentile moved
-0.858 while the mode and the 99th did not. The turns below are what
that number is made of.

  {na:<10} band spans {loA:+.3f} to {hiA:+.3f}   ({len(bandA)} turns)
  {nb:<10} band spans {loB:+.3f} to {hiB:+.3f}   ({len(bandB)} turns)""")

    summarize_events(bandA, na)
    summarize_events(bandB, nb)

    show(f"{na.upper()} — turns in the band", bandA, args.n,
         args.concept, has_ctx)
    show(f"{nb.upper()} — turns in the band", bandB, args.n,
         args.concept, has_ctx)

    if args.extremes:
        topA = sorted(A, key=lambda r: -r["_v"])[:args.n // 2]
        topB = sorted(B, key=lambda r: -r["_v"])[:args.n // 2]
        print(f"\n\n{'#' * 78}")
        print("THE EXTREME TAIL — which did NOT differ between conditions")
        print(f"{'#' * 78}")
        show(f"{na.upper()} — highest {args.concept}", topA,
             args.n // 2, args.concept, has_ctx)
        show(f"{nb.upper()} — highest {args.concept}", topB,
             args.n // 2, args.concept, has_ctx)

    print(f"""

{'=' * 78}
READING THIS
{'=' * 78}

The statistics say moderately-high protest became rarer while typical
and extreme protest did not change. What they cannot say is what the
suppressed material was.

Read the two bands against each other:

  - if {nb}'s band contains demands, accusations or escalation that
    {na}'s does not, the learning removed a specific KIND of protest;
  - if both contain the same kinds and {na}'s are simply milder, the
    change is intensity within a category after all, and the
    occupancy story needs revisiting;
  - if the bands look alike, the difference is in how OFTEN such turns
    occur rather than in what they are -- which is what the mixture
    fit implies and would be confirmation rather than new information.

Check the event-type and availability breakdowns too: if the
suppressed turns cluster on particular event types, the effect is
situational rather than general.""")


if __name__ == "__main__":
    main()