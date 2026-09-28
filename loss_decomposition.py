"""
Does the loss decomposition explain Result 3?

THE DERIVATION
--------------
The objective being optimised is

    L(theta) = NLL(revision | situation) + beta * KL(theta || theta_0)

which decomposes over event types:

    L = sum_e  p(e) * E[ NLL(revision_e | situation_e) ]  +  beta * KL

The gradient contribution from event type e scales with how far the
revision sits from what the child already says. If reaction and
revision nearly coincide, there is almost nothing to learn from that
event type; if they diverge sharply, the gradient is large.

So the objective predicts:

    desensitisation(e)  proportional to  divergence(reaction_e,
                                                    revision_e)

and this is a prediction across ALL TEN event types at once, not a
single comparison.

WHY IT WOULD EXPLAIN RESULT 3
-----------------------------
Conflict is where the mother pushes back hardest, so the
reflection-informed revision departs furthest from the original
protest, so the gradient on conflict is largest, so conflict
desensitises most. That would turn the observed conflict effect from
a described pattern into a consequence of the objective.

STATUS: NOT A NOVEL MECHANISM. Gradient-magnitude-based data
selection is a mature field -- "examples with larger gradients
contribute more to model learning" is the standard motivation behind
GrADS, LESS, ClusterUCB and others. Those methods use gradient
magnitude PROSPECTIVELY to choose training data; this uses it
RETROSPECTIVELY to explain uneven behaviour change. Same mechanism,
different direction. The value here is explanatory, not novel.

WHAT IS MEASURED
----------------
    divergence(e)      how far the training target sat from what the
                       child originally said, per event type, over
                       gated turns. Three proxies, since the actual
                       gradient was not logged:
                         - token Jaccard distance
                         - length ratio
                         - the logged NLL, when present
    desensitisation(e) change in P(top 15% of protest | event e)
                       between learning and frozen

then the correlation between them across event types.

DATA REQUIREMENT
----------------
Needs the RAW learning logs, which contain `target` and
`child_reaction` on gated turns. extract_emotions.py --score-all
dropped `target` in older versions, so scored affect files will not
work -- use a backup of the learning log.

Usage:
    python loss_decomposition.py \\
        --log learn_on_backup.jsonl \\
        --learning "runs_learning/*_on.jsonl" \\
        --frozen "runs_learning/*_off.jsonl"
"""

import argparse
import glob
import json
from collections import defaultdict


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def jaccard_distance(a, b):
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta and not tb:
        return 0.0
    if not ta or not tb:
        return 1.0
    return 1 - len(ta & tb) / len(ta | tb)


def load_logs(patterns):
    """Raw learning logs -- need `target` on gated turns."""
    rows = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("target") and r.get("child_reaction"):
                        rows.append(r)
    return rows


def load_scored(patterns, concept):
    rows = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    a = r.get("activations")
                    if a and concept in a:
                        r["_v"] = a[concept]
                        rows.append(r)
    return rows


def quantile(vals, q):
    s = sorted(vals)
    i = q * (len(s) - 1)
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (i - lo) * (s[hi] - s[lo])


def hit_rates(rows, band_q):
    """P(top band | event) per event type."""
    vals = [r["_v"] for r in rows]
    cut = quantile(vals, band_q)
    tot = defaultdict(int)
    hit = defaultdict(int)
    for r in rows:
        e = r.get("event_type", "?")
        tot[e] += 1
        if r["_v"] >= cut:
            hit[e] += 1
    return {e: hit[e] / n for e, n in tot.items() if n >= 20}, dict(tot)


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    return pearson(rank(xs), rank(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", nargs="+", required=True,
                    help="RAW learning logs containing `target`")
    ap.add_argument("--learning", nargs="+", required=True,
                    help="scored affect files, learning condition")
    ap.add_argument("--frozen", nargs="+", required=True,
                    help="scored affect files, frozen condition")
    ap.add_argument("--concept", default="protesting")
    ap.add_argument("--band", type=float, default=0.85)
    args = ap.parse_args()

    logs = load_logs(args.log)
    if not logs:
        raise SystemExit(
            "No rows with a `target` field. This needs the RAW learning "
            "log; --score-all dropped `target` in older versions, so "
            "scored affect files will not work. Try learn_on_backup.jsonl "
            "or rerun learning_loop.py keeping the log.")

    print("=" * 78)
    print("DOES THE LOSS DECOMPOSITION EXPLAIN THE EVENT-TYPE PATTERN?")
    print("=" * 78)
    print(f"\n{len(logs)} gated turns with a training target")

    # ---- divergence per event type ----
    div = defaultdict(list)
    lenr = defaultdict(list)
    nll = defaultdict(list)
    for r in logs:
        e = r.get("event_type", "?")
        div[e].append(jaccard_distance(r["child_reaction"], r["target"]))
        lr = len(r["target"].split()) / max(1, len(r["child_reaction"].split()))
        lenr[e].append(lr)
        if r.get("nll") is not None:
            nll[e].append(r["nll"])

    # ---- desensitisation per event type ----
    A = load_scored(args.learning, args.concept)
    B = load_scored(args.frozen, args.concept)
    ha, na_ = hit_rates(A, args.band)
    hb, nb_ = hit_rates(B, args.band)

    events = sorted(set(div) & set(ha) & set(hb),
                    key=lambda e: -(hb.get(e, 0) - ha.get(e, 0)))
    if len(events) < 4:
        raise SystemExit(f"only {len(events)} event types with enough data")

    print(f"\n{'event':<18} {'n gated':>8} {'divergence':>11} "
          f"{'len ratio':>10} {'hit L':>8} {'hit F':>8} {'desens':>9}")
    print("-" * 78)
    D, S, L = [], [], []
    for e in events:
        d = mean(div[e])
        s = hb[e] - ha[e]
        D.append(d)
        S.append(s)
        L.append(mean(lenr[e]))
        print(f"{e:<18} {len(div[e]):>8} {d:>11.3f} {mean(lenr[e]):>10.2f} "
              f"{ha[e]:>7.1%} {hb[e]:>7.1%} {s:>+9.1%}")

    print(f"\n{'=' * 78}")
    print("THE PREDICTION")
    print(f"{'=' * 78}")
    rp = pearson(D, S)
    rs = spearman(D, S)
    rl = pearson(L, S)
    print(f"""
  divergence vs desensitisation:  r = {rp:+.3f}   rho = {rs:+.3f}
  length ratio vs desensitisation: r = {rl:+.3f}
  ({len(events)} event types)""")

    if nll:
        shared = [e for e in events if nll.get(e)]
        if len(shared) >= 4:
            N = [mean(nll[e]) for e in shared]
            SS = [hb[e] - ha[e] for e in shared]
            print(f"  logged NLL vs desensitisation:   "
                  f"r = {pearson(N, SS):+.3f}  ({len(shared)} types)")

    print(f"\n{'=' * 78}")
    print("READING THIS")
    print(f"{'=' * 78}")

    if rs > 0.5 and rp > 0.4:
        print(f"""
Event types whose training target diverged most from what the child
already said are the ones that desensitised most. That is what the
objective predicts, so Result 3 is a consequence of the loss rather
than a bare observation: conflict desensitised because conflict is
where the revision departed furthest from the original protest.

The mechanism is textbook -- larger gradients produce larger updates,
which is the premise of gradient-based data selection. What it buys
here is an explanation for WHICH behaviours change and by how much,
derived from the objective rather than described after the fact.""")
    elif abs(rs) < 0.3:
        print(f"""
Divergence does not predict desensitisation across event types
(rho = {rs:+.3f}). The loss decomposition does not explain the
pattern, so something other than gradient magnitude is determining
which event types change -- the KL constraint acting unevenly, the
diversity gate rejecting some event types more often, or an
interaction the per-type average hides.""")
    else:
        print(f"""
The relationship is in the predicted direction but weak
(rho = {rs:+.3f}). With {len(events)} event types there is little
power; suggestive at best.""")

    if abs(rl) > abs(rp):
        print(f"""
Note that LENGTH RATIO predicts desensitisation better than divergence
does (r = {rl:+.3f} against {rp:+.3f}). If revisions are simply longer
for some event types, that alone may drive the gradient, and the
explanation is about verbosity rather than about content departing
from the original.""")

    print("""
CAVEATS. The actual per-example gradient was not logged; Jaccard
distance and length ratio are proxies for it and could both be poor
ones. Divergence is averaged within event type, which hides variance.
Only gated turns have targets, so event types that gate rarely are
estimated from few examples -- check the n column. And with about ten
event types, a correlation needs to be large to mean much.""")


if __name__ == "__main__":
    main()