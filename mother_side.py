"""
What is different about conflict, on the MOTHER's side?

FIVE ELIMINATED CANDIDATES
--------------------------
Conflict desensitises four times more than any other event type
(+38.6% against a range of +2.0% to -15.9%), replicated 6/6 seeds,
with base rates identical by construction and the effect the same at
0.9 and 0.15 caregiver reliability. Five explanations have been tested
and none accounts for it:

    lexical divergence of targets    r = -0.004, flat across types
    length ratio of targets          +0.697, but +0.078 without conflict
    directional shift of targets     -0.765, but +0.004 without conflict
    logged training loss             -0.659, wrong sign
    stereotypy of frozen behaviour   all |r| < 0.27, and the MOST
                                     templated type (mundane, top share
                                     0.37) moved the OPPOSITE way

On every measured property, conflict is unremarkable -- mid-range on
entropy, first-token entropy, length, repetition -- and extreme only
on the outcome.

WHAT HAS NEVER BEEN CHARACTERISED
---------------------------------
The mother. Every analysis so far measured the child: its reactions,
its training targets, its activations, its weights. The one channel
connecting an event type to what the child learns, which nobody has
looked at, is what the mother SAYS during those events.

Conflict is the only event type where the mother is in opposition to
the child by construction -- not merely unavailable, but taking a
contrary position. If her conflict turns differ systematically from
her other turns, that is a property of the interaction rather than of
the child, and it would sit upstream of everything already tested.

WHAT IS MEASURED, PER EVENT TYPE
--------------------------------
    mother length, repetition, entropy   is she more templated here?
    unresponsive rate                    the classifier from the
                                         manipulation check
    mother emotion profile               her turns scored with the
                                         same vectors as the child's
    child-mother divergence              how far apart the two turns
                                         sit on the protest axis --
                                         opposition, measured
    reaction-to-mother alignment         does the child's reaction
                                         track her position?

Each is correlated against desensitisation with leave-one-out, since
three earlier correlations here were a single point in disguise.

Usage:
    python mother_side.py --log "runs_reliability/R900_*_learn.jsonl" \\
                          --learning "runs_learning/*_on.jsonl" \\
                          --frozen "runs_learning/*_off.jsonl"
"""

import argparse
import glob
import json
import math
import re
from collections import Counter, defaultdict

UNRESPONSIVE = [
    r"\bgive me (a|one|two|five|\d+) (minute|second|sec)",
    r"\bin a (minute|second|sec|bit)\b", r"\bhang on\b", r"\bhold on\b",
    r"\bnot (right )?now\b", r"\blater\b", r"\bwhen i('m| am) done\b",
    r"\bafter (i|we) finish", r"\bi'?m (in the middle of|busy|on the phone)",
    r"\bwait (a|one) (minute|second|sec)", r"\bstop (it|that)?\b",
    r"\byou'?re fine\b", r"\bdon'?t (be|worry|cry|start)\b",
    r"\benough\b", r"\bit'?s (just|only)\b", r"\bnever mind\b",
    r"\bthat'?s (nice|enough)\b", r"\bbecause i said so\b",
    r"^no[,.! ]", r"\bnot today\b",
]
UNRESP_RE = [re.compile(p, re.I) for p in UNRESPONSIVE]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def entropy(items):
    if not items:
        return 0.0
    c = Counter(items)
    n = len(items)
    return -sum((k / n) * math.log2(k / n) for k in c.values())


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
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for p, i in enumerate(o):
            r[i] = p
        return r
    return pearson(rank(xs), rank(ys))


def loo(names, xs, ys):
    return [(names[i],
             pearson([v for j, v in enumerate(xs) if j != i],
                     [v for j, v in enumerate(ys) if j != i]))
            for i in range(len(names))]


def quantile(v, q):
    s = sorted(v)
    i = q * (len(s) - 1)
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (i - lo) * (s[hi] - s[lo])


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


def hit_rates(rows, q, min_n=20):
    cut = quantile([r["_v"] for r in rows], q)
    tot, hit = defaultdict(int), defaultdict(int)
    for r in rows:
        e = r.get("event_type", "?")
        tot[e] += 1
        if r["_v"] >= cut:
            hit[e] += 1
    return {e: hit[e] / n for e, n in tot.items() if n >= min_n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", nargs="+", required=True,
                    help="logs containing mother_response")
    ap.add_argument("--learning", nargs="+", required=True)
    ap.add_argument("--frozen", nargs="+", required=True)
    ap.add_argument("--concept", default="protesting")
    ap.add_argument("--band", type=float, default=0.85)
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load as load_model
    from extract_emotions import (load_vectors, capture_residual,
                                  BASE_MODEL, CHILD_ADAPTER)

    # mother turns
    turns = []
    for pattern in args.log:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("mother_response") and r.get("child_reaction"):
                        turns.append(r)
    if not turns:
        raise SystemExit("no rows with mother_response")

    print("=" * 78)
    print("WHAT IS DIFFERENT ABOUT CONFLICT, ON THE MOTHER'S SIDE?")
    print("=" * 78)
    print(f"\n{len(turns)} turns with the mother's text")

    vectors, layer, baseline = load_vectors()
    model, tok = load_model(BASE_MODEL, adapter_path=CHILD_ADAPTER)
    vec = mx.array(vectors[args.concept])
    base = mx.array(baseline) if baseline else None
    cache = {}

    def score(t):
        if t not in cache:
            p = capture_residual(model, tok, t, layer)[-1]
            if base is not None:
                p = p - base
            cache[t] = float(mx.sum(p * vec))
        return cache[t]

    print(f"scoring mother and child turns (layer {layer}) ...")

    by_event = defaultdict(lambda: {"m": [], "c": [], "gap": [],
                                    "unresp": [], "mlen": []})
    for r in turns:
        e = r.get("event_type", "?")
        m = r["mother_response"]
        c = r["child_reaction"]
        sm, sc = score(m), score(c)
        d = by_event[e]
        d["m"].append(m)
        d["c"].append(c)
        d["gap"].append(sc - sm)
        d["unresp"].append(1 if any(p.search(m) for p in UNRESP_RE) else 0)
        d["mlen"].append(len(m.split()))

    A = load_scored(args.learning, args.concept)
    B = load_scored(args.frozen, args.concept)
    ha, hb = hit_rates(A, args.band), hit_rates(B, args.band)

    events = sorted(set(by_event) & set(ha) & set(hb),
                    key=lambda e: -(hb[e] - ha[e]))
    events = [e for e in events if len(by_event[e]["m"]) >= 15]
    if len(events) < 5:
        raise SystemExit(f"only {len(events)} usable event types")

    print(f"\n{'event':<17} {'n':>4} {'m.words':>8} {'m.rep':>7} "
          f"{'unresp':>8} {'m.protest':>10} {'gap c-m':>9} {'desens':>8}")
    print("-" * 78)
    rows = []
    for e in events:
        d = by_event[e]
        m_protest = mean([score(t) for t in d["m"]])
        rep = 1 - len(set(d["m"])) / len(d["m"])
        rows.append({
            "event": e, "n": len(d["m"]), "mlen": mean(d["mlen"]),
            "rep": rep, "unresp": mean(d["unresp"]),
            "m_protest": m_protest, "gap": mean(d["gap"]),
            "desens": hb[e] - ha[e],
        })
        print(f"{e:<17} {len(d['m']):>4} {mean(d['mlen']):>8.1f} "
              f"{rep:>7.2f} {mean(d['unresp']):>7.0%} "
              f"{m_protest:>+10.3f} {mean(d['gap']):>+9.3f} "
              f"{hb[e]-ha[e]:>+8.1%}")

    print("""
  m.protest is the MOTHER's turn scored on the child's protest vector
  -- a rough proxy for how oppositional her line is. gap c-m is the
  child's score minus hers: how far apart the two sit.""")

    # examples
    print(f"\n  mother's turns during the two extreme event types:")
    for e in (rows[0]["event"], rows[-1]["event"]):
        print(f"\n    {e}:")
        for t in by_event[e]["m"][:3]:
            print(f"      {t[:64]}")

    # correlations
    D = [r["desens"] for r in rows]
    names = [r["event"] for r in rows]
    print(f"\n{'=' * 78}")
    print("CORRELATION WITH DESENSITISATION")
    print(f"{'=' * 78}")
    print(f"\n{'measure':<22} {'r':>9} {'rho':>9} {'LOO range':>22}")
    print("-" * 78)
    measures = {
        "mother length": [r["mlen"] for r in rows],
        "mother repetition": [r["rep"] for r in rows],
        "unresponsive rate": [r["unresp"] for r in rows],
        "mother protest score": [r["m_protest"] for r in rows],
        "child-mother gap": [r["gap"] for r in rows],
    }
    results = {}
    for label, vals in measures.items():
        r = pearson(vals, D)
        rho = spearman(vals, D)
        l = [v for _, v in loo(names, vals, D)]
        results[label] = (r, rho, min(l), max(l))
        print(f"{label:<22} {r:>+9.3f} {rho:>+9.3f} "
              f"{f'{min(l):+.2f} to {max(l):+.2f}':>22}")

    print(f"\n{'=' * 78}")
    print("READING THIS")
    print(f"{'=' * 78}")

    survivors = [(l, v) for l, v in results.items()
                 if abs(v[0]) > 0.5 and min(abs(v[2]), abs(v[3])) > 0.3]

    if survivors:
        best = max(survivors, key=lambda kv: abs(kv[1][0]))
        print(f"""
{best[0]} correlates with desensitisation at r = {best[1][0]:+.3f} and
survives leave-one-out ({best[1][2]:+.2f} to {best[1][3]:+.2f}).

This is the first candidate to survive that check, and it sits on the
MOTHER's side -- a property of the interaction rather than of the
child, its targets, or its prior behaviour. Worth pursuing: it would
mean the event type matters through what the caregiver does in it.

Note the tension with the caregiver-independence result, though.
Desensitisation was identical at 0.9 and 0.15 reliability, so whatever
this is, it cannot be availability. It would have to be something
about her turns that does not change with how often she is free.""")
    else:
        strongest = max(results.items(), key=lambda kv: abs(kv[1][0]))
        print(f"""
Nothing on the mother's side survives either. The strongest is
{strongest[0]} at r = {strongest[1][0]:+.3f}, leave-one-out
{strongest[1][2]:+.2f} to {strongest[1][3]:+.2f}.

Six candidate mechanisms now, all eliminated with the same check. The
honest conclusion is that conflict desensitisation is robust and its
cause is not in any measured property of the targets, the environment,
the child's prior behaviour, or the caregiver's turns.

At this point the elimination sequence is the contribution. It tells
anyone reproducing this where not to look, which is more than most
papers provide.""")

    print("""
CAVEATS. The mother's turns are scored with vectors built from the
CHILD adapter, so "mother protest" means how the child's
representation reads her text -- a proxy, not her state. The
unresponsive classifier has known misclassification in both
directions. Ten event types, and the mother-side measures come from
the reliability-sweep logs while desensitisation comes from the
learning runs, so they are different runs under the same settings.""")


if __name__ == "__main__":
    main()