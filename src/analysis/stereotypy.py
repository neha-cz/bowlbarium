"""
Does stereotypy predict desensitisation?

THE HYPOTHESIS, FROM THE LITERATURE
-----------------------------------
Four candidate explanations for the conflict effect have failed, all
in the same way: they were properties of the TRAINING TARGETS, and
with ten event types and conflict extreme on every candidate, the
cross-type correlation could not separate cause from coincidence
(divergence r = -0.004; length r = +0.697 but +0.078 without conflict;
directional shift r = -0.765 but +0.004 without conflict; logged NLL
r = -0.659, wrong sign).

The fine-tuning literature suggests looking somewhere else entirely.
Ji et al. found that additional fine-tuning rapidly erases earlier
FINE-TUNED behaviours while PRETRAINED behaviours are far more robust,
and a geometric account holds that early training creates dominant
behavioural manifolds while later fine-tuning is a shallow
displacement carrying a reversion component back toward them.

The child's protest came from a cold-start LoRA on 65 hand-written
examples -- a shallow, recent displacement. And the frozen child's
conflict responses were dominated by bare "No!" repeated across turns,
which is what a narrow, templated fine-tuned behaviour looks like.

So the prediction is that desensitisation tracks how STEREOTYPED the
frozen child's responses are for an event type -- a property of the
pre-existing behaviour, not of the training targets. That is what
makes it a different hypothesis rather than a fifth version of the
same one.

WHAT IS MEASURED, PER EVENT TYPE (frozen condition only)
--------------------------------------------------------
    repetition     share of responses that are exact duplicates
    entropy        Shannon entropy over response types, in bits
    top share      share taken by the single most common response
    first token    entropy over first words -- relevant because RLHF
                   mainly alters the first few output tokens, and
                   "No!" is a first-token behaviour
    length         mean words, since terse responses are more
                   templatable

then correlated against desensitisation, with leave-one-out on every
correlation because that check has overturned three previous results.

Usage:
    python src/analysis/stereotypy.py --learning "runs/learning/*_on.jsonl" \\
                         --frozen "runs/learning/*_off.jsonl"
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
import math
from collections import Counter, defaultdict


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


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


def leave_one_out(names, xs, ys):
    out = []
    for i in range(len(names)):
        out.append((names[i],
                    pearson([v for j, v in enumerate(xs) if j != i],
                            [v for j, v in enumerate(ys) if j != i])))
    return out


def entropy(items):
    if not items:
        return 0.0
    c = Counter(items)
    n = len(items)
    return -sum((k / n) * math.log2(k / n) for k in c.values())


def load(patterns, concept):
    rows = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    a = r.get("activations")
                    if a and concept in a and r.get("child_reaction"):
                        r["_v"] = a[concept]
                        rows.append(r)
    return rows


def quantile(vals, q):
    s = sorted(vals)
    i = q * (len(s) - 1)
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (i - lo) * (s[hi] - s[lo])


def hit_rates(rows, band_q, min_n=20):
    cut = quantile([r["_v"] for r in rows], band_q)
    tot, hit = defaultdict(int), defaultdict(int)
    for r in rows:
        e = r.get("event_type", "?")
        tot[e] += 1
        if r["_v"] >= cut:
            hit[e] += 1
    return {e: hit[e] / n for e, n in tot.items() if n >= min_n}


def stereotypy(rows):
    """Per event type, how templated are the responses?"""
    by_event = defaultdict(list)
    for r in rows:
        by_event[r.get("event_type", "?")].append(r["child_reaction"])
    out = {}
    for e, texts in by_event.items():
        if len(texts) < 20:
            continue
        c = Counter(texts)
        firsts = [t.split()[0].lower().strip('.,!?') if t.split() else ""
                  for t in texts]
        out[e] = {
            "n": len(texts),
            "repetition": 1 - len(c) / len(texts),
            "entropy": entropy(texts),
            "top_share": c.most_common(1)[0][1] / len(texts),
            "first_entropy": entropy(firsts),
            "length": mean([len(t.split()) for t in texts]),
            "top_text": c.most_common(1)[0][0],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learning", nargs="+", required=True)
    ap.add_argument("--frozen", nargs="+", required=True)
    ap.add_argument("--concept", default="protesting")
    ap.add_argument("--band", type=float, default=0.85)
    args = ap.parse_args()

    A = load(args.learning, args.concept)
    B = load(args.frozen, args.concept)
    if not A or not B:
        raise SystemExit("no rows loaded")

    print("=" * 78)
    print("DOES STEREOTYPY PREDICT DESENSITISATION?")
    print("=" * 78)
    print(f"""
Measured on the FROZEN condition only -- a property of the child's
pre-existing behaviour, not of the training targets. Every previous
candidate was a target property, and all four failed the same way.""")

    ha, hb = hit_rates(A, args.band), hit_rates(B, args.band)
    st = stereotypy(B)

    events = sorted(set(st) & set(ha) & set(hb),
                    key=lambda e: -(hb[e] - ha[e]))
    if len(events) < 5:
        raise SystemExit(f"only {len(events)} event types with enough data")

    print(f"\n{'event':<17} {'n':>5} {'repeat':>8} {'entropy':>8} "
          f"{'top':>7} {'1st ent':>8} {'words':>7} {'desens':>8}")
    print("-" * 78)
    rows = []
    for e in events:
        s = st[e]
        d = hb[e] - ha[e]
        rows.append((e, s, d))
        print(f"{e:<17} {s['n']:>5} {s['repetition']:>8.2f} "
              f"{s['entropy']:>8.2f} {s['top_share']:>7.2f} "
              f"{s['first_entropy']:>8.2f} {s['length']:>7.1f} "
              f"{d:>+8.1%}")

    print(f"\n  most common frozen response per event type:")
    for e, s, d in rows[:4]:
        print(f"    {e:<16} {s['top_share']:>5.0%}  \"{s['top_text'][:44]}\"")

    # ---- correlations ----
    D = [d for _, _, d in rows]
    names = [e for e, _, _ in rows]
    print(f"\n{'=' * 78}")
    print("CORRELATION WITH DESENSITISATION")
    print(f"{'=' * 78}")
    print(f"\n{'measure':<20} {'r':>9} {'rho':>9} {'LOO range':>22}")
    print("-" * 78)

    measures = {
        "repetition": [s["repetition"] for _, s, _ in rows],
        "entropy": [s["entropy"] for _, s, _ in rows],
        "top share": [s["top_share"] for _, s, _ in rows],
        "first-token entropy": [s["first_entropy"] for _, s, _ in rows],
        "mean length": [s["length"] for _, s, _ in rows],
    }
    results = {}
    for label, vals in measures.items():
        r = pearson(vals, D)
        rho = spearman(vals, D)
        loo = [v for _, v in leave_one_out(names, vals, D)]
        results[label] = (r, rho, min(loo), max(loo))
        print(f"{label:<20} {r:>+9.3f} {rho:>+9.3f} "
              f"{f'{min(loo):+.2f} to {max(loo):+.2f}':>22}")

    print("""
  Predicted signs: repetition and top share POSITIVE (more templated
  -> more desensitisation); entropy, first-token entropy and length
  NEGATIVE. The LOO range is what matters -- three previous
  correlations here were one point in disguise.""")

    print(f"\n{'=' * 78}")
    print("READING THIS")
    print(f"{'=' * 78}")

    # a measure "holds" if it is strong AND survives leave-one-out
    holds = [(l, v) for l, v in results.items()
             if abs(v[0]) > 0.5 and min(abs(v[2]), abs(v[3])) > 0.3]
    correct_sign = [
        l for l, v in results.items()
        if (l in ("repetition", "top share") and v[0] > 0.5)
        or (l in ("entropy", "first-token entropy", "mean length")
            and v[0] < -0.5)]

    if holds and set(l for l, _ in holds) & set(correct_sign):
        good = [l for l, _ in holds if l in correct_sign]
        print(f"""
Stereotypy predicts desensitisation, in the predicted direction, on
{', '.join(good)}, and the relationship survives leave-one-out.

That supports the fine-tuning-fragility account: the behaviours that
erode are the ones that were most templated to begin with -- shallow,
narrow displacements from the pretrained manifold, which subsequent
fine-tuning reverts most easily. Conflict desensitises most because
conflict-protest was the child's most stereotyped behaviour, not
because of any property of the conflict training targets.

Unlike the four earlier candidates, this is a property of the
PRE-EXISTING behaviour, which is why it can be measured without
depending on conflict being extreme on the target side.""")
    elif holds:
        print(f"""
Some measures correlate and survive leave-one-out, but not in the
predicted direction: {', '.join(l for l, _ in holds)}. Read the signs
in the table before interpreting -- a relationship pointing the wrong
way is not support for the hypothesis.""")
    else:
        strongest = max(results.items(), key=lambda kv: abs(kv[1][0]))
        print(f"""
No stereotypy measure both correlates strongly and survives
leave-one-out. The strongest is {strongest[0]} at r =
{strongest[1][0]:+.3f}, with leave-one-out spanning
{strongest[1][2]:+.2f} to {strongest[1][3]:+.2f}.

So the fine-tuning-fragility account does not explain the pattern
either. That is five candidate mechanisms eliminated, and the honest
statement is that conflict desensitisation is robust and unexplained
-- which is worth reporting as such, since the elimination sequence
tells the next person where not to look.""")

    print("""
CAVEATS. Stereotypy is measured on the frozen condition, which is the
right baseline, but the frozen child was itself produced by the same
cold start -- so "stereotyped" and "fine-tuning-dependent" are not
cleanly separable here. Ten event types gives little power. And the
repetition measures are sensitive to how many turns each event type
has, which the n column shows.""")


if __name__ == "__main__":
    main()