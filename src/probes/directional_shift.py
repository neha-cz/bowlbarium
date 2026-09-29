"""
Directional shift: which way did the training target move?

WHY THE PREVIOUS TEST FAILED
----------------------------
loss_decomposition.py measured how DIFFERENT each event type's
training target was from the child's original reaction, and found
nothing:

    Jaccard divergence   0.899-0.938 across all ten types -- a range
                         of 0.04, so no variance to correlate with
    length ratio         r = +0.697, but leave-one-out showed the
                         whole correlation is conflict alone (drops to
                         +0.078 without it, stays 0.66-0.80 without any
                         other type)
    logged NLL           r = -0.659, the wrong sign

And conflict is 7.0 sd above what a length fit on the other nine
predicts (-2.8% predicted, +38.6% actual). So no measure of target
MAGNITUDE explains it.

WHAT WAS NEVER MEASURED
-----------------------
Direction. A revision can be lexically distant from the reaction and
just as protesting -- which is what uniform Jaccard around 0.92 looks
like. The quantity that matters for a protest effect is how far the
target moved ALONG THE PROTEST AXIS:

    delta(e) = protest(target_e) - protest(reaction_e)

If conflict revisions are specifically less protesting than the
reactions they replace, the gradient pushes protest down hardest on
conflict, and desensitisation follows. That is a directional claim the
earlier proxies could not express.

A SUPPORTING HINT
-----------------
Conflict was UNDER-gated: 35 of 432 gated turns (8.1%) against a
10.6% base rate. Fewer updates, largest change. So it is not that
conflict received more gradient steps -- the steps must differ in
kind, which is what a directional measure would show.

Usage:
    python src/probes/directional_shift.py \\
        --log "runs/reliability/R900_*_learn.jsonl" \\
        --learning "runs/learning/*_on.jsonl" \\
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
from collections import defaultdict


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
        x2 = [v for j, v in enumerate(xs) if j != i]
        y2 = [v for j, v in enumerate(ys) if j != i]
        out.append((names[i], pearson(x2, y2)))
    return out


def quantile(vals, q):
    s = sorted(vals)
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


def hit_rates(rows, band_q, min_n=20):
    vals = [r["_v"] for r in rows]
    cut = quantile(vals, band_q)
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
                    help="raw learning logs with `target`")
    ap.add_argument("--learning", nargs="+", required=True)
    ap.add_argument("--frozen", nargs="+", required=True)
    ap.add_argument("--concept", default="protesting")
    ap.add_argument("--band", type=float, default=0.85)
    ap.add_argument("--layer", type=int, default=None)
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load as load_model
    from extract_emotions import (load_vectors, capture_residual, mean_pool,
                                  BASE_MODEL, CHILD_ADAPTER)

    # ---- gather targets ----
    pairs = []
    for pattern in args.log:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("target") and r.get("child_reaction"):
                        pairs.append((r.get("event_type", "?"),
                                      r["child_reaction"], r["target"]))
    if not pairs:
        raise SystemExit("no rows with a `target` field")

    print("=" * 78)
    print("DIRECTIONAL SHIFT  --  which way did the training target move?")
    print("=" * 78)
    print(f"\n{len(pairs)} gated turns with a target")

    vectors, layer, baseline = load_vectors()
    if args.layer is not None:
        layer = args.layer
    print(f"scoring with vectors from layer {layer} ...")

    model, tok = load_model(BASE_MODEL, adapter_path=CHILD_ADAPTER)
    vec = mx.array(vectors[args.concept])
    base = mx.array(baseline) if baseline else None

    def score(text):
        pooled = capture_residual(model, tok, text, layer)[-1]
        if base is not None:
            pooled = pooled - base
        return float(mx.sum(pooled * vec))

    by_event = defaultdict(list)
    seen = {}
    for i, (e, reaction, target) in enumerate(pairs):
        for t in (reaction, target):
            if t not in seen:
                seen[t] = score(t)
        by_event[e].append(seen[target] - seen[reaction])
        if (i + 1) % 50 == 0:
            print(f"\r  {i+1}/{len(pairs)}", end="", flush=True)
    print()

    # ---- desensitisation ----
    A = load_scored(args.learning, args.concept)
    B = load_scored(args.frozen, args.concept)
    ha, hb = hit_rates(A, args.band), hit_rates(B, args.band)

    events = sorted(set(by_event) & set(ha) & set(hb),
                    key=lambda e: -(hb[e] - ha[e]))
    print(f"\n{'event':<18} {'n':>5} {'delta protest':>15} "
          f"{'hit L':>8} {'hit F':>8} {'desens':>9}")
    print("-" * 78)
    D, S = [], []
    for e in events:
        d = mean(by_event[e])
        s = hb[e] - ha[e]
        D.append(d)
        S.append(s)
        print(f"{e:<18} {len(by_event[e]):>5} {d:>+15.3f} "
              f"{ha[e]:>7.1%} {hb[e]:>7.1%} {s:>+9.1%}")

    r = pearson(D, S)
    rho = spearman(D, S)
    print(f"""
{'=' * 78}
delta protest vs desensitisation:  r = {r:+.3f}   rho = {rho:+.3f}
  (negative r means: event types whose targets pushed protest DOWN
   most are the ones that desensitised most -- the predicted sign)""")

    print(f"\nleave-one-out on r, to check it is not one point:")
    for name, rr in leave_one_out(events, D, S):
        flag = "  <-- collapses without this" if abs(rr) < 0.3 <= abs(r) else ""
        print(f"  without {name:<18} r = {rr:+.3f}{flag}")

    print(f"\n{'=' * 78}")
    print("READING THIS")
    print(f"{'=' * 78}")

    loo = [rr for _, rr in leave_one_out(events, D, S)]
    robust = all(abs(v) > 0.3 for v in loo) if abs(r) > 0.3 else False

    if r < -0.5 and robust:
        print("""
Event types whose training targets moved furthest DOWN the protest
axis are the ones that desensitised most, and the relationship does
not depend on any single event type.

That explains the conflict result: conflict revisions are the ones
that most reduce protest relative to the original reaction, so the
gradient on conflict pushes protest down hardest. Magnitude measures
(Jaccard, length, NLL) missed it because a revision can be very
different from the reaction while being equally protesting -- only the
directional measure captures what the update is actually doing.""")
    elif r < -0.5:
        print("""
The relationship has the predicted sign but depends on a single event
type -- the same failure mode as the length correlation, which
collapsed from +0.697 to +0.078 without conflict. Not an explanation
yet.""")
    elif abs(r) < 0.3:
        print("""
Direction does not predict desensitisation either. Neither the
magnitude of the target's departure nor its direction on the protest
axis accounts for which event types change.

That is worth stating plainly rather than continuing to search:
conflict desensitises far more than any measured property of its
training targets predicts, and the cause is not in the targets. The
next place to look is the REFLECTIONS -- what the child says it
noticed -- rather than the revisions the reflections produced.""")
    else:
        print(f"""
The correlation is {r:+.3f}, in the opposite direction from the
prediction. Event types whose targets pushed protest UP desensitised
more, which no account of the gradient explains and probably
indicates the per-type averages are hiding something.""")

    print("""
CAVEATS. Targets are scored with vectors extracted from the child
adapter, but the targets were generated under a revision prompt that
asked for longer output -- and the protest vector retains some length
sensitivity, so a longer target may score differently for reasons
unrelated to protest. Per-type averages hide within-type variance.
Ten event types gives little power for a correlation.""")


if __name__ == "__main__":
    main()