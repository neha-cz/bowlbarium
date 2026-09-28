"""
Decoupling: does learning flatten the event-to-protest mapping?

THE REFRAME
-----------
Eight candidate mechanisms have been tested for why CONFLICT
specifically desensitises, and all eight failed. Conflict is
unremarkable on divergence, length, target direction, training loss,
stereotypy, mother behaviour, and the positive-gap grouping -- extreme
only on the outcome.

The engagement manipulation then went the wrong way: instructing the
mother to keep conflict exchanges open DOUBLED the desensitisation
(learning hit rate 9.1% against a control's 36.4%, conflict lift
falling to 0.61 -- below chance).

That suggests the question was wrong. Look at the learning arms across
runs: conflict lift 2.42 in the control, 0.61 when engaged, while
mundane sits at 1.67 and separation at 2.50. The learning child's
high-protest turns are spread across ordinary events. Conflict has the
strongest event-to-protest coupling in the FROZEN child -- lift 4.24,
four times chance -- and learning flattens that coupling.

Under this reading conflict is not special as a cause. It is special
as the place where the most structure existed, so it is where
flattening shows most. Event types with weak coupling have less to
lose, which is why they appear unchanged or rise as mass redistributes.

THE PREDICTION
--------------
    desensitisation(e)  should track  frozen lift(e)

across all ten event types, not just conflict.

THE PROBLEM THAT MAKES THIS NON-TRIVIAL
---------------------------------------
Frozen lift and desensitisation are both computed from the frozen hit
rate, so they are mathematically linked: an event type with a high
frozen hit rate can drop further simply because it has further to
drop. A raw correlation between them is guaranteed to be positive and
proves nothing.

Two nulls address this:

  PROPORTIONAL   if learning multiplies every hit rate by a constant
                 factor, the drop is proportional to the frozen rate
                 by construction. Compare observed drops against
                 hit_frozen * (1 - k), with k fitted.
  SHUFFLE        redistribute the learning arm's high-protest turns
                 across event types at random, preserving the totals,
                 and recompute. This gives the distribution of
                 apparent decoupling under no event-specific effect.

Flattening is only interesting if it exceeds both.

Usage:
    python decoupling.py --learning "runs_learning/*_on.jsonl" \\
                         --frozen "runs_learning/*_off.jsonl"
"""

import argparse
import glob
import json
import random
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


def quantile(v, q):
    s = sorted(v)
    i = q * (len(s) - 1)
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (i - lo) * (s[hi] - s[lo])


def load(patterns, concept):
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


def band_flags(rows, q):
    """Per row: is it in the top band? Plus per-event counts."""
    cut = quantile([r["_v"] for r in rows], q)
    flags = [r["_v"] >= cut for r in rows]
    return flags, cut


def rates(rows, flags):
    tot, hit = defaultdict(int), defaultdict(int)
    for r, f in zip(rows, flags):
        e = r.get("event_type", "?")
        tot[e] += 1
        hit[e] += bool(f)
    return tot, hit


def coupling_spread(hits, tots, events):
    """How unevenly the band is distributed across event types.

    Standard deviation of lift across event types: high means the band
    is concentrated in particular events, near zero means it is spread
    in proportion to how often events occur -- i.e. decoupled.
    """
    n_all = sum(tots[e] for e in events)
    n_band = sum(hits[e] for e in events)
    if n_band == 0:
        return 0.0
    lifts = []
    for e in events:
        share_band = hits[e] / n_band
        share_all = tots[e] / n_all
        lifts.append(share_band / share_all if share_all else 0.0)
    m = mean(lifts)
    return (sum((l - m) ** 2 for l in lifts) / (len(lifts) - 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learning", nargs="+", required=True)
    ap.add_argument("--frozen", nargs="+", required=True)
    ap.add_argument("--concept", default="protesting")
    ap.add_argument("--band", type=float, default=0.85)
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A = load(args.learning, args.concept)
    B = load(args.frozen, args.concept)
    if not A or not B:
        raise SystemExit("no rows loaded")

    fa, _ = band_flags(A, args.band)
    fb, _ = band_flags(B, args.band)
    tot_a, hit_a = rates(A, fa)
    tot_b, hit_b = rates(B, fb)

    events = sorted([e for e in set(tot_a) & set(tot_b)
                     if tot_a[e] >= 20 and tot_b[e] >= 20])
    if len(events) < 5:
        raise SystemExit(f"only {len(events)} usable event types")

    print("=" * 78)
    print(f"DECOUPLING  --  does learning flatten the event-protest map?")
    print("=" * 78)

    n_all_a = sum(tot_a[e] for e in events)
    n_all_b = sum(tot_b[e] for e in events)
    n_band_a = sum(hit_a[e] for e in events)
    n_band_b = sum(hit_b[e] for e in events)

    print(f"\n{'event':<17} {'hit F':>8} {'hit L':>8} {'drop':>8} "
          f"{'lift F':>8} {'lift L':>8}")
    print("-" * 78)
    rows = []
    for e in sorted(events,
                    key=lambda x: -(hit_b[x]/tot_b[x] - hit_a[x]/tot_a[x])):
        hf = hit_b[e] / tot_b[e]
        hl = hit_a[e] / tot_a[e]
        lf = (hit_b[e]/n_band_b) / (tot_b[e]/n_all_b) if n_band_b else 0
        ll = (hit_a[e]/n_band_a) / (tot_a[e]/n_all_a) if n_band_a else 0
        rows.append({"e": e, "hf": hf, "hl": hl, "drop": hf - hl,
                     "lf": lf, "ll": ll})
        print(f"{e:<17} {hf:>7.1%} {hl:>7.1%} {hf-hl:>+8.1%} "
              f"{lf:>8.2f} {ll:>8.2f}")

    # ---- 1. spread of lift ----
    sf = coupling_spread(hit_b, tot_b, events)
    sl = coupling_spread(hit_a, tot_a, events)
    print(f"\n{'=' * 78}")
    print("1. IS THE BAND MORE EVENLY SPREAD AFTER LEARNING?")
    print(f"{'=' * 78}")
    print(f"""
  spread of lift across event types (sd):
    frozen    {sf:.3f}
    learning  {sl:.3f}
    change    {sl - sf:+.3f}

  Lower spread means the high-protest band is distributed more in
  proportion to how often events occur -- the definition of
  decoupling. Higher means it concentrated further.""")

    # ---- 2. proportional null ----
    print(f"\n{'=' * 78}")
    print("2. IS IT JUST PROPORTIONAL SHRINKAGE?")
    print(f"{'=' * 78}")
    k = (n_band_a / n_all_a) / (n_band_b / n_all_b) if n_band_b else 1.0
    print(f"""
  If learning multiplied every event's hit rate by the same factor,
  each drop would be hit_frozen * (1 - k), with k = {k:.3f}.""")
    print(f"\n{'event':<17} {'observed':>10} {'proportional':>14} "
          f"{'residual':>10}")
    print("-" * 78)
    resid = []
    for r in rows:
        pred = r["hf"] * (1 - k)
        resid.append(r["drop"] - pred)
        print(f"{r['e']:<17} {r['drop']:>+10.1%} {pred:>+14.1%} "
              f"{r['drop']-pred:>+10.1%}")
    print(f"""
  A residual near zero everywhere means learning shrank all event
  types by the same factor and nothing is event-specific. Large
  residuals mean some events lost more than proportional.""")

    # ---- 3. shuffle null ----
    print(f"\n{'=' * 78}")
    print("3. SHUFFLE NULL")
    print(f"{'=' * 78}")
    rng = random.Random(args.seed)
    obs_spread = sl
    null = []
    labels = [r.get("event_type", "?") for r in A]
    for _ in range(args.permutations):
        shuffled = list(fa)
        rng.shuffle(shuffled)
        t, h = defaultdict(int), defaultdict(int)
        for e, f in zip(labels, shuffled):
            t[e] += 1
            h[e] += bool(f)
        null.append(coupling_spread(h, t, events))
    null.sort()
    p_low = (sum(1 for v in null if v <= obs_spread) + 1) / (len(null) + 1)
    mu = mean(null)
    print(f"""
  Randomly redistributing the learning arm's band across turns, keeping
  totals fixed, gives the spread expected with NO event-specific
  structure.

    observed learning spread   {obs_spread:.3f}
    shuffled null mean         {mu:.3f}
    p(spread <= observed)      {p_low:.3f}

  If the observed spread is at or below the null, the learning arm has
  no more event structure than chance -- fully decoupled. If it is
  above, structure remains.""")

    # ---- reading ----
    print(f"\n{'=' * 78}")
    print("READING THIS")
    print(f"{'=' * 78}")

    r_lift = pearson([r["lf"] for r in rows], [r["drop"] for r in rows])
    r_resid = pearson([r["lf"] for r in rows], resid)
    print(f"""
  correlation(frozen lift, drop)              {r_lift:+.3f}
  correlation(frozen lift, residual)          {r_resid:+.3f}

  The first is guaranteed positive by construction -- events with high
  frozen rates have further to fall. The SECOND is the informative
  one: it asks whether high-coupling events lost more than
  proportional shrinkage predicts.""")

    if sl < sf and p_low > 0.2 and abs(r_resid) < 0.4:
        print("""
Learning flattened the event-protest mapping, the learning arm's
remaining structure is not distinguishable from chance, and the
per-event losses are close to proportional.

That supports the decoupling reading: there is one effect -- uniform
downward pressure from the training targets -- and conflict looked
special only because it had the most coupling to lose. The eight
failed mechanism hunts were asking why one bin moved most, when every
bin moved and that one started highest.""")
    elif sl < sf and abs(r_resid) >= 0.4:
        print(f"""
Flattening occurred, but the residuals correlate with frozen lift
({r_resid:+.3f}) -- high-coupling events lost MORE than proportional
shrinkage predicts. So it is not purely uniform pressure; something
does target the strongly-coupled events specifically, and the
conflict question survives in a generalised form.""")
    elif sl >= sf:
        print(f"""
The learning arm's lift spread is NOT lower than the frozen arm's
({sl:.3f} against {sf:.3f}). Learning did not flatten the
event-protest mapping, so the decoupling reading is wrong and
conflict-specific suppression stands as the description.""")
    else:
        print("""
Mixed: flattening occurred but the learning arm retains structure
beyond the shuffle null. Partial decoupling, with some event
specificity remaining.""")

    print("""
CAVEATS. The band is defined within condition, so both arms have the
same number of band turns by construction and only the DISTRIBUTION
can differ -- which is what is being measured, but it means absolute
protest levels play no part here. Ten event types with 20-180 turns
each; the smaller ones are noisy. The shuffle null destroys all event
structure, which is a strong null -- a system could be partly
decoupled and still sit above it.""")


if __name__ == "__main__":
    main()