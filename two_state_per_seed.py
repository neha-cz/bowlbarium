"""
Per-seed replication of the activation-energy result.

WHAT IS BEING TESTED
--------------------
two_state.py found, on pooled data:

    activated fraction   0.386 (learning) vs 0.662 (frozen)
    baseline mean       -0.534           vs -0.639
    activated mean      +0.746           vs +0.620
    dE / T              +0.466           vs -0.670   difference +1.136

with bootstrap intervals that do NOT overlap. The states are the same
and the occupancy differs, which is the barrier case: learning raises
the cost of entering the protest state by about 1.1 kT rather than
changing what that state is.

That is one condition pair. Four earlier findings in this project
looked this convincing pooled and did not survive per-seed
examination -- a 22-point gate-rate effect that averaged to zero, a
"withdrawn" effect consistent across three seeds that reversed on the
next five, a "distressed" effect that also appeared in the null
control, and a double-well prediction that came back unimodal at eight
seeds.

So: fit the mixture separately within each seed's learning/frozen
pair, and check whether the barrier shift holds SIGN and rough
MAGNITUDE in every one. Pooling can manufacture a mixture out of
between-seed heterogeneity -- if seeds differ in their mean, the
pooled distribution is a mixture of seeds rather than of states, and
that is a mundane explanation this test rules out.

WHAT WOULD COUNT
----------------
    sign consistent in every pair, mean/sd well above 1
        -> the barrier shift is real
    sign flips between pairs
        -> the pooled result was between-seed heterogeneity
    components move within pairs
        -> a state change, not a barrier change, and the Boltzmann
           reading does not apply per-seed even if it seemed to pooled

Usage:
    python two_state_per_seed.py \\
        --pair 600 "runs_learning/L600_on.jsonl" "runs_learning/L600_off.jsonl" \\
        --pair 601 "runs_learning/L601_on.jsonl" "runs_learning/L601_off.jsonl" \\
        --pair 602 "runs_learning/L602_on.jsonl" "runs_learning/L602_off.jsonl"
"""

import argparse
import glob
import json
import math

from two_state import (fit_mixture, activation_energy, loglik_single, bic,
                       load_values)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=3, action="append",
                    metavar=("SEED", "LEARNING", "FROZEN"), required=True)
    ap.add_argument("--concept", default="protesting")
    args = ap.parse_args()

    print("=" * 78)
    print(f"ACTIVATION ENERGY, PER SEED  --  {args.concept}")
    print("=" * 78)
    print("""
The mixture is fit WITHIN each pair. Pooling across seeds can
manufacture a two-component structure out of between-seed differences
in the mean, which would look like two states and be nothing of the
kind.""")

    rows = []
    for seed, lp, fp in args.pair:
        L = load_values(lp, args.concept)
        F = load_values(fp, args.concept)
        if len(L) < 80 or len(F) < 80:
            print(f"\nseed {seed}: too little data "
                  f"({len(L)}, {len(F)}) -- skipped")
            continue
        fl = fit_mixture(L)
        ff = fit_mixture(F)
        dl = activation_energy(fl)
        df = activation_energy(ff)
        rows.append({
            "seed": seed, "n_l": len(L), "n_f": len(F),
            "wl": fl["w_act"], "wf": ff["w_act"],
            "dE_l": dl, "dE_f": df, "d": dl - df,
            "mb_l": fl["m_base"], "mb_f": ff["m_base"],
            "ma_l": fl["m_act"], "ma_f": ff["m_act"],
        })

    if len(rows) < 2:
        raise SystemExit("need at least two usable pairs")

    print(f"\n{'seed':>6} {'act frac L':>11} {'act frac F':>11} "
          f"{'dE_L':>8} {'dE_F':>8} {'diff':>8}")
    print("-" * 78)
    for r in rows:
        print(f"{r['seed']:>6} {r['wl']:>11.3f} {r['wf']:>11.3f} "
              f"{r['dE_l']:>+8.3f} {r['dE_f']:>+8.3f} {r['d']:>+8.3f}")

    diffs = [r["d"] for r in rows]
    signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in diffs]
    consistent = all(s == signs[0] for s in signs) and signs[0] != 0
    sd = stdev(diffs)
    ratio = abs(mean(diffs)) / sd if sd > 0 else float("inf")

    print("-" * 78)
    print(f"{'mean':>6} {'':>11} {'':>11} {'':>8} {'':>8} "
          f"{mean(diffs):>+8.3f}")
    print(f"{'sd':>6} {'':>11} {'':>11} {'':>8} {'':>8} {sd:>8.3f}")

    # Do the COMPONENTS stay put within each pair? That is what makes
    # it a barrier rather than a state change.
    print(f"\n{'seed':>6} {'|d baseline mean|':>19} {'|d activated mean|':>20}")
    print("-" * 78)
    max_shift = 0.0
    for r in rows:
        db = abs(r["mb_l"] - r["mb_f"])
        da = abs(r["ma_l"] - r["ma_f"])
        max_shift = max(max_shift, db, da)
        print(f"{r['seed']:>6} {db:>19.3f} {da:>20.3f}")

    print(f"\n{'=' * 78}")
    print("READING THIS")
    print(f"{'=' * 78}")
    print(f"""
dE difference across {len(rows)} pairs: mean {mean(diffs):+.3f}, "
sd {sd:.3f}, mean/sd {ratio:.2f}
sign consistent: {'YES' if consistent else 'no'}
largest component shift within a pair: {max_shift:.3f}""")

    if consistent and ratio > 1.5 and max_shift < 0.35:
        print("""
The barrier shift holds in every pair, with a mean well outside the
between-seed spread, and the components stay put within each pair.
That is the same bar the protest-reduction result cleared, and it
makes this a replicated finding rather than a striking single
measurement.

The claim: learning does not change what the protest state is, it
changes how often the child enters it -- a barrier raised by roughly
the amount above, in units of temperature.""")
    elif consistent and ratio > 1.5:
        print(f"""
The barrier shift is consistent across pairs, but the components move
by up to {max_shift:.3f} within a pair. That is enough to muddy the
barrier-versus-state distinction: part of what looks like changed
occupancy may be the states themselves shifting. Worth reporting the
occupancy change without the activation-energy interpretation.""")
    elif consistent:
        print("""
Sign holds but the spread is wide relative to the effect. Suggestive,
not established -- more pairs would settle it, and at three pairs a
consistent sign has a 25% chance of arising by luck.""")
    else:
        print("""
The sign flips between pairs. The pooled activation-energy difference
was between-seed heterogeneity, not a barrier change, and this joins
the four earlier findings that did not survive per-seed examination.

Worth noting what that means concretely: pooling several seeds whose
means differ produces a distribution that IS a mixture -- of seeds.
EM finds two components because two components are there, but they are
runs rather than states.""")

    print("""
CAVEATS. Each pair has roughly 180 values per condition, so the
mixture parameters are far less stable than in the pooled fit; the
per-seed activation energies will be noisy even if the underlying
effect is real. Components that overlap heavily can swap identity
between fits, which shows up as an implausibly large single-pair
difference. And with three pairs, sign consistency alone is weak
evidence -- the mean-to-spread ratio is doing most of the work.""")


if __name__ == "__main__":
    main()