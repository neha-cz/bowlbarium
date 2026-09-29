"""
Temperature from fluctuation-dissipation.

THE PROBLEM THIS FIXES
----------------------
two_state_per_seed.py found the activated-state occupancy drops from
0.66 to 0.39 under learning, replicated across three seed pairs. That
was reported as an activation energy of 1.7 kT.

But T was never measured. dE/T = -log(w_act/w_base) is a LOG-ODDS
RATIO, and calling it an energy in units of temperature requires
knowing T. The activations are z-scored, so the scale is arbitrary and
"1.7 kT" is a log-odds ratio in physics costume. That is what makes
the framing decorative rather than substantive -- the same flaw that
sank the ten framings before it.

The fluctuation-dissipation theorem determines T from data:

    chi = <dx^2> / T        =>        T = <dx^2> / chi

    <dx^2>   spontaneous fluctuation -- the variance when nothing is
             being driven
    chi      response -- how far the mean moves per unit of applied
             perturbation

Both halves exist in this project's data:

    fluctuation  the FROZEN runs. Weights are fixed, so all variance
                 is spontaneous. This is the thermal bath.
    response     the reliability manipulation. Changing caregiver
                 availability by a known amount moves the mean by a
                 measurable amount, so chi = d<x> / d(reliability).

Their ratio is T in the units the activations are measured in. With T
determined, the two-state occupancy becomes an energy on a fixed scale
rather than a bare log-odds.

WHAT WOULD MAKE THIS SUBSTANTIVE
--------------------------------
1. T is well-determined -- the response is linear in the perturbation,
   so chi is a slope rather than one difference divided by another.
2. T is stable across conditions. A temperature that changes when the
   perturbation changes is not a temperature.
3. T can be compared against beta, the KL coefficient. toy_model.py
   derived T = beta; effective_temperature.py tested it with variance
   alone and found T_eff flat across a 25-fold beta range. FDT uses
   response OVER fluctuation, which is the actual definition, so this
   is a fairer test of the same claim.

Usage:
    python src/analysis/fdt_temperature.py \\
        --frozen "runs/learning/*_off.jsonl" \\
        --response 0.15 "runs/reliability/R900_0.15_learn.jsonl" \\
        --response 0.40 "runs/reliability/R900_0.4_learn.jsonl" \\
        --response 0.65 "runs/reliability/R900_0.65_learn.jsonl" \\
        --response 0.90 "runs/reliability/R900_0.9_learn.jsonl"
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
import random

from fep_layer import CONCEPT_WEIGHTS


def load_values(pattern, concept):
    vals = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                a = r.get("activations")
                if a and concept in a:
                    vals.append(a[concept])
    return vals


def mean(xs):
    return sum(xs) / len(xs)


def variance(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def linfit(xs, ys):
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None, None, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    inter = my - slope * mx
    pred = [slope * x + inter for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, inter, r2


def boot_ci(fn, n_boot=800, seed=0):
    vals = sorted(v for v in (fn(i) for i in range(n_boot)) if v is not None)
    if len(vals) < 20:
        return None, None
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", nargs="+", required=True,
                    help="fixed-weight runs: the fluctuation half")
    ap.add_argument("--response", nargs=2, action="append",
                    metavar=("RELIABILITY", "GLOB"), required=True,
                    help="a reliability level and its runs; repeatable")
    ap.add_argument("--concept", default="protesting",
                    choices=list(CONCEPT_WEIGHTS))
    ap.add_argument("--occupancy", nargs=2, type=float,
                    default=[0.386, 0.662],
                    metavar=("W_LEARN", "W_FROZEN"),
                    help="activated fractions from two_state.py, used to "
                         "convert the log-odds into an energy once T is "
                         "known")
    args = ap.parse_args()

    print("=" * 76)
    print(f"TEMPERATURE FROM FLUCTUATION-DISSIPATION  --  {args.concept}")
    print("=" * 76)

    # ---------- fluctuation ----------
    froz = []
    for p in args.frozen:
        froz += load_values(p, args.concept)
    if len(froz) < 100:
        raise SystemExit(f"too few frozen values ({len(froz)})")
    fluct = variance(froz)

    print(f"\nFLUCTUATION  (frozen runs: weights fixed, so all variance is")
    print(f"              spontaneous)")
    print(f"  n = {len(froz)},  <dx^2> = {fluct:.4f}")

    # ---------- response ----------
    levels = []
    for lvl, pattern in args.response:
        vals = load_values(pattern, args.concept)
        if len(vals) < 50:
            print(f"  skipping reliability {lvl}: {len(vals)} values")
            continue
        levels.append((float(lvl), vals))
    levels.sort()
    if len(levels) < 2:
        raise SystemExit("need at least two reliability levels")

    print(f"\nRESPONSE  (how the mean moves with the perturbation)")
    print(f"\n{'reliability':>12} {'n':>6} {'mean':>10} {'variance':>10}")
    print("-" * 76)
    for lvl, vals in levels:
        print(f"{lvl:>12.2f} {len(vals):>6} {mean(vals):>+10.4f} "
              f"{variance(vals):>10.4f}")

    xs = [l for l, _ in levels]
    ys = [mean(v) for _, v in levels]
    chi, _, r2 = linfit(xs, ys)

    print(f"\n  susceptibility chi = d<x>/d(reliability) = {chi:+.4f}")
    print(f"  linearity R^2 = {r2:.3f}")
    if r2 < 0.7:
        print("""
  !! The response is NOT linear in the perturbation. FDT assumes
     linear response, so chi is not well defined and the temperature
     below should not be trusted. This is the first thing that has to
     hold for any of it to mean anything.""")

    # ---------- temperature ----------
    T = fluct / abs(chi) if chi else float("nan")
    print(f"\n{'=' * 76}")
    print("TEMPERATURE")
    print(f"{'=' * 76}")
    print(f"\n  T = <dx^2> / |chi| = {fluct:.4f} / {abs(chi):.4f} "
          f"= {T:.4f}")

    # bootstrap over runs, resampling values within each level
    rng = random.Random(0)

    def one_boot(_):
        f = [froz[rng.randrange(len(froz))] for _ in range(len(froz))]
        bx, by = [], []
        for lvl, vals in levels:
            s = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
            bx.append(lvl)
            by.append(mean(s))
        c, _, _ = linfit(bx, by)
        if not c:
            return None
        return variance(f) / abs(c)

    lo, hi = boot_ci(one_boot, n_boot=600)
    if lo is not None:
        print(f"  95% CI [{lo:.4f}, {hi:.4f}]")
        width = (hi - lo) / T if T else float("inf")
        print(f"  interval width is {width:.0%} of the estimate")
        if width > 1.5:
            print("""
  The interval is wider than the estimate. T is not determined by
  this data, and an energy expressed in units of it inherits that
  uncertainty entirely.""")

    # ---------- stability check ----------
    print(f"\n{'=' * 76}")
    print("IS IT ACTUALLY A TEMPERATURE?")
    print(f"{'=' * 76}")
    print("""
A temperature must not depend on which perturbation was used to
measure it. Recomputing chi from each adjacent pair of levels gives
independent estimates -- if they scatter, the quantity is not a
temperature.""")
    print(f"\n{'levels':>16} {'chi':>10} {'T':>10}")
    print("-" * 76)
    Ts = []
    for i in range(len(levels) - 1):
        (l1, v1), (l2, v2) = levels[i], levels[i + 1]
        c = (mean(v2) - mean(v1)) / (l2 - l1)
        t = fluct / abs(c) if c else float("nan")
        Ts.append(t)
        print(f"{f'{l1:.2f}-{l2:.2f}':>16} {c:>+10.4f} {t:>10.4f}")

    finite = [t for t in Ts if t == t and t < 1e6]
    if len(finite) >= 2:
        spread = (max(finite) - min(finite)) / mean(finite)
        print(f"\n  spread across pairs: {spread:.0%} of the mean")
        if spread > 0.5:
            print("""
  The estimates disagree by more than half their own size. T depends
  on which perturbation measured it, which means it is not a
  temperature -- it is a ratio that happens to have the right units.
  The two-state energy therefore stays a log-odds ratio.""")
        else:
            print("""
  The estimates agree reasonably. T behaves like a state variable
  rather than an artifact of the particular perturbation, which is the
  minimum requirement for calling it a temperature.""")

    # ---------- the payoff ----------
    print(f"\n{'=' * 76}")
    print("THE TWO-STATE ENERGY, ON A DETERMINED SCALE")
    print(f"{'=' * 76}")
    wl, wf = args.occupancy
    dEl = -math.log(wl / (1 - wl))
    dEf = -math.log(wf / (1 - wf))
    print(f"""
  activated fraction   learning {wl:.3f}, frozen {wf:.3f}
  log-odds             learning {dEl:+.3f}, frozen {dEf:+.3f}
  difference           {dEl - dEf:+.3f}   (this is what was reported
                                          as "1.7 kT")

  multiplied by T = {T:.4f}:

      barrier change = {(dEl - dEf) * T:+.4f} in activation units""")

    if lo is not None:
        print(f"      with T uncertainty: "
              f"[{(dEl - dEf) * lo:+.4f}, {(dEl - dEf) * hi:+.4f}]")

    print("""
Whether that number means anything rests entirely on the two checks
above -- linear response, and a T that does not depend on the
perturbation used to measure it. If either fails, the honest statement
remains the occupancy change itself: the activated state is entered
less often, by a factor that is measured and replicated, with no
energy scale attached.""")


if __name__ == "__main__":
    main()