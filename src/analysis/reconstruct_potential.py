"""
Reconstruct the potential from the data instead of assuming one.

WHY THIS IS DIFFERENT FROM WHAT CAME BEFORE
-------------------------------------------
Three thermodynamic framings have now failed, and they failed the same
way: a mechanism was ASSERTED, consequences were derived, predictions
were tested against the system.

    FEP                 posited a controller -> ablated to null
    dissipative         posited flux         -> no threshold found
    double-well         posited feedback     -> unimodal at 8 seeds

This inverts the method. Rather than guessing the landscape and
checking its consequences, it estimates the landscape FROM the
trajectories. That is how non-equilibrium physics handles systems too
complicated to derive -- climate, neural recordings, molecular
dynamics. Nothing is assumed about the number of wells, the presence
of feedback, or whether a potential exists at all.

THE METHOD
----------
For a one-dimensional stochastic process the Kramers-Moyal
coefficients are conditional moments of the increments:

    drift      D1(x) = <dx | x>          systematic force at x
    diffusion  D2(x) = <dx^2 | x> / 2    noise amplitude at x

Both are estimated by binning on x and averaging within bins. If the
dynamics are a gradient flow with state-dependent noise, the
stationary distribution is exp(-U_eff) with

    U_eff(x) = - integral D1(x)/D2(x) dx

Two minima in U_eff means bistability -- MEASURED, not predicted. One
minimum means there is no second basin to find.

WHAT THE FROZEN RUNS ARE FOR
----------------------------
Runs made with --no-update have fixed weights, so any motion in the
activation series is pure fluctuation. That is the thermal baseline:
D1 should be near zero throughout. Learning runs have fluctuation PLUS
whatever systematic force the updates create. Comparing the two
separates the force from the noise, which every previous analysis
conflated.

A THIRD THING IT CAN SHOW
-------------------------
In more than one dimension the drift field can be decomposed into a
gradient part and a rotational part. A large rotational component
means the dynamics are NOT a gradient flow -- there is no potential,
and the system carries persistent probability currents. That is a
genuine non-equilibrium signature and a more interesting outcome than
any particular well shape.

Usage:
    python src/analysis/reconstruct_potential.py --learning runs/learning/L600_on.jsonl \\
                                    --frozen   runs/learning/L600_off.jsonl
    python src/analysis/reconstruct_potential.py --learning "runs_*/*_on.jsonl" \\
                                    --frozen "runs_*/*_off.jsonl" --bins 12
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

from fep_layer import CONCEPT_WEIGHTS


def episode_means(rows, concept):
    """Collapse a run to one value per episode.

    WHY: the turn-level analysis found no temporal structure --
    entropy_production.py showed shuffling the turn order changed
    sigma by less than a thousandth, in both learning and frozen arms.
    But real time dependence exists at the EPISODE scale: protesting
    moved +0.096 -> -0.475 across a run in the drift analysis, and
    -1.2 -> +0.7 across 60 episodes in the timescale run.

    So the fast variable is structureless noise and the slow variable
    carries the dynamics. Averaging within episodes removes the former
    and leaves the latter, which is where any drift field worth
    reconstructing has to live.
    """
    from collections import defaultdict
    by_ep = defaultdict(list)
    for r in rows:
        by_ep[r["episode"]].append(r["activations"][concept])
    return [sum(v) / len(v) for _, v in sorted(by_ep.items())]


def load_series(paths, concept, level="turn"):
    """Concatenate per-run trajectories. Transitions are never taken
    across a run boundary.

    level="turn"    one point per exchange (fast, structureless)
    level="episode" one point per episode  (slow, where the drift is)
    """
    runs = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                rows = [json.loads(line) for line in f]
            if not rows or "activations" not in rows[0]:
                continue
            if concept not in rows[0]["activations"]:
                continue
            if level == "episode":
                runs.append(episode_means(rows, concept))
            else:
                runs.append([r["activations"][concept] for r in rows])
    return runs


def transitions(runs):
    """(x, dx) pairs, within runs only."""
    out = []
    for series in runs:
        for i in range(len(series) - 1):
            out.append((series[i], series[i + 1] - series[i]))
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def km_coefficients(pairs, bins, lo=None, hi=None, min_per_bin=15):
    """Drift and diffusion, binned on x.

    min_per_bin matters: a bin with a handful of transitions gives a
    drift estimate dominated by noise, and a spurious well is exactly
    what that looks like.
    """
    xs = [x for x, _ in pairs]
    lo = min(xs) if lo is None else lo
    hi = max(xs) if hi is None else hi
    width = (hi - lo) / bins

    grouped = [[] for _ in range(bins)]
    for x, dx in pairs:
        if not (lo <= x <= hi):
            continue
        b = min(int((x - lo) / width), bins - 1)
        grouped[b].append(dx)

    out = []
    for b, dxs in enumerate(grouped):
        centre = lo + (b + 0.5) * width
        n = len(dxs)
        if n < min_per_bin:
            out.append({"x": centre, "n": n, "D1": None, "D2": None,
                        "sem": None})
            continue
        d1 = mean(dxs)
        d2 = mean([d * d for d in dxs]) / 2
        sd = (mean([(d - d1) ** 2 for d in dxs])) ** 0.5
        out.append({"x": centre, "n": n, "D1": d1, "D2": d2,
                    "sem": sd / (n ** 0.5)})
    return out


def potential(coeffs):
    """U(x) = -integral D1/D2 dx, by the trapezoid rule over usable
    bins. Only defined up to an additive constant, so it is shifted to
    a minimum of zero."""
    usable = [c for c in coeffs if c["D1"] is not None and c["D2"] > 0]
    if len(usable) < 3:
        return []
    U, u = [], 0.0
    prev = None
    for c in usable:
        ratio = c["D1"] / c["D2"]
        if prev is not None:
            dx = c["x"] - prev["x"]
            u -= 0.5 * (ratio + prev["ratio"]) * dx
        U.append({"x": c["x"], "U": u, "n": c["n"]})
        prev = {"x": c["x"], "ratio": ratio}
    lo = min(p["U"] for p in U)
    for p in U:
        p["U"] -= lo
    return U


def count_wells(U, min_depth=0.05):
    """Interior local minima deeper than min_depth relative to the
    barriers around them. The threshold keeps small wiggles from
    counting as wells."""
    if len(U) < 5:
        return 0, []
    wells = []
    for i in range(1, len(U) - 1):
        if U[i]["U"] < U[i - 1]["U"] and U[i]["U"] <= U[i + 1]["U"]:
            left = max(p["U"] for p in U[:i + 1])
            right = max(p["U"] for p in U[i:])
            depth = min(left, right) - U[i]["U"]
            if depth >= min_depth:
                wells.append({"x": U[i]["x"], "depth": depth})
    return len(wells), wells


def sparkline(vals, height=9):
    """Crude vertical plot, so the shape is visible without matplotlib."""
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi == lo:
        hi = lo + 1e-9
    rows = []
    for h in range(height, 0, -1):
        thresh = lo + (hi - lo) * (h - 0.5) / height
        rows.append("".join("#" if v >= thresh else " " for v in vals))
    return rows


def analyse(label, runs, concept, bins, min_per_bin, lo=None, hi=None):
    pairs = transitions(runs)
    print(f"\n{'=' * 74}")
    print(f"{label}  --  {concept}")
    print(f"{'=' * 74}")
    print(f"{len(runs)} run(s), {len(pairs)} transitions")
    if not pairs:
        return None, None

    coeffs = km_coefficients(pairs, bins, lo, hi, min_per_bin)
    print(f"\n{'x':>8} {'n':>6} {'drift D1':>11} {'+-sem':>9} "
          f"{'diffusion D2':>14}")
    print("-" * 74)
    for c in coeffs:
        if c["D1"] is None:
            print(f"{c['x']:>8.2f} {c['n']:>6}      (too few)")
        else:
            flag = " *" if abs(c["D1"]) > 2 * c["sem"] else ""
            print(f"{c['x']:>8.2f} {c['n']:>6} {c['D1']:>+11.4f} "
                  f"{c['sem']:>9.4f} {c['D2']:>14.4f}{flag}")
    print("  (* drift more than twice its standard error from zero)")

    U = potential(coeffs)
    if U:
        n_wells, wells = count_wells(U)
        print(f"\nreconstructed potential U(x) = -int D1/D2 dx")
        vals = [p["U"] for p in U]
        for row in sparkline(vals):
            print("   " + row)
        print("   " + "".join("^" if any(abs(p["x"] - w["x"]) < 1e-9
                                         for w in wells) else "-"
                              for p in U))
        print(f"   x from {U[0]['x']:.2f} to {U[-1]['x']:.2f}")
        print(f"\ninterior wells: {n_wells}"
              + ("  at " + ", ".join(f"x={w['x']:.2f} "
                                     f"(depth {w['depth']:.3f})"
                                     for w in wells) if wells else ""))
    return coeffs, U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learning", nargs="+", required=True,
                    help="affect jsonl from runs WITH weight updates")
    ap.add_argument("--frozen", nargs="+", default=None,
                    help="affect jsonl from --no-update runs; the "
                         "thermal baseline")
    ap.add_argument("--concept", default="protesting",
                    choices=list(CONCEPT_WEIGHTS))
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--min-per-bin", type=int, default=15)
    ap.add_argument("--level", choices=["turn", "episode"], default="turn",
                    help="episode averages over each episode first. The "
                         "turn level has no detectable temporal structure "
                         "(see entropy_production.py), so episode is where "
                         "any real drift field should be.")
    args = ap.parse_args()

    learn_runs = load_series(args.learning, args.concept, args.level)
    if not learn_runs:
        raise SystemExit("no usable learning runs found -- check the paths "
                         "and that the files have been scored")

    froz_runs = (load_series(args.frozen, args.concept, args.level)
                 if args.frozen else [])

    # Shared binning range, so the two are directly comparable.
    all_x = [x for r in learn_runs + froz_runs for x in r]
    lo, hi = min(all_x), max(all_x)

    print(f"resolution: {args.level}-level increments")
    lc, lU = analyse("LEARNING (weights update)", learn_runs,
                     args.concept, args.bins, args.min_per_bin, lo, hi)
    fc, fU = analyse("FROZEN (thermal baseline)", froz_runs,
                     args.concept, args.bins, args.min_per_bin, lo, hi) \
        if froz_runs else (None, None)

    print(f"\n{'=' * 74}")
    print("READING THIS")
    print(f"{'=' * 74}")

    # The frozen baseline will itself show a well: any stationary
    # distribution does, since fluctuations around a mean look like a
    # restoring force. Verified on synthetic iid-normal data, which
    # reconstructed as a single well despite having no dynamics at all.
    # So the comparison must be about how the SHAPES differ, not about
    # whether the learning runs have a well.
    if fc:
        f_sig = sum(1 for c in fc if c["D1"] is not None
                    and abs(c["D1"]) > 2 * c["sem"])
        l_sig = sum(1 for c in lc if c["D1"] is not None
                    and abs(c["D1"]) > 2 * c["sem"])
        print(f"""
bins with drift significantly away from zero:
    frozen   {f_sig}
    learning {l_sig}

The frozen runs have fixed weights, so any drift there is measurement
artifact rather than force -- it is the noise floor. Drift in the
learning runs beyond that floor is the systematic force the updates
create.""")
        if fc and lc:
            f_mean = mean([abs(c["D1"]) for c in fc if c["D1"] is not None])
            l_mean = mean([abs(c["D1"]) for c in lc if c["D1"] is not None])
            print(f"\nmean |D1|: frozen {f_mean:.4f}, learning {l_mean:.4f}"
                  f"  (ratio {l_mean / f_mean:.2f}x)"
                  if f_mean else "")

    if lU:
        n_wells, wells = count_wells(lU)
        f_wells = count_wells(fU)[0] if fU else None
        if f_wells is not None:
            print(f"\nwells: learning {n_wells}, frozen {f_wells}")
            print("(the frozen runs have fixed weights, so their well is")
            print(" the stationary distribution of the measure itself --")
            print(" only a DIFFERENCE in well structure is about learning)")
        if n_wells >= 2 and (f_wells is None or f_wells < 2):
            print(f"""
The reconstructed potential has {n_wells} interior wells where the
frozen baseline has fewer. That is
bistability MEASURED rather than predicted, and it is what the
double-well model claimed but the bimodality test did not find. Check
the per-bin counts before believing it -- a well built from a
low-population bin is noise.""")
        elif n_wells == 1:
            print("""
One interior well. The dynamics have a single attractor on this
coordinate, which is consistent with the bimodality test coming back
unimodal and with the single-well model's recovery prediction.""")
        else:
            print("""
No interior well -- the potential is monotonic across the measured
range, so the attractor lies at or beyond an edge. Either the range
is too narrow to contain it, or the drift is uniformly in one
direction, which would mean a steady force rather than relaxation
toward a state.""")

    print("""
CAVEATS. Kramers-Moyal estimation assumes the process is Markov in the
plotted coordinate; a single emotion activation almost certainly is
not, since the weights and the context also carry state. Bins with few
transitions give unreliable drift, which is exactly what a spurious
well looks like. And the increments here are one turn apart, which may
be long relative to the underlying dynamics.

What this method gets right is the direction of inference: the
landscape comes out of the data rather than being assumed and then
tested.""")


if __name__ == "__main__":
    main()