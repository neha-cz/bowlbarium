"""
Asymmetric potential: fit each side of the minimum separately.

WHY THE POLYNOMIAL FIT FAILED
-----------------------------
potential_of_mean_force.py fit U(x) with a quartic and returned a4
POSITIVE in both conditions (+0.071 learning, +0.108 frozen), meaning
a stiffening trap -- the opposite of what excess kurtosis (+1.78 vs
+0.16) predicts.

The sparklines showed why. Both potentials rise steeply on the left,
run flat and low across the middle, then climb hard past x ~ 2. That
is a ONE-SIDED WALL, not a symmetric trap, and a3 = -0.41 in both
conditions confirms it. A symmetric polynomial family fits that badly:
the quartic term ends up describing the right-hand wall rather than
tail weight, which is why it disagrees with kurtosis.

Three further signs the fit was unstable:
  - the quartic improved the fit by 85-86% in BOTH conditions, so the
    anharmonicity did not distinguish them at all;
  - harmonic a2 (0.273, 0.260) and anharmonic a2 (0.596, 0.246)
    disagree wildly within each condition;
  - the coefficients therefore depend on model choice more than on the
    data.

WHAT THIS DOES INSTEAD
----------------------
Split at the minimum and characterise each side on its own terms:

    left wall    slope and curvature below the minimum
    right wall   slope and curvature above it
    asymmetry    ratio of the two

For a potential that confines on one side and releases on the other,
the walls are the meaningful objects. The question the kurtosis
difference poses is whether the RIGHT WALL -- the high-protest side --
moved between conditions, since that is where a heavy upper tail would
show.

Power-law walls are fit rather than polynomials:

    U(x) ~ A * |x - x0|^p

    p = 2   harmonic
    p < 2   softer than harmonic; the wall flattens outward
    p > 2   steeper; a hard wall

Fitting log U against log|x - x0| makes p a slope, which is far more
stable than a high-order polynomial coefficient.

Usage:
    python src/analysis/asymmetric_potential.py \\
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
import math

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


def quantile(xs, q):
    s = sorted(xs)
    i = q * (len(s) - 1)
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (i - lo) * (s[hi] - s[lo])


def kde(vals, grid):
    n = len(vals)
    m = mean(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5
    bw = 1.06 * sd * n ** (-1 / 5)
    c = 1.0 / (n * bw * (2 * math.pi) ** 0.5)
    out = []
    for g in grid:
        s = 0.0
        for v in vals:
            z = (g - v) / bw
            if abs(z) < 6:
                s += math.exp(-0.5 * z * z)
        out.append(c * s)
    return out, bw


def linfit(xs, ys, weights=None):
    """Weighted least squares slope and intercept."""
    w = weights if weights is not None else [1.0] * len(xs)
    sw = sum(w)
    if sw == 0 or len(xs) < 3:
        return None, None
    mx = sum(wi * x for x, wi in zip(xs, w)) / sw
    my = sum(wi * y for y, wi in zip(ys, w)) / sw
    num = sum(wi * (x - mx) * (y - my) for x, y, wi in zip(xs, ys, w))
    den = sum(wi * (x - mx) ** 2 for x, wi in zip(xs, w))
    if abs(den) < 1e-14:
        return None, None
    slope = num / den
    return slope, my - slope * mx


def wall_exponent(xs, us, wts, x0, side):
    """Fit U ~ A|x-x0|^p on one side, via log-log regression.

    p comes out as the slope, which is stable in a way a quartic
    coefficient is not.
    """
    lx, lu, lw = [], [], []
    for x, u, w in zip(xs, us, wts):
        d = (x - x0) if side == "right" else (x0 - x)
        if d <= 1e-6 or u <= 1e-6:
            continue
        lx.append(math.log(d))
        lu.append(math.log(u))
        lw.append(w)
    if len(lx) < 4:
        return None, None, 0
    p, logA = linfit(lx, lu, lw)
    return p, (math.exp(logA) if logA is not None else None), len(lx)


def analyse(name, vals, grid):
    dens, bw = kde(vals, grid)
    floor = max(dens) * 1e-3
    pts = [(g, d) for g, d in zip(grid, dens) if d > floor]
    xs = [p[0] for p in pts]
    us = [-math.log(p[1]) for p in pts]
    wts = [p[1] for p in pts]
    umin = min(us)
    us = [u - umin for u in us]
    x0 = xs[us.index(0.0)]

    lp, lA, ln = wall_exponent(xs, us, wts, x0, "left")
    rp, rA, rn = wall_exponent(xs, us, wts, x0, "right")

    # Height of each wall at a fixed distance from the minimum, which
    # is comparable across conditions in a way A and p separately are
    # not.
    def height_at(d, p, A, side):
        if p is None or A is None:
            return None
        return A * (d ** p)

    print(f"\n{'=' * 76}")
    print(f"{name}")
    print(f"{'=' * 76}")
    print(f"n={len(vals)}  minimum at x0={x0:+.3f}  bandwidth={bw:.3f}")
    print(f"range covered: {xs[0]:+.2f} to {xs[-1]:+.2f}")

    print(f"\n{'wall':<8} {'exponent p':>12} {'coefficient A':>15} "
          f"{'points':>8} {'U at d=1':>10} {'U at d=2':>10}")
    print("-" * 76)
    for side, p, A, n in (("left", lp, lA, ln), ("right", rp, rA, rn)):
        if p is None:
            print(f"{side:<8} {'(too few points)':>12}")
            continue
        h1 = height_at(1.0, p, A, side)
        h2 = height_at(2.0, p, A, side)
        print(f"{side:<8} {p:>12.3f} {A:>15.3f} {n:>8} "
              f"{h1:>10.3f} {h2:>10.3f}")

    print("\n  p = 2 harmonic; p < 2 softens outward; p > 2 hard wall")
    return {"x0": x0, "lp": lp, "lA": lA, "rp": rp, "rA": rA,
            "q90": quantile(vals, 0.90), "q99": quantile(vals, 0.99),
            "q10": quantile(vals, 0.10)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs=2, metavar=("NAME", "GLOB"), required=True)
    ap.add_argument("--b", nargs=2, metavar=("NAME", "GLOB"), required=True)
    ap.add_argument("--concept", default="protesting",
                    choices=list(CONCEPT_WEIGHTS))
    ap.add_argument("--points", type=int, default=80)
    args = ap.parse_args()

    na, pa = args.a
    nb, pb = args.b
    A = load_values(pa, args.concept)
    B = load_values(pb, args.concept)

    lo = min(min(A), min(B))
    hi = max(max(A), max(B))
    grid = [lo + (hi - lo) * i / (args.points - 1) for i in range(args.points)]

    print("=" * 76)
    print(f"ASYMMETRIC POTENTIAL  --  {args.concept}")
    print("=" * 76)
    print("""
The quartic fit failed because the potential is one-sided: steep on the
left, flat through the middle, climbing hard on the right. Each wall is
fit separately as a power law, which is stable where a symmetric
polynomial family is not.""")

    RA = analyse(na.upper(), A, grid)
    RB = analyse(nb.upper(), B, grid)

    print(f"\n{'=' * 76}")
    print("COMPARISON")
    print(f"{'=' * 76}")
    print(f"\n{'':<24} {na:>12} {nb:>12} {'difference':>13}")
    print("-" * 76)
    rows = [("minimum position", RA["x0"], RB["x0"]),
            ("left exponent p", RA["lp"], RB["lp"]),
            ("right exponent p", RA["rp"], RB["rp"]),
            ("left coefficient A", RA["lA"], RB["lA"]),
            ("right coefficient A", RA["rA"], RB["rA"]),
            ("10th percentile", RA["q10"], RB["q10"]),
            ("90th percentile", RA["q90"], RB["q90"]),
            ("99th percentile", RA["q99"], RB["q99"])]
    for label, va, vb in rows:
        if va is None or vb is None:
            continue
        print(f"{label:<24} {va:>+12.3f} {vb:>+12.3f} {va - vb:>+13.3f}")

    print(f"\n{'=' * 76}")
    print("READING THIS")
    print(f"{'=' * 76}")

    if None in (RA["rp"], RB["rp"], RA["lp"], RB["lp"]):
        print("\nOne or both walls had too few resolvable points to fit.")
        return

    dr = RA["rp"] - RB["rp"]
    dl = RA["lp"] - RB["lp"]
    print(f"""
right wall (the high-protest side, where a heavy upper tail lives):
    {na}  p = {RA['rp']:.3f}
    {nb}  p = {RB['rp']:.3f}
    difference {dr:+.3f}

left wall:
    {na}  p = {RA['lp']:.3f}
    {nb}  p = {RB['lp']:.3f}
    difference {dl:+.3f}""")

    if abs(dr) > 0.3 and abs(dr) > abs(dl) * 1.5:
        direction = "softer" if dr < 0 else "harder"
        print(f"""
The RIGHT wall differs between conditions and the left does not. The
{na} condition has a {direction} wall on the high-protest side, which
is where the excess-kurtosis difference should live. That is a
localised, one-sided change: learning alters what happens when the
child is far ABOVE typical protest, and leaves the low side alone.""")
    elif abs(dl) > 0.3 and abs(dl) > abs(dr) * 1.5:
        print("""
The LEFT wall differs and the right does not, so the change is on the
low-protest side. That is the opposite of where the kurtosis
difference suggested it would be, and worth reconciling against the
percentiles above before interpreting.""")
    elif abs(dr) < 0.2 and abs(dl) < 0.2:
        print("""
Neither wall differs appreciably. The two potentials have the same
shape on both sides and differ only in position -- which returns to the
simple translation account, and means the 80% shape share in the
divergence is NOT captured by wall shape. It would then most likely be
sparse tail structure that neither this fit nor the polynomial one can
resolve.""")
    else:
        print("""
Both walls differ somewhat, with neither dominating. The shape change
is distributed rather than localised to one side.""")

    print(f"""
The percentiles are the model-free check on all of this. If the 90th
and 99th differ much more than the 10th, the change really is in the
upper tail regardless of what any fit says.""")

    print("""
CAVEATS. Power-law wall fits depend on where the minimum is placed,
and the minimum is itself estimated from a kernel density. Points far
from the minimum carry few samples, so the exponents are constrained
mostly by the near region. Log-log regression weights small deviations
heavily unless downweighted, which is why the fit is density-weighted
here.""")


if __name__ == "__main__":
    main()