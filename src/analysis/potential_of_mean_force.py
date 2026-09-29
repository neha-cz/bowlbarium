"""
Boltzmann inversion: the potential of mean force, from the distribution.

WHY THIS IS LICENSED HERE
-------------------------
entropy_production.py found no detectable irreversibility at either
turn or episode resolution, in either the learning or frozen arm. The
system is consistent with detailed balance. That is precisely the
condition under which the equilibrium relation holds:

    p(x)  proportional to  exp(-U(x) / T)     =>     U(x) = -T log p(x)

So the potential can be read off the observed distribution. This is
Boltzmann inversion -- the standard route to a potential of mean force
in molecular dynamics -- and it is a DIFFERENT estimator from
reconstruct_potential.py, which used drift and diffusion. At
equilibrium the two must agree, so comparing them is an independent
check on both.

WHAT THE SHAPE FINDING MEANS PHYSICALLY
---------------------------------------
decompose_shift.py found 80% of the learning/frozen divergence lives
in higher moments, with excess kurtosis +1.78 (learning) against +0.16
(frozen).

Heavier-than-Gaussian tails mean the density falls off more slowly at
large |x|, which means U grows more slowly than quadratic there. A
quadratic U is a linear restoring force -- a harmonic trap. Anything
flatter is ANHARMONIC: the restoring force weakens far from the
centre.

So the prediction from the kurtosis difference is that the learning
condition has a SOFTER trap. The child is less strongly pulled back
when it is far from typical behaviour. That is a physical statement
about confinement rather than about location, and no analysis in this
project has looked at it.

WHAT IS FIT
-----------
    U(x) = a2 x^2 + a3 x^3 + a4 x^4       (linear term absorbed by
                                           centring; constant dropped)

    a2 > 0, a4 ~ 0   harmonic: Gaussian, linear restoring force
    a4 < 0           softening: force weakens outward, heavy tails
    a4 > 0           stiffening: force grows faster than linear
    a3 =/= 0         asymmetric trap, one side steeper

Usage:
    python src/analysis/potential_of_mean_force.py \\
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


def moments(xs):
    n = len(xs)
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    sd = var ** 0.5
    z = [(x - m) / sd for x in xs]
    return m, var, sum(v ** 3 for v in z) / n, sum(v ** 4 for v in z) / n - 3


def kde(vals, grid, bw=None):
    """Gaussian kernel density. Smoother than a histogram, which
    matters because U = -log p amplifies noise in sparse bins.
    Silverman's rule for the bandwidth."""
    n = len(vals)
    m = mean(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5
    if bw is None:
        bw = 1.06 * sd * n ** (-1 / 5)
    out = []
    c = 1.0 / (n * bw * (2 * math.pi) ** 0.5)
    for g in grid:
        s = 0.0
        for v in vals:
            z = (g - v) / bw
            if abs(z) < 6:
                s += math.exp(-0.5 * z * z)
        out.append(c * s)
    return out, bw


def polyfit(xs, ys, degree, weights=None):
    """Least squares by normal equations. Small degree, so this is
    stable enough without external dependencies."""
    n = degree + 1
    w = weights if weights is not None else [1.0] * len(xs)
    A = [[sum(wi * (x ** (i + j)) for x, wi in zip(xs, w))
          for j in range(n)] for i in range(n)]
    b = [sum(wi * yi * (x ** i) for x, yi, wi in zip(xs, ys, w))
         for i in range(n)]
    # gaussian elimination
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(A[r][i]))
        A[i], A[p] = A[p], A[i]
        b[i], b[p] = b[p], b[i]
        if abs(A[i][i]) < 1e-14:
            return None
        for r in range(i + 1, n):
            f = A[r][i] / A[i][i]
            for c in range(i, n):
                A[r][c] -= f * A[i][c]
            b[r] -= f * b[i]
    coef = [0.0] * n
    for i in reversed(range(n)):
        s = b[i] - sum(A[i][j] * coef[j] for j in range(i + 1, n))
        coef[i] = s / A[i][i]
    return coef


def sparkline(vals, height=9):
    lo, hi = min(vals), max(vals)
    if hi == lo:
        hi = lo + 1e-9
    rows = []
    for h in range(height, 0, -1):
        t = lo + (hi - lo) * (h - 0.5) / height
        rows.append("".join("#" if v >= t else " " for v in vals))
    return rows


def analyse(name, vals, grid):
    m, var, sk, ku = moments(vals)
    dens, bw = kde(vals, grid)

    # U = -log p, in units of T. Only defined where the density is
    # resolvable; very low density means very few samples.
    floor = max(dens) * 1e-3
    pts = [(g, -math.log(d)) for g, d in zip(grid, dens) if d > floor]
    gx = [p[0] for p in pts]
    gu = [p[1] for p in pts]
    umin = min(gu)
    gu = [u - umin for u in gu]

    # Weight the fit by density: the tails have few samples and would
    # otherwise dominate a least-squares fit of a log quantity.
    wts = [d for g, d in zip(grid, dens) if d > floor]

    xc = [g - m for g in gx]                 # centre before fitting
    quad = polyfit(xc, gu, 2, wts)
    quart = polyfit(xc, gu, 4, wts)

    def rss(coef):
        return sum(w * (u - sum(c * (x ** i) for i, c in enumerate(coef))) ** 2
                   for x, u, w in zip(xc, gu, wts))

    r_quad, r_quart = rss(quad), rss(quart)
    improvement = (r_quad - r_quart) / r_quad if r_quad > 0 else 0.0

    print(f"\n{'=' * 76}")
    print(f"{name}")
    print(f"{'=' * 76}")
    print(f"n={len(vals)}  mean={m:+.3f}  sd={var**0.5:.3f}  "
          f"skew={sk:+.3f}  excess kurtosis={ku:+.3f}  bandwidth={bw:.3f}")

    print(f"\nU(x) = -log p(x), in units of T:")
    for row in sparkline(gu):
        print("   " + row)
    print(f"   x from {gx[0]:.2f} to {gx[-1]:.2f}, "
          f"depth {max(gu):.2f} T")

    print(f"\n  harmonic fit    U = {quad[2]:+.4f} x^2 "
          f"{quad[1]:+.4f} x")
    print(f"  anharmonic fit  U = {quart[4]:+.4f} x^4 {quart[3]:+.4f} x^3 "
          f"{quart[2]:+.4f} x^2 {quart[1]:+.4f} x")
    print(f"  quartic term improves the fit by {improvement:.0%}")

    return {"m": m, "var": var, "skew": sk, "kurt": ku,
            "quad": quad, "quart": quart, "improvement": improvement,
            "gx": gx, "gu": gu}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs=2, metavar=("NAME", "GLOB"), required=True)
    ap.add_argument("--b", nargs=2, metavar=("NAME", "GLOB"), required=True)
    ap.add_argument("--concept", default="protesting",
                    choices=list(CONCEPT_WEIGHTS))
    ap.add_argument("--points", type=int, default=60)
    args = ap.parse_args()

    na, pa = args.a
    nb, pb = args.b
    A = load_values(pa, args.concept)
    B = load_values(pb, args.concept)
    if len(A) < 100 or len(B) < 100:
        raise SystemExit(f"too little data: {len(A)}, {len(B)}")

    lo = min(min(A), min(B))
    hi = max(max(A), max(B))
    grid = [lo + (hi - lo) * i / (args.points - 1) for i in range(args.points)]

    print("=" * 76)
    print(f"POTENTIAL OF MEAN FORCE  --  {args.concept}")
    print("=" * 76)
    print("""
Licensed by the entropy-production null: with detailed balance holding,
p(x) proportional to exp(-U/T), so U = -log p is a genuine potential
rather than a curve fit.""")

    RA = analyse(f"{na.upper()}", A, grid)
    RB = analyse(f"{nb.upper()}", B, grid)

    print(f"\n{'=' * 76}")
    print("COMPARISON")
    print(f"{'=' * 76}")
    print(f"\n{'':<22} {na:>12} {nb:>12} {'difference':>13}")
    print("-" * 76)
    for key, label, idx in (("quad", "curvature a2 (x^2)", 2),
                            ("quart", "asymmetry a3 (x^3)", 3),
                            ("quart", "quartic a4 (x^4)", 4)):
        va = RA[key][idx]
        vb = RB[key][idx]
        print(f"{label:<22} {va:>+12.4f} {vb:>+12.4f} {va - vb:>+13.4f}")
    print(f"{'minimum position':<22} {RA['m']:>+12.3f} {RB['m']:>+12.3f} "
          f"{RA['m'] - RB['m']:>+13.3f}")
    print(f"{'excess kurtosis':<22} {RA['kurt']:>+12.3f} "
          f"{RB['kurt']:>+12.3f} {RA['kurt'] - RB['kurt']:>+13.3f}")

    print(f"\n{'=' * 76}")
    print("READING THIS")
    print(f"{'=' * 76}")

    a4_a, a4_b = RA["quart"][4], RB["quart"][4]
    a2_a, a2_b = RA["quad"][2], RB["quad"][2]

    print(f"""
The quartic coefficient is the anharmonicity: a4 < 0 means the trap
SOFTENS outward (restoring force weakens far from centre, heavy
tails); a4 > 0 means it stiffens.

    {na}: a4 = {a4_a:+.4f}
    {nb}: a4 = {a4_b:+.4f}""")

    if a4_a < 0 and a4_b >= 0:
        print(f"""
The {na} trap is anharmonic and softening while {nb} is not. Learning
weakens the confinement -- the child is pulled back less strongly when
far from typical behaviour. That is the physical content of the
kurtosis difference, and it is a claim about the FORCE rather than
about where the distribution sits.""")
    elif abs(a4_a - a4_b) < 0.02:
        print("""
The quartic coefficients are indistinguishable. The traps have the
same shape and differ only in position, which contradicts the 80%
shape share found in the divergence -- worth checking whether the
kurtosis difference is concentrated in a tail region too sparse for
the fit to weight.""")
    else:
        print(f"""
The two traps differ in anharmonicity by {a4_a - a4_b:+.4f}, so their
SHAPES differ, not only their positions. Which direction that runs
matters for interpretation: read the signs above rather than the
magnitude alone.""")

    print(f"""
Curvature at the minimum: {na} {a2_a:.4f}, {nb} {a2_b:.4f}. A smaller
a2 means a wider, shallower well -- weaker confinement near the centre
too, not just in the tails.""")

    print("""
CAVEATS. U = -log p amplifies noise wherever the density is small, so
the tails of these curves rest on few samples; the fit is
density-weighted for that reason, which also means the quartic term is
constrained mostly by the well rather than by the tails it is supposed
to describe. Kernel bandwidth affects the apparent curvature. And the
whole inversion assumes equilibrium -- justified by the entropy
production null, but that null was itself measured with limited
power.""")


if __name__ == "__main__":
    main()