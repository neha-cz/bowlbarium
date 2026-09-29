"""
Two-state model: does learning raise an activation energy?

THE PATTERN THIS EXPLAINS
-------------------------
asymmetric_potential.py found something specific and slightly odd:

    minimum position   -0.544 vs -0.615   (essentially unchanged)
    10th percentile    -0.163              (barely moves)
    90th percentile    -0.858              (moves a lot)
    99th percentile    +0.170              (unchanged, wrong sign)

So learning compresses the UPPER-MIDDLE of the distribution while
leaving both the mode and the extreme tail alone. A single shifting
population cannot do that -- shifting a distribution moves all its
quantiles together.

Two populations can. If the child occupies either a BASELINE state or
an ACTIVATED state, and learning changes how often it enters the
activated one without changing what either state looks like, then:

    the mode is set by the baseline state        -> unchanged
    the shoulder is the activated population     -> compressed
    the far tail is the activated state's own
      tail, or a rarer process                   -> largely unchanged

which is exactly the observed pattern.

THE THERMODYNAMICS
------------------
For a two-state system the population ratio is Boltzmann:

    p_activated / p_baseline = exp(-dE / T)

so the activation energy follows directly from the mixing weights:

    dE / T = -log(w_activated / w_baseline)

and the difference between conditions,

    d(dE) = dE_learning - dE_frozen

is the change in barrier height in units of temperature. That is a
measured quantity with units, not an analogy -- and it makes a
falsifiable prediction about which parameters move:

    barrier change   -> mixing WEIGHTS differ, component means and
                        widths do not
    state change     -> the components themselves differ, which would
                        be a different mechanism entirely

WHAT IS FIT
-----------
A two-component Gaussian mixture by expectation-maximisation, per
condition. Compared against a single Gaussian by BIC, so the two-state
description has to earn its extra parameters before anything is read
from it.

Usage:
    python src/analysis/two_state.py \\
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


def var(xs, m=None):
    m = mean(xs) if m is None else m
    return sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)


def npdf(x, m, v):
    if v <= 1e-12:
        return 1e-300
    return math.exp(-((x - m) ** 2) / (2 * v)) / math.sqrt(2 * math.pi * v)


def fit_mixture(xs, iters=400, seed=0, tol=1e-9):
    """Two-component 1-D Gaussian mixture by EM.

    Initialised by splitting at the median, which is a stable starting
    point for a skewed distribution -- random starts on this data land
    in different optima run to run.
    """
    rng = random.Random(seed)
    s = sorted(xs)
    lo = s[:len(s) // 2]
    hi = s[len(s) // 2:]
    m1, m2 = mean(lo), mean(hi)
    v1, v2 = max(var(lo), 1e-4), max(var(hi), 1e-4)
    w1 = 0.5

    prev = None
    for _ in range(iters):
        # E step
        resp = []
        ll = 0.0
        for x in xs:
            a = w1 * npdf(x, m1, v1)
            b = (1 - w1) * npdf(x, m2, v2)
            t = a + b
            if t <= 0:
                resp.append(0.5)
                continue
            resp.append(a / t)
            ll += math.log(t)
        # M step
        n1 = sum(resp)
        n2 = len(xs) - n1
        if n1 < 2 or n2 < 2:
            break
        w1 = n1 / len(xs)
        m1 = sum(r * x for r, x in zip(resp, xs)) / n1
        m2 = sum((1 - r) * x for r, x in zip(resp, xs)) / n2
        v1 = max(sum(r * (x - m1) ** 2 for r, x in zip(resp, xs)) / n1, 1e-4)
        v2 = max(sum((1 - r) * (x - m2) ** 2
                     for r, x in zip(resp, xs)) / n2, 1e-4)
        if prev is not None and abs(ll - prev) < tol:
            break
        prev = ll

    # order so component 1 is the lower-mean (baseline) state
    if m1 > m2:
        m1, m2, v1, v2, w1 = m2, m1, v2, v1, 1 - w1
    return {"w_base": w1, "w_act": 1 - w1, "m_base": m1, "m_act": m2,
            "sd_base": v1 ** 0.5, "sd_act": v2 ** 0.5, "ll": prev or 0.0}


def loglik_single(xs):
    m, v = mean(xs), var(xs)
    return sum(math.log(max(npdf(x, m, v), 1e-300)) for x in xs)


def bic(ll, k, n):
    return k * math.log(n) - 2 * ll


def activation_energy(fit):
    """dE/T = -log(w_act / w_base)."""
    if fit["w_act"] <= 0 or fit["w_base"] <= 0:
        return float("nan")
    return -math.log(fit["w_act"] / fit["w_base"])


def boot_dE(xs, n_boot=300, seed=1):
    rng = random.Random(seed)
    out = []
    for _ in range(n_boot):
        s = [xs[rng.randrange(len(xs))] for _ in range(len(xs))]
        try:
            out.append(activation_energy(fit_mixture(s)))
        except Exception:
            continue
    out = sorted(v for v in out if v == v)
    if len(out) < 20:
        return None, None
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def report(name, xs):
    f = fit_mixture(xs)
    n = len(xs)
    ll1 = loglik_single(xs)
    b1 = bic(ll1, 2, n)
    b2 = bic(f["ll"], 5, n)
    dE = activation_energy(f)
    lo, hi = boot_dE(xs)

    print(f"\n{'=' * 76}")
    print(name)
    print(f"{'=' * 76}")
    print(f"n = {n}")
    print(f"\n{'component':<12} {'weight':>9} {'mean':>9} {'sd':>8}")
    print("-" * 76)
    print(f"{'baseline':<12} {f['w_base']:>9.3f} {f['m_base']:>+9.3f} "
          f"{f['sd_base']:>8.3f}")
    print(f"{'activated':<12} {f['w_act']:>9.3f} {f['m_act']:>+9.3f} "
          f"{f['sd_act']:>8.3f}")

    print(f"\n  BIC single Gaussian {b1:.1f}, two-state {b2:.1f} "
          f"-> {'two-state' if b2 < b1 else 'single'} preferred "
          f"(lower is better)")
    ci = f" [{lo:+.3f}, {hi:+.3f}]" if lo is not None else ""
    print(f"  activation energy dE/T = {dE:+.3f}{ci}")
    return f, dE, (b2 < b1), (lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs=2, metavar=("NAME", "GLOB"), required=True)
    ap.add_argument("--b", nargs=2, metavar=("NAME", "GLOB"), required=True)
    ap.add_argument("--concept", default="protesting",
                    choices=list(CONCEPT_WEIGHTS))
    args = ap.parse_args()

    na, pa = args.a
    nb, pb = args.b
    A = load_values(pa, args.concept)
    B = load_values(pb, args.concept)
    if len(A) < 100 or len(B) < 100:
        raise SystemExit(f"too little data: {len(A)}, {len(B)}")

    print("=" * 76)
    print(f"TWO-STATE MODEL  --  {args.concept}")
    print("=" * 76)
    print("""
A distribution with an unchanged mode, a compressed shoulder and an
unchanged far tail is what two populations look like when the MIXING
WEIGHT changes. For a two-state system the ratio is Boltzmann, so the
weights give an activation energy directly.""")

    fA, dEA, okA, ciA = report(na.upper(), A)
    fB, dEB, okB, ciB = report(nb.upper(), B)

    print(f"\n{'=' * 76}")
    print("COMPARISON")
    print(f"{'=' * 76}")
    print(f"\n{'':<22} {na:>12} {nb:>12} {'difference':>13}")
    print("-" * 76)
    for label, key in (("activated fraction", "w_act"),
                       ("baseline mean", "m_base"),
                       ("activated mean", "m_act"),
                       ("baseline sd", "sd_base"),
                       ("activated sd", "sd_act")):
        print(f"{label:<22} {fA[key]:>+12.3f} {fB[key]:>+12.3f} "
              f"{fA[key] - fB[key]:>+13.3f}")
    print(f"{'dE / T':<22} {dEA:>+12.3f} {dEB:>+12.3f} "
          f"{dEA - dEB:>+13.3f}")

    print(f"\n{'=' * 76}")
    print("READING THIS")
    print(f"{'=' * 76}")

    if not (okA and okB):
        print("""
The two-state description does NOT beat a single Gaussian by BIC in at
least one condition. It has not earned its extra parameters, and the
activation energies below should not be interpreted.""")

    d_w = fA["w_act"] - fB["w_act"]
    d_mb = abs(fA["m_base"] - fB["m_base"])
    d_ma = abs(fA["m_act"] - fB["m_act"])
    d_dE = dEA - dEB

    print(f"""
The prediction distinguishes two mechanisms:

  BARRIER change   the mixing weights differ, the component means do
                   not. Learning changes how OFTEN the activated state
                   is entered, not what it is.
  STATE change     the components themselves move. A different
                   mechanism, and not an activation energy at all.

  activated fraction differs by {d_w:+.3f}
  baseline mean differs by      {d_mb:.3f}
  activated mean differs by     {d_ma:.3f}""")

    if abs(d_w) > 0.05 and d_mb < 0.2 and d_ma < 0.3:
        overlap = (ciA[0] is not None and ciB[0] is not None
                   and not (ciA[1] < ciB[0] or ciB[1] < ciA[0]))
        print(f"""
This is the barrier case. The states themselves are unchanged; what
differs is the population in them. Learning shifts the activation
energy by {d_dE:+.3f} T -- a measured barrier change with units,
which is the first quantity in this project derived from a
thermodynamic relation rather than assumed by one.""")
        if overlap:
            print("""
  But the bootstrap intervals on the two energies OVERLAP, so the
  difference is not resolved at this sample size. Directionally
  suggestive, not established.""")
    elif d_mb >= 0.2 or d_ma >= 0.3:
        print("""
The COMPONENTS moved, not just their weights. That is a state change
rather than a barrier change, so the activation-energy reading does
not apply -- the two-state picture may still describe the data, but
the thermodynamic interpretation of the weights does not follow.""")
    else:
        print("""
The mixing weights barely differ. The two-state account does not
explain the shoulder compression, and the pattern needs another
explanation.""")

    print("""
CAVEATS. EM on a skewed unimodal distribution will always return two
components whether or not two populations exist -- the BIC comparison
is what guards against reading structure into a fit, and it is a weak
guard. Component identities are not stable across bootstrap samples
when the components overlap heavily, which widens the intervals. And
the Boltzmann reading assumes the two states are in equilibrium with
each other, which the entropy-production null supports but did not
test directly.""")


if __name__ == "__main__":
    main()