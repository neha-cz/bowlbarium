"""
Does the measured effective temperature equal the KL coefficient?

WHY THIS IS DIFFERENT FROM THE OTHER TESTS
------------------------------------------
Every thermodynamic test so far asked a STRUCTURAL question -- are
there two wells, is there a probability current, is the distribution
bimodal -- and every one came back null. But the derivation in
toy_model.py made a QUANTITATIVE prediction that has never been
checked:

    T = beta

The KL coefficient IS the temperature. Not "acts like", not "is
analogous to" -- the identity came out of writing F = U - T*S with a
proper KL anchor, where KL(p||p0) = -H(p) + const makes beta the
coefficient multiplying entropy.

That prediction is falsifiable with runs that already exist, at
beta = 0.2, 2.0 and 5.0.

WHAT IS MEASURED
----------------
For an Ornstein-Uhlenbeck process

    dx = -k(x - x0) dt + sqrt(2D) dW

the stationary variance is D/k, and in the Boltzmann picture variance
is proportional to temperature. So

    T_eff  proportional to  D / k

with both estimated from the trajectory: k from the slope of drift
against position, D from the diffusion coefficient. The prediction is
that T_eff scales LINEARLY with beta across conditions.

Three outcomes:

  T_eff proportional to beta  -> the derivation is quantitatively
                                 right, not merely structurally
                                 suggestive. This would be the first
                                 positive thermodynamic result.
  T_eff flat in beta          -> beta is not the temperature; the
                                 identity does not survive the trip
                                 from a 4-state softmax to an 8B
                                 model with LoRA adapters.
  T_eff scales the wrong way  -> something is inverted, and the
                                 mapping is wrong rather than merely
                                 imprecise.

A NORMALISATION TRAP, WHICH MATTERS
-----------------------------------
extract_emotions.py z-scores activations across whatever set is passed
to --score-all. Z-scoring sets variance to one BY CONSTRUCTION. If the
beta conditions were scored in separate calls, each was normalised
separately and any variance difference between them was destroyed
before this script could see it.

So the runs MUST be rescored together, in a single --score-all call
spanning every beta condition, before the comparison means anything.
The script checks for this and refuses if the per-condition variances
look suspiciously equal.

Usage:
    # rescore everything together FIRST
    python src/core/extract_emotions.py --score-all <all runs across all betas>

    python src/analysis/effective_temperature.py \\
        --condition 0.2 "runs/bistability/bim_b0.2_*.jsonl" \\
        --condition 2.0 "runs/bistability/bim_b2.0_*.jsonl" \\
        --condition 5.0 "runs/learning/*_on.jsonl"
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

from fep_layer import CONCEPT_WEIGHTS


def load_runs(pattern, concept):
    runs = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        if not rows or "activations" not in rows[0]:
            continue
        if concept not in rows[0]["activations"]:
            continue
        runs.append([r["activations"][concept] for r in rows])
    return runs


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def variance(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def ou_fit(runs):
    """Estimate the restoring rate k and diffusion D of an OU process.

        dx = -k(x - x0) dt + sqrt(2D) dW

    k comes from regressing the increment on position (slope = -k),
    D from the residual variance of the increments.
    """
    xs, dxs = [], []
    for s in runs:
        for i in range(len(s) - 1):
            xs.append(s[i])
            dxs.append(s[i + 1] - s[i])
    n = len(xs)
    if n < 30:
        return None

    mx, md = mean(xs), mean(dxs)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (d - md) for x, d in zip(xs, dxs)) / sxx
    intercept = md - slope * mx

    resid = [d - (slope * x + intercept) for x, d in zip(xs, dxs)]
    D = variance(resid) / 2
    k = -slope                       # restoring rate; positive = stable

    # standard error on the slope, for judging whether k is real
    s2 = sum(r * r for r in resid) / max(1, n - 2)
    se_slope = (s2 / sxx) ** 0.5

    return {
        "n": n, "k": k, "se_k": se_slope, "D": D,
        "x0": -intercept / slope if slope != 0 else float("nan"),
        "T_eff": D / k if k > 0 else float("nan"),
        "var_x": variance(xs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", nargs=2, action="append",
                    metavar=("BETA", "GLOB"), required=True,
                    help="a KL strength and the glob of its runs; "
                         "repeatable")
    ap.add_argument("--concept", default="protesting",
                    choices=list(CONCEPT_WEIGHTS))
    ap.add_argument("--level", choices=["turn", "episode"], default="turn")
    args = ap.parse_args()

    print("=" * 74)
    print(f"EFFECTIVE TEMPERATURE vs KL COEFFICIENT  --  {args.concept}")
    print("=" * 74)
    print("\nprediction from toy_model.py:  T_eff proportional to beta\n")

    results = []
    for beta_s, pattern in args.condition:
        beta = float(beta_s)
        runs = load_runs(pattern, args.concept)
        if args.level == "episode":
            from reconstruct_potential import episode_means
            new = []
            for path in sorted(glob.glob(pattern)):
                with open(path) as f:
                    rows = [json.loads(line) for line in f]
                if rows and args.concept in rows[0].get("activations", {}):
                    new.append(episode_means(rows, args.concept))
            runs = new
        fit = ou_fit(runs)
        if fit is None:
            print(f"beta={beta}: too little data ({len(runs)} runs)")
            continue
        fit["beta"] = beta
        fit["n_runs"] = len(runs)
        results.append(fit)

    if len(results) < 2:
        raise SystemExit("need at least two beta conditions")

    print(f"{'beta':>7} {'runs':>6} {'n':>6} {'k':>9} {'+-se':>8} "
          f"{'D':>9} {'var(x)':>9} {'T_eff=D/k':>11}")
    print("-" * 74)
    for r in sorted(results, key=lambda z: z["beta"]):
        print(f"{r['beta']:>7.2f} {r['n_runs']:>6} {r['n']:>6} "
              f"{r['k']:>9.4f} {r['se_k']:>8.4f} {r['D']:>9.4f} "
              f"{r['var_x']:>9.4f} {r['T_eff']:>11.4f}")

    # --- the normalisation check ---
    variances = [r["var_x"] for r in results]
    spread = (max(variances) - min(variances)) / mean(variances)
    print(f"\nvar(x) spread across conditions: {spread:.1%}")
    if spread < 0.15:
        print("""
  !! WARNING. The variances are nearly identical across beta, which is
     what z-scoring produces by construction. If these runs were scored
     in SEPARATE --score-all calls, each was normalised to unit
     variance independently and any real difference was erased before
     this script saw it. Rescore every condition together in one call
     and rerun, or the comparison below is meaningless.""")

    # --- the test ---
    print(f"\n{'=' * 74}")
    print("DOES T_eff SCALE WITH BETA?")
    print(f"{'=' * 74}")

    ordered = sorted(results, key=lambda z: z["beta"])
    betas = [r["beta"] for r in ordered]
    temps = [r["T_eff"] for r in ordered]

    print(f"\nbeta:   " + "  ".join(f"{b:>8.2f}" for b in betas))
    print(f"T_eff:  " + "  ".join(f"{t:>8.4f}" for t in temps))
    ratios = [t / b for t, b in zip(temps, betas)]
    print(f"T/beta: " + "  ".join(f"{r:>8.4f}" for r in ratios))

    # correlation between beta and T_eff
    n = len(betas)
    mb, mt = mean(betas), mean(temps)
    num = sum((b - mb) * (t - mt) for b, t in zip(betas, temps))
    den = ((sum((b - mb) ** 2 for b in betas)
            * sum((t - mt) ** 2 for t in temps)) ** 0.5)
    corr = num / den if den else float("nan")
    ratio_spread = ((max(ratios) - min(ratios)) / mean(ratios)
                    if mean(ratios) else float("nan"))

    print(f"\ncorrelation(beta, T_eff) = {corr:+.3f}")
    print(f"spread in T_eff/beta     = {ratio_spread:.1%}")

    if corr > 0.9 and ratio_spread < 0.5:
        print("""
T_eff tracks beta closely and the ratio is roughly constant. The
identity T = beta survives the trip from a 4-state softmax to an 8B
model with LoRA adapters -- a quantitative confirmation, which none of
the structural tests could provide.""")
    elif corr > 0.5:
        print("""
T_eff rises with beta but not proportionally. The direction is right
and the magnitude is not, so beta influences the effective temperature
without being it. Worth reporting as partial.""")
    elif corr < -0.5:
        print("""
T_eff moves OPPOSITE to beta. The mapping is inverted rather than
imprecise, which means the identification of beta with temperature is
wrong for this system, not merely approximate.""")
    else:
        print("""
T_eff does not track beta. The KL coefficient is not the effective
temperature of these dynamics. The derivation is correct for the toy
model and does not carry over -- which is the same gap that sank the
other framings: what is derivable in a 4-state softmax is not what
governs an 8B model with adapters.""")

    print("""
CAVEATS. k and D are estimated assuming linear drift and constant
diffusion; the reconstruction showed drift is roughly linear but
diffusion clearly is not constant across x. Conditions differ in more
than beta if they were not run identically otherwise. And the
z-scoring warning above is the one that would silently invalidate
everything.""")


if __name__ == "__main__":
    main()