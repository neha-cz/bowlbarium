"""
Thermodynamic structure in parameter space.

WHICH PREVIOUS FRAMINGS PORT, AND WHICH DO NOT
----------------------------------------------
Most do not. Entropy production, Kramers-Moyal reconstruction,
coupling, state space grids and hysteresis all require TRAJECTORIES,
and the adapters are static endpoints -- one point in parameter space
per run. Saving adapters mid-run would fix that; it was not done.

What does port is the DISTRIBUTIONAL analysis. Each adapter provides
6.8M weight changes, which is a distribution that can be characterised
exactly as the activation distributions were.

THE DIAGNOSTIC QUESTION
-----------------------
Heavy-tailed gradient noise in SGD is an established finding: the
noise often follows an alpha-stable rather than a Gaussian law, making
the dynamics Levy motion rather than Brownian.

This matters beyond curiosity. The Langevin picture -- and therefore
the whole notion of a temperature in parameter space -- ASSUMES
Gaussian noise. Under alpha-stable noise with alpha < 2 the variance
is infinite, the fluctuation-dissipation relation does not hold, and
there is no temperature to measure.

fdt_temperature.py already found no measurable temperature in
activation space: R^2 = 0.469 on linearity, and a susceptibility that
flipped sign between adjacent perturbation pairs. If the weight
updates turn out heavy-tailed, that is not a twelfth independent null
-- it is an EXPLANATION for the previous ones.

WHAT IS MEASURED
----------------
1. TAIL INDEX of the weight-change distribution, by the Hill
   estimator. alpha >= 2 is Gaussian-like (finite variance, Langevin
   applies). alpha < 2 is alpha-stable (infinite variance, no
   temperature).

2. SIGNAL / NOISE DECOMPOSITION. All adapters share a direction
   (cosine 0.68 on effective updates). Projecting each displacement
   onto the shared axis splits it:

       dtheta = c * u_shared  +  residual

   The residual is what differs between runs -- the diffusion. Its
   scale is the closest thing to a temperature available here.

3. THE SGD-LANGEVIN PREDICTION. In that picture the stationary
   temperature scales as learning-rate over batch-size. With
   lr = 1e-6 and batch = 2 known, the predicted scale can be compared
   against the measured residual, which is a quantitative check rather
   than another framing.

Usage:
    python parameter_thermo.py \\
        --base adapters_child_r32 \\
        --adapter lowA  runs_hysteresis/LH_adapter_A \\
        --adapter lowC  runs_timescale/T850_LH_adapter_A \\
        --adapter highA runs_hysteresis/HH_adapter_A \\
        --adapter highB runs_timescale/T850_HH_adapter_A \\
        --lr 1e-6 --batch 2
"""

import argparse
import math
import os


def load_adapter(path):
    import numpy as np
    from safetensors.numpy import load_file
    f = path if path.endswith(".safetensors") else os.path.join(
        path, "adapters.safetensors")
    if not os.path.exists(f):
        raise SystemExit(f"not found: {f}")
    raw = load_file(f)
    return {k: np.asarray(v, dtype=np.float64).ravel() for k, v in raw.items()}


def displacement_vector(base, other):
    import numpy as np
    keys = sorted(k for k in set(base) & set(other)
                  if base[k].shape == other[k].shape)
    return np.concatenate([other[k] - base[k] for k in keys])


def hill_estimator(x, k_frac=0.02):
    """Tail index alpha from the largest |x| values.

    alpha = k / sum(log(x_(i)/x_(k+1))) over the top k order
    statistics. Small alpha means heavy tails:

        alpha >= 2   finite variance, Gaussian-like
        alpha < 2    alpha-stable, infinite variance
        alpha ~ 1    Cauchy-like
    """
    import numpy as np
    a = np.sort(np.abs(x))[::-1]
    a = a[a > 0]
    k = max(50, int(len(a) * k_frac))
    k = min(k, len(a) - 2)
    if k < 20:
        return float("nan")
    top = a[:k]
    thresh = a[k]
    if thresh <= 0:
        return float("nan")
    s = np.sum(np.log(top / thresh))
    return float(k / s) if s > 0 else float("nan")


def moments(x):
    import numpy as np
    m = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    z = (x - m) / sd
    return m, sd, float(np.mean(z ** 3)), float(np.mean(z ** 4) - 3)


def gaussian_tail_check(x):
    """Fraction of mass beyond 3 and 5 sd. A Gaussian puts 0.27% beyond
    3 sd and 6e-5% beyond 5. Large excesses mean heavy tails, and this
    needs no fitting at all."""
    import numpy as np
    sd = np.std(x, ddof=1)
    m = np.mean(x)
    z = np.abs((x - m) / sd)
    return float(np.mean(z > 3)), float(np.mean(z > 5)), float(np.max(z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", nargs=2, action="append",
                    metavar=("NAME", "PATH"), required=True)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    import numpy as np

    print("=" * 78)
    print("PARAMETER-SPACE THERMODYNAMICS")
    print("=" * 78)

    base = load_adapter(args.base)
    disps = {}
    for name, path in args.adapter:
        disps[name] = displacement_vector(base, load_adapter(path))
    names = list(disps)
    n_par = len(disps[names[0]])
    print(f"\n{len(names)} adapters, {n_par:,} parameters each")

    # ---------- 1. is the noise Gaussian? ----------
    print(f"\n{'=' * 78}")
    print("1. IS THE WEIGHT-CHANGE DISTRIBUTION GAUSSIAN?")
    print(f"{'=' * 78}")
    print("""
The Langevin picture -- and any temperature in parameter space --
assumes Gaussian noise. Heavy tails mean alpha-stable dynamics, where
the variance is infinite and no temperature exists.""")
    print(f"\n{'adapter':<10} {'sd':>10} {'skew':>9} {'ex.kurt':>10} "
          f"{'>3sd':>9} {'>5sd':>9} {'max z':>8} {'Hill a':>8}")
    print("-" * 78)
    for n in names:
        v = disps[n]
        _, sd, sk, ku = moments(v)
        p3, p5, mz = gaussian_tail_check(v)
        a = hill_estimator(v)
        print(f"{n:<10} {sd:>10.2e} {sk:>+9.3f} {ku:>+10.2f} "
              f"{p3:>8.3%} {p5:>8.4%} {mz:>8.1f} {a:>8.2f}")
    print("""
  Gaussian reference: >3sd = 0.270%, >5sd = 0.00006%, excess
  kurtosis = 0. Hill alpha >= 2 means finite variance.""")

    kurts = [moments(disps[n])[3] for n in names]
    alphas = [hill_estimator(disps[n]) for n in names]
    heavy = np.nanmean(kurts) > 1.0

    # ---------- 2. signal / noise ----------
    print(f"\n{'=' * 78}")
    print("2. SHARED DIRECTION vs RESIDUAL")
    print(f"{'=' * 78}")
    M = np.stack([disps[n] / np.linalg.norm(disps[n]) for n in names])
    u = M.mean(axis=0)
    u /= np.linalg.norm(u)

    print(f"\n{'adapter':<10} {'||dtheta||':>12} {'proj on u':>12} "
          f"{'residual':>11} {'signal frac':>13}")
    print("-" * 78)
    res_norms = []
    for n in names:
        v = disps[n]
        c = float(np.dot(v, u))
        r = v - c * u
        rn = float(np.linalg.norm(r))
        res_norms.append(rn)
        vn = float(np.linalg.norm(v))
        print(f"{n:<10} {vn:>12.4f} {c:>12.4f} {rn:>11.4f} "
              f"{(c / vn) ** 2:>12.1%}")

    mean_sig = np.mean([(float(np.dot(disps[n], u))
                         / np.linalg.norm(disps[n])) ** 2 for n in names])
    print(f"""
  The shared axis carries {mean_sig:.1%} of each update on average; the
  rest is run-specific. A high signal fraction means the loop has one
  dominant learning direction and the residual is diffusion around it.""")

    # residual isotropy: are the leftover components exchangeable?
    r0 = disps[names[0]] - float(np.dot(disps[names[0]], u)) * u
    chunks = np.array_split(r0, 20)
    cv = np.std([np.std(c) for c in chunks]) / np.mean(
        [np.std(c) for c in chunks])
    print(f"  residual isotropy: sd varies {cv:.1%} across 20 blocks "
          f"of parameters")
    print("   (small means the diffusion is roughly uniform over "
          "parameters,\n    which is what the Langevin picture assumes)")

    # ---------- 3. SGD-Langevin scale ----------
    print(f"\n{'=' * 78}")
    print("3. THE SGD-LANGEVIN PREDICTION")
    print(f"{'=' * 78}")
    ratio = args.lr / args.batch
    mean_res = float(np.mean(res_norms))
    per_par = mean_res / math.sqrt(n_par)
    print(f"""
  learning rate {args.lr:g}, batch {args.batch}  ->  lr/batch = {ratio:.2e}
  mean residual norm {mean_res:.4f} over {n_par:,} parameters
  per-parameter residual scale {per_par:.3e}
  ratio to lr/batch: {per_par / ratio:.2f}""")
    print("""
  In the Langevin picture the stationary scale goes as lr/batch. A
  single (lr, batch) setting cannot test that -- it needs runs at
  several values, and the number above is a reference point rather
  than a test. What it does say is the ORDER OF MAGNITUDE the
  diffusion sits at.""")

    # ---------- verdict ----------
    print(f"\n{'=' * 78}")
    print("READING THIS")
    print(f"{'=' * 78}")
    print(f"""
mean excess kurtosis {np.mean(kurts):+.2f}, mean Hill alpha "
{np.nanmean(alphas):.2f}""")

    if heavy and np.nanmean(alphas) < 2:
        print("""
The weight changes are HEAVY-TAILED, with a tail index below 2. Under
alpha-stable noise the variance is infinite, the
fluctuation-dissipation relation does not hold, and there is no
temperature in parameter space either.

That is not a twelfth independent null -- it EXPLAINS the earlier
ones. fdt_temperature.py could not find a temperature because the
noise is not Gaussian, so the quantity it was estimating does not
exist. The Langevin framework, and every energy expressed in units of
kT in this project, rests on an assumption the data violates.""")
    elif heavy:
        print("""
The distribution is leptokurtic but the tail index is at or above 2,
so the variance is finite and the Langevin picture is not ruled out.
The heavy centre with finite tails is more consistent with a mixture
of scales across layers than with alpha-stable noise.""")
    else:
        print("""
The weight changes are approximately Gaussian. The Langevin picture is
not violated at the level of the noise distribution, so the absence of
a measurable temperature in activation space has some other cause --
most likely that activations are several transformations removed from
the parameters and the mapping is not linear.""")

    print("""
CAVEATS. The Hill estimator is sensitive to how many order statistics
are used; 2% of the sample is a common choice and other values shift
alpha somewhat. LoRA parameters are not exchangeable -- A and B
factors have different scales and roles, so pooling them into one
distribution mixes populations, which inflates kurtosis on its own.
Splitting by tensor would be the cleaner test. And four adapters give
a very rough shared direction; with more runs the signal/noise split
would be better determined.""")


if __name__ == "__main__":
    main()