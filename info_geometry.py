"""
Information geometry of the distribution shift.

WHY THIS FITS AND THE OTHERS DID NOT
------------------------------------
Nine tests for dynamical structure came back null: no temporal
structure in the child's series, no probability currents, no attractor
structure in the dyadic grid, no directed coupling. What survives is
that CONDITIONS DIFFER IN THE DISTRIBUTION of the child's affect.

Every failed framing measured structure in TRAJECTORIES. This one
measures structure in the space of DISTRIBUTIONS, which is where the
surviving effects actually live. KL divergence is the natural
divergence on that space and Fisher information its metric -- and by
the Crooks fluctuation theorem, a KL divergence IS dissipated work, so
the thermodynamic content is an identity rather than an analogy.

TWO QUESTIONS THIS CAN ANSWER THAT A DIFFERENCE OF MEANS CANNOT
---------------------------------------------------------------

1. IS MEAN-AND-VARIANCE A COMPLETE DESCRIPTION?

   A Gaussian is the maximum-entropy distribution given a mean and a
   variance. If a condition's distribution is Gaussian, those two
   numbers exhaust its information content and there is nothing else
   to find -- which would justify stopping. Departures from Gaussian
   (skew, excess kurtosis) are structure that reporting a mean throws
   away.

2. DID THE DISTRIBUTION TRANSLATE, OR CHANGE SHAPE?

   These are different physical claims. Pure translation means
   learning applies a uniform force -- every state shifted equally.
   A variance change means it altered the noise. A shape change means
   it restructured which states are accessible.

   For two Gaussians the divergence splits exactly:

       KL(P||Q) = log(s_q/s_p)  +  (s_p^2)/(2 s_q^2)  -  1/2      [dilation]
                + (mu_p - mu_q)^2 / (2 s_q^2)                     [translation]

   so the fraction of the divergence carried by the mean shift is
   computable. Every analysis in this project has assumed pure
   translation without testing it.

Usage:
    python info_geometry.py \\
        --condition low  "runs_learning/*_on.jsonl" \\
        --condition high "runs_learning/*_off.jsonl"
"""

import argparse
import glob
import json
import math

from fep_layer import CONCEPT_WEIGHTS


def load_values(patterns, concept):
    vals = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                rows = [json.loads(line) for line in f]
            for r in rows:
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
    if sd == 0:
        return m, var, 0.0, 0.0
    z = [(x - m) / sd for x in xs]
    skew = sum(v ** 3 for v in z) / n
    kurt = sum(v ** 4 for v in z) / n - 3.0        # excess
    return m, var, skew, kurt


def differential_entropy_gaussian(var):
    return 0.5 * math.log(2 * math.pi * math.e * var)


def histogram_entropy(xs, bins=20):
    """Entropy of the empirical distribution, in nats, using equal-width
    bins. Compared against the Gaussian entropy at the same variance,
    the gap is the NEGENTROPY -- how much structure the distribution has
    beyond mean and variance."""
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return 0.0
    w = (hi - lo) / bins
    counts = [0] * bins
    for x in xs:
        counts[min(int((x - lo) / w), bins - 1)] += 1
    n = len(xs)
    h = 0.0
    for c in counts:
        if c:
            p = c / n
            h -= p * math.log(p)
    return h + math.log(w)          # discrete -> differential


def kl_gaussian(mp, vp, mq, vq):
    """KL(P||Q) for Gaussians, split into its two parts."""
    dilation = 0.5 * math.log(vq / vp) + vp / (2 * vq) - 0.5
    translation = (mp - mq) ** 2 / (2 * vq)
    return dilation, translation


def fisher_rao_1d(mp, vp, mq, vq):
    """Fisher-Rao distance on the 1-D Gaussian family.

    The natural geometric distance between two distributions -- the
    length of the shortest path through distribution space, which is
    what a thermodynamic length measures.
    """
    sp, sq = vp ** 0.5, vq ** 0.5
    num = ((mp - mq) ** 2 / 2 + (sp - sq) ** 2) ** 0.5
    den = ((mp - mq) ** 2 / 2 + (sp + sq) ** 2) ** 0.5
    if den == 0:
        return 0.0
    r = num / den
    return (2 ** 0.5) * math.log((1 + r) / (1 - r)) if r < 1 else float("inf")


def bootstrap_ci(xs, stat, n_boot=2000, seed=0):
    import random
    rng = random.Random(seed)
    vals = sorted(stat([xs[rng.randrange(len(xs))] for _ in xs])
                  for _ in range(n_boot))
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", nargs=2, action="append",
                    metavar=("NAME", "GLOB"), required=True)
    ap.add_argument("--concept", default="protesting",
                    choices=list(CONCEPT_WEIGHTS))
    ap.add_argument("--bins", type=int, default=20)
    args = ap.parse_args()

    conds = []
    for name, pattern in args.condition:
        vals = load_values([pattern], args.concept)
        if len(vals) < 50:
            print(f"skipping {name}: only {len(vals)} values")
            continue
        conds.append((name, vals))
    if len(conds) < 2:
        raise SystemExit("need at least two conditions with data")

    print("=" * 76)
    print(f"INFORMATION GEOMETRY OF THE DISTRIBUTION SHIFT  --  "
          f"{args.concept}")
    print("=" * 76)

    # ---- question 1: is mean+variance complete? ----
    print("\n1. IS MEAN AND VARIANCE A COMPLETE DESCRIPTION?")
    print("   (a Gaussian is maximum-entropy given those two; any gap is")
    print("    structure that reporting a mean discards)\n")
    print(f"{'condition':<12} {'n':>6} {'mean':>9} {'sd':>8} {'skew':>8} "
          f"{'ex.kurt':>9} {'H_obs':>8} {'H_gauss':>9} {'gap':>8}")
    print("-" * 76)
    stats = {}
    for name, vals in conds:
        m, v, sk, ku = moments(vals)
        h_obs = histogram_entropy(vals, args.bins)
        h_g = differential_entropy_gaussian(v)
        stats[name] = (m, v, sk, ku, h_obs, h_g)
        print(f"{name:<12} {len(vals):>6} {m:>+9.3f} {v**0.5:>8.3f} "
              f"{sk:>+8.3f} {ku:>+9.3f} {h_obs:>8.3f} {h_g:>9.3f} "
              f"{h_g - h_obs:>8.3f}")

    print("\n   skew and excess kurtosis near 0 mean Gaussian. The")
    print("   negentropy gap (H_gauss - H_obs) is how many nats of")
    print("   structure exist beyond mean and variance.")

    # ---- question 2: translation or shape change? ----
    print("\n\n2. DID THE DISTRIBUTION TRANSLATE, OR CHANGE SHAPE?\n")
    base_name, base_vals = conds[0]
    mb, vb = stats[base_name][0], stats[base_name][1]

    print(f"   all compared against '{base_name}'\n")
    print(f"{'condition':<12} {'KL total':>10} {'translation':>13} "
          f"{'dilation':>10} {'%translation':>13} {'Fisher-Rao':>12}")
    print("-" * 76)
    for name, vals in conds[1:]:
        m, v = stats[name][0], stats[name][1]
        dil, tra = kl_gaussian(m, v, mb, vb)
        tot = dil + tra
        fr = fisher_rao_1d(m, v, mb, vb)
        pct = tra / tot if tot > 0 else float("nan")
        print(f"{name:<12} {tot:>10.4f} {tra:>13.4f} {dil:>10.4f} "
              f"{pct:>12.0%} {fr:>12.4f}")

    print("""
   KL is in nats. By the Crooks fluctuation theorem a KL divergence is
   dissipated work in units of temperature, so these are the
   thermodynamic cost of moving between the conditions.

   %translation near 100% means the distribution shifted bodily and
   nothing else changed -- learning acting as a uniform force. A large
   dilation share means the NOISE changed, which is a different claim
   and one no previous analysis could have detected.""")

    # ---- verdict ----
    print(f"\n{'=' * 76}")
    print("READING THIS")
    print(f"{'=' * 76}")

    gaps = [stats[n][5] - stats[n][4] for n, _ in conds]
    skews = [abs(stats[n][2]) for n, _ in conds]
    kurts = [abs(stats[n][3]) for n, _ in conds]
    max_gap = max(gaps)
    print(f"\nlargest negentropy gap: {max_gap:.3f} nats")
    print(f"largest |skew|: {max(skews):.3f}, "
          f"largest |excess kurtosis|: {max(kurts):.3f}")

    if max(skews) < 0.4 and max(kurts) < 0.8:
        print("""
The distributions are close to Gaussian, so mean and variance are a
complete description of them. There is no higher-order structure being
discarded by reporting a mean -- which means the analyses in this
project have not been missing anything at the distribution level, and
"the distribution shifted" really is the whole finding.""")
    else:
        print("""
The distributions depart from Gaussian, so mean and variance do NOT
exhaust them. Reporting only a mean discards real structure, and the
skew or kurtosis differences between conditions are themselves
measurable effects that no analysis here has used.""")

    if len(conds) >= 2:
        m2, v2 = stats[conds[1][0]][0], stats[conds[1][0]][1]
        dil, tra = kl_gaussian(m2, v2, mb, vb)
        if tra + dil > 0 and tra / (tra + dil) > 0.8:
            print("""
The shift is almost entirely translation. Learning moves the whole
distribution without changing its width, which is a specific and
falsifiable statement -- a uniform force on the affect coordinate --
rather than the vaguer "the mean changed".""")
        elif tra + dil > 0 and tra / (tra + dil) < 0.4:
            print("""
Most of the divergence is DILATION, not translation. The conditions
differ mainly in the WIDTH of the distribution rather than its
location. That is a different finding from the one this project has
been reporting, and worth checking against the raw values before
relying on it.""")

    print("""
CAVEATS. The Gaussian KL decomposition assumes both distributions are
Gaussian; the negentropy gap above says how badly that assumption
fails. Activations are z-scored per scoring call, so variances are
comparable only among files scored together -- if the conditions were
scored separately, the dilation term is meaningless. Bootstrap
intervals are not computed here; with a few hundred values the third
and fourth moments are noisy.""")


if __name__ == "__main__":
    main()