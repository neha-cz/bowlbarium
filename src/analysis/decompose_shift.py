"""
Three-way decomposition, and effect sizes that survive the skew.

TWO PROBLEMS WITH THE PREVIOUS ANALYSIS
---------------------------------------
info_geometry.py reported the learning/frozen shift as 89%
translation. That figure is 89% of the GAUSSIAN-APPROXIMATED
divergence, and the Gaussian formula only knows about means and
variances. It is structurally blind to shape.

Shape differs substantially between those conditions:

    learning   skew +1.330,  excess kurtosis +1.780
    frozen     skew +0.950,  excess kurtosis +0.164

An order of magnitude apart in tail weight. So the Gaussian
decomposition is missing whatever the divergence carries in the third
and fourth moments, and "89% translation" cannot be trusted as stated.

The second problem is downstream of the same skew. Every effect size
in this project -- Result 1's 0.212, Result 2's 0.80 -- is a DIFFERENCE
OF MEANS. On a distribution with skew above 1, the mean sits well above
the median and is dragged by tail events, so a mean shift can reflect a
change in how often extreme turns occur rather than a change in
typical behaviour. That has never been checked.

WHAT THIS DOES
--------------
1. Estimates the total divergence NON-PARAMETRICALLY, then splits it:

       translation   mean shift, at matched variance
       dilation      variance change
       shape         everything the Gaussian form cannot represent
                     = KL_nonparametric - KL_gaussian

2. Recomputes the effect with distribution-free statistics:

       median difference       location without tail leverage
       Hodges-Lehmann          median of all pairwise differences
       Cliff's delta           P(x>y) - P(y>x); assumes nothing about
                               shape at all

   If these agree with the mean difference, the skew is harmless and
   the reported effects stand. If they shrink, the effects were partly
   carried by tails.

Usage:
    python src/analysis/decompose_shift.py \\
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


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def moments(xs):
    n = len(xs)
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    sd = var ** 0.5
    z = [(x - m) / sd for x in xs]
    return m, var, sum(v ** 3 for v in z) / n, sum(v ** 4 for v in z) / n - 3


def kl_gaussian_parts(mp, vp, mq, vq):
    dilation = 0.5 * math.log(vq / vp) + vp / (2 * vq) - 0.5
    translation = (mp - mq) ** 2 / (2 * vq)
    return translation, dilation


def kl_histogram(p_vals, q_vals, bins=30, alpha=0.5):
    """Non-parametric KL(P||Q) from shared-edge histograms.

    Laplace smoothing (alpha) keeps empty Q bins from sending the
    divergence to infinity. Both samples are binned on the SAME edges,
    which is what makes the two histograms comparable.
    """
    lo = min(min(p_vals), min(q_vals))
    hi = max(max(p_vals), max(q_vals))
    if hi == lo:
        return 0.0
    w = (hi - lo) / bins

    def hist(vals):
        c = [alpha] * bins
        for v in vals:
            c[min(int((v - lo) / w), bins - 1)] += 1
        t = sum(c)
        return [x / t for x in c]

    P, Q = hist(p_vals), hist(q_vals)
    return sum(p * math.log(p / q) for p, q in zip(P, Q) if p > 0)


def hodges_lehmann(xs, ys, cap=400, seed=0):
    """Median of pairwise differences. Subsampled when the full
    product would be large."""
    rng = random.Random(seed)
    a = xs if len(xs) <= cap else [rng.choice(xs) for _ in range(cap)]
    b = ys if len(ys) <= cap else [rng.choice(ys) for _ in range(cap)]
    return median([p - q for p in a for q in b])


def cliffs_delta(xs, ys, cap=400, seed=0):
    """P(x>y) - P(y>x). Distribution-free: no assumption about shape,
    variance, or the presence of tails."""
    rng = random.Random(seed)
    a = xs if len(xs) <= cap else [rng.choice(xs) for _ in range(cap)]
    b = ys if len(ys) <= cap else [rng.choice(ys) for _ in range(cap)]
    gt = lt = 0
    for p in a:
        for q in b:
            if p > q:
                gt += 1
            elif p < q:
                lt += 1
    n = len(a) * len(b)
    return (gt - lt) / n


def boot_ci(xs, ys, stat, n_boot=600, seed=1):
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        a = [xs[rng.randrange(len(xs))] for _ in range(len(xs))]
        b = [ys[rng.randrange(len(ys))] for _ in range(len(ys))]
        vals.append(stat(a, b))
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs=2, metavar=("NAME", "GLOB"), required=True)
    ap.add_argument("--b", nargs=2, metavar=("NAME", "GLOB"), required=True)
    ap.add_argument("--concept", default="protesting",
                    choices=list(CONCEPT_WEIGHTS))
    ap.add_argument("--bins", type=int, default=30)
    args = ap.parse_args()

    na, pa = args.a
    nb, pb = args.b
    A = load_values(pa, args.concept)
    B = load_values(pb, args.concept)
    if len(A) < 100 or len(B) < 100:
        raise SystemExit(f"too little data: {len(A)} and {len(B)} values")

    ma, va, ska, kua = moments(A)
    mb, vb, skb, kub = moments(B)

    print("=" * 76)
    print(f"DECOMPOSING THE SHIFT  --  {args.concept}")
    print("=" * 76)
    print(f"\n{'condition':<12} {'n':>6} {'mean':>9} {'median':>9} "
          f"{'sd':>8} {'skew':>8} {'ex.kurt':>9}")
    print("-" * 76)
    print(f"{na:<12} {len(A):>6} {ma:>+9.3f} {median(A):>+9.3f} "
          f"{va**0.5:>8.3f} {ska:>+8.3f} {kua:>+9.3f}")
    print(f"{nb:<12} {len(B):>6} {mb:>+9.3f} {median(B):>+9.3f} "
          f"{vb**0.5:>8.3f} {skb:>+8.3f} {kub:>+9.3f}")

    # ---------- 1. three-way decomposition ----------
    tra, dil = kl_gaussian_parts(ma, va, mb, vb)
    kl_gauss = tra + dil
    kl_np = kl_histogram(A, B, args.bins)
    shape = kl_np - kl_gauss

    print(f"\n{'=' * 76}")
    print("1. THREE-WAY DECOMPOSITION")
    print(f"{'=' * 76}")
    print(f"\n{'component':<28} {'nats':>10} {'share':>10}")
    print("-" * 76)
    total = kl_np if kl_np > 0 else float("nan")
    for label, v in (("translation (mean)", tra),
                     ("dilation (variance)", dil),
                     ("shape (higher moments)", shape)):
        print(f"{label:<28} {v:>10.4f} {v / total:>9.0%}")
    print("-" * 76)
    print(f"{'total (non-parametric)':<28} {kl_np:>10.4f}")
    print(f"{'  of which Gaussian form':<28} {kl_gauss:>10.4f}")

    if shape < 0:
        print("""
  The shape term is NEGATIVE, which means the histogram estimate came
  out below the Gaussian one. That is an estimator artifact -- binned
  KL is biased at finite samples -- and it means the shape
  contribution is too small to resolve here rather than genuinely
  negative.""")
    elif shape / total > 0.3:
        print(f"""
  Shape carries {shape / total:.0%} of the divergence -- structure the
  Gaussian decomposition cannot represent, and which the earlier "89%
  translation" figure silently omitted. The conditions differ in their
  TAILS as much as in their location.""")
    else:
        print(f"""
  Shape carries only {shape / total:.0%}. Despite both distributions
  being visibly skewed, the DIFFERENCE between them is mostly
  location and scale, so the Gaussian decomposition was adequate after
  all.""")

    # ---------- 2. robust effect sizes ----------
    print(f"\n{'=' * 76}")
    print("2. DOES THE EFFECT SURVIVE THE SKEW?")
    print(f"{'=' * 76}")
    print(f"\n(both distributions are right-skewed, so the mean sits above")
    print(f" the median and is dragged by the upper tail)\n")

    d_mean = ma - mb
    d_med = median(A) - median(B)
    hl = hodges_lehmann(A, B)
    cd = cliffs_delta(A, B)

    lo_m, hi_m = boot_ci(A, B, lambda a, b: mean(a) - mean(b))
    lo_d, hi_d = boot_ci(A, B, lambda a, b: median(a) - median(b))
    lo_c, hi_c = boot_ci(A, B, lambda a, b: cliffs_delta(a, b, cap=200),
                         n_boot=200)

    print(f"{'statistic':<26} {'value':>10} {'95% CI':>22}")
    print("-" * 76)
    print(f"{'difference of means':<26} {d_mean:>+10.3f} "
          f"{f'[{lo_m:+.3f}, {hi_m:+.3f}]':>22}")
    print(f"{'difference of medians':<26} {d_med:>+10.3f} "
          f"{f'[{lo_d:+.3f}, {hi_d:+.3f}]':>22}")
    print(f"{'Hodges-Lehmann':<26} {hl:>+10.3f}")
    print(f"{'Cliffs delta':<26} {cd:>+10.3f} "
          f"{f'[{lo_c:+.3f}, {hi_c:+.3f}]':>22}")

    print("""
  Cliff's delta is P(a>b) - P(b>a): 0 is no effect, +/-0.15 small,
  +/-0.33 medium, +/-0.47 large. It assumes nothing about shape.""")

    ratio = abs(d_med / d_mean) if d_mean else float("nan")
    print(f"\nmedian shift is {ratio:.0%} of the mean shift")
    if ratio > 0.7:
        print("""
The effect survives. Location moved, not just the tail, so the
mean-based effect sizes reported throughout this project are not
artifacts of the skew.""")
    elif ratio > 0.3:
        print("""
The median shift is materially smaller than the mean shift. Part of
the reported effect is carried by the upper tail -- a change in how
often extreme turns occur -- rather than by typical behaviour. Both are
real, but they are different claims and the writeup should say which.""")
    else:
        print("""
The median barely moves while the mean does. The effect is almost
entirely a TAIL phenomenon: extreme turns became rarer or more common,
while typical behaviour is unchanged. Every mean-difference effect
size in this project should be restated on that basis.""")

    if abs(cd) < 0.15:
        print(f"""
Cliff's delta of {cd:+.3f} is below the conventional threshold for a
small effect. Whatever the means say, a randomly chosen turn from one
condition is barely more likely to exceed one from the other.""")


if __name__ == "__main__":
    main()