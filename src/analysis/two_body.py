"""
Two-body coupling: information flow in both directions.

WHY THIS IS A DIFFERENT OBJECT
------------------------------
coupling.py found no directed mother-to-child influence and returned
null. But it was run on data where the mother was not part of a
dynamical system at all: her availability was drawn independently each
turn from a fixed distribution, and her weights never updated. She was
an external field, not a coupled degree of freedom.

A one-body system in a stationary bath cannot show two-body structure,
so that null was partly guaranteed by the setup.

With learning_loop.py --feedback c, availability depends on the
child's recent bidding:

    r_eff = clip(r0 + c * bid_rate, 0, 1)

Now the mother's state is a function of the child's, the child responds
to the mother, and there is an actual coupled system to measure.

THE POSITIVE CONTROL THAT MAKES THIS TRUSTWORTHY
------------------------------------------------
With feedback on, the child -> mother channel is TRUE BY CONSTRUCTION:
r_eff is computed from the child's bidding, so information must flow
that way. Any estimator that fails to detect it is broken.

That gives a built-in calibration the previous coupling analysis
lacked. The real question is the OTHER direction -- whether the
mother's availability predicts the child's next affective state beyond
what the child's own state predicts -- and it can only be believed if
the known direction registers first.

WHAT IS MEASURED
----------------
    child -> mother   bid_rate(t) against r_eff(t+1)   [control]
    mother -> child   r_eff(t) against protest(t+1),
                      partialled on protest(t)          [the question]

r_eff is used rather than keyword-coded mother text. It is the
mother's actual availability state, logged per turn, so this avoids
the classifier whose misclassification rate was already a known
limitation.

Nulls use circular shifts, which preserve each series' own
autocorrelation while breaking alignment -- a full shuffle was
miscalibrated and gave 5% false positives on independent data.

Usage:
    # generate data with feedback, then score it
    python src/core/learning_loop.py events.jsonl --out fb.jsonl --feedback 1.0
    python src/core/extract_emotions.py --score-all fb.jsonl other.jsonl

    python src/analysis/two_body.py --runs "runs/feedback/*.jsonl"
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
import random

from fep_layer import CONCEPT_WEIGHTS


def load(patterns, concept):
    runs = []
    missing_feedback = 0
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                rows = [json.loads(line) for line in f]
            if not rows or "activations" not in rows[0]:
                continue
            if rows[0].get("r_eff") is None:
                missing_feedback += 1
                continue
            runs.append({
                "name": path.split("/")[-1],
                "r_eff": [r["r_eff"] for r in rows],
                "bid": [r.get("bid_rate", 0.0) for r in rows],
                "child": [r["activations"][concept] for r in rows],
            })
    return runs, missing_feedback


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def corr(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = mean(xs), mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def partial(xs, ys, zs):
    """Partial correlation, with the correlations clamped to [-1, 1].

    Without clamping, floating-point error can push a correlation just
    past 1, making (1 - r^2) negative and its square root complex --
    which raised a TypeError at lag 10.
    """
    def clamp(v):
        return max(-1.0, min(1.0, v))
    rxy, rxz, ryz = (clamp(corr(xs, ys)), clamp(corr(xs, zs)),
                     clamp(corr(ys, zs)))
    inner = (1 - rxz ** 2) * (1 - ryz ** 2)
    if inner <= 1e-12:
        return 0.0
    return (rxy - rxz * ryz) / (inner ** 0.5)


def stats(runs, lag=1):
    """Lagged pairs in both directions, pooled across runs.

    LAG MATTERS HERE. r_eff is a running average over a 20-turn
    window, so its autocorrelation at lag 1 was +0.975 in the first
    run of this analysis -- meaning r_eff(t+1) is almost entirely
    determined by r_eff(t), and a partial correlation conditioning on
    r_eff(t) has nothing left to explain. The control returned exactly
    +0.000 for that reason, not because the channel was absent.

    Testing at a lag comparable to the averaging window gives the
    mother variable room to have actually moved.
    """
    c2m_x, c2m_y, c2m_z = [], [], []      # bid(t) -> r_eff(t+1)
    m2c_x, m2c_y, m2c_z = [], [], []      # r_eff(t) -> child(t+1)
    auto_a, auto_b = [], []
    for r in runs:
        n = len(r["child"])
        for t in range(n - lag):
            c2m_x.append(r["bid"][t])
            c2m_y.append(r["r_eff"][t + lag])
            c2m_z.append(r["r_eff"][t])
            m2c_x.append(r["r_eff"][t])
            m2c_y.append(r["child"][t + lag])
            m2c_z.append(r["child"][t])
            auto_a.append(r["child"][t])
            auto_b.append(r["child"][t + lag])
    return {
        "c2m": corr(c2m_x, c2m_y),
        "c2m_partial": partial(c2m_x, c2m_y, c2m_z),
        "m2c": corr(m2c_x, m2c_y),
        "m2c_partial": partial(m2c_x, m2c_y, m2c_z),
        "child_auto": corr(auto_a, auto_b),
        "mother_auto": corr([r["r_eff"][t] for r in runs
                             for t in range(len(r["r_eff"]) - lag)],
                            [r["r_eff"][t + lag] for r in runs
                             for t in range(len(r["r_eff"]) - lag)]),
        "n": len(m2c_x),
    }


def shifted(runs, rng, which):
    """Circular-shift one agent's series, leaving the other intact."""
    out = []
    for r in runs:
        n = len(r["child"])
        k = rng.randint(max(2, n // 10), n - max(2, n // 10)) if n > 6 else 1
        s = dict(r)
        if which == "mother":
            s["r_eff"] = r["r_eff"][k:] + r["r_eff"][:k]
        else:
            s["child"] = r["child"][k:] + r["child"][:k]
            s["bid"] = r["bid"][k:] + r["bid"][:k]
        out.append(s)
    return out


def test(runs, key, which, n_perm, rng, lag=1):
    obs = stats(runs, lag)[key]
    null = [stats(shifted(runs, rng, which), lag)[key]
            for _ in range(n_perm)]
    vals = sorted(abs(v) for v in null)
    mu = mean(vals)
    sd = (mean([(v - mu) ** 2 for v in vals])) ** 0.5
    p = (sum(1 for v in vals if v >= abs(obs)) + 1) / (len(vals) + 1)
    z = (abs(obs) - mu) / sd if sd > 0 else float("nan")
    return obs, mu, sd, z, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--concept", default="protesting",
                    choices=list(CONCEPT_WEIGHTS))
    ap.add_argument("--permutations", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lags", nargs="+", type=int, default=[1, 5, 10, 20],
                    help="r_eff averages over a 20-turn window, so lag 1 "
                         "tests a variable that has barely moved; sweeping "
                         "up to the window length is what gives it room")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    runs, skipped = load(args.runs, args.concept)
    if not runs:
        raise SystemExit(
            "No runs with r_eff found"
            + (f" ({skipped} files had activations but no r_eff)"
               if skipped else "")
            + ".\nThis analysis needs data generated with "
              "learning_loop.py --feedback c, and scored with a version "
              "of extract_emotions.py that preserves r_eff.")

    print("=" * 76)
    print(f"TWO-BODY COUPLING  --  {args.concept}")
    print("=" * 76)
    print(f"\n{len(runs)} run(s) with feedback"
          + (f", {skipped} skipped (no r_eff)" if skipped else ""))
    for r in runs:
        print(f"  {r['name']}: {len(r['child'])} turns, "
              f"r_eff {min(r['r_eff']):.2f}-{max(r['r_eff']):.2f}")

    print(f"\n{'=' * 76}")
    print("AUTOCORRELATION BY LAG")
    print(f"{'=' * 76}")
    print(f"\n{'lag':>6} {'child':>10} {'mother':>10} {'pairs':>8}")
    print("-" * 76)
    for L in args.lags:
        s = stats(runs, L)
        print(f"{L:>6} {s['child_auto']:>+10.3f} {s['mother_auto']:>+10.3f} "
              f"{s['n']:>8}")
    print("""
  The mother's autocorrelation is the key number. Where it approaches
  1, r_eff(t+lag) is already determined by r_eff(t) and the partial
  correlation has no variance left to work with -- at lag 1 it was
  +0.975 and the control returned exactly +0.000.""")

    print(f"\n{'=' * 76}")
    print("DIRECTION 1 — child to mother  [POSITIVE CONTROL]")
    print(f"{'=' * 76}")
    print(f"\n{'lag':>6} {'partial':>10} {'null':>10} {'z':>8} {'p':>8}")
    print("-" * 76)
    control_ok = False
    for L in args.lags:
        o, mu, sd, z, pv = test(runs, "c2m_partial", "child",
                                args.permutations, rng, L)
        control_ok = control_ok or pv < 0.05
        print(f"{L:>6} {o:>+10.3f} {mu:>10.3f} {z:>+8.2f} {pv:>8.3f}"
              + ("  <-- detected" if pv < 0.05 else ""))

    print(f"\n{'=' * 76}")
    print("DIRECTION 2 — mother to child  [THE QUESTION]")
    print(f"{'=' * 76}")
    print(f"\n{'lag':>6} {'partial':>10} {'null':>10} {'z':>8} {'p':>8}")
    print("-" * 76)
    best = None
    for L in args.lags:
        o2, mu2, sd2, z2, p2 = test(runs, "m2c_partial", "mother",
                                    args.permutations, rng, L)
        if best is None or p2 < best[1]:
            best = (o2, p2, L)
        print(f"{L:>6} {o2:>+10.3f} {mu2:>10.3f} {z2:>+8.2f} {p2:>8.3f}"
              + ("  <-- detected" if p2 < 0.05 else ""))

    o2, p2, best_lag = best
    print(f"\n{'=' * 76}")
    print("READING THIS")
    print(f"{'=' * 76}")

    if p2 < 0.05 and control_ok:
        print(f"""
Both directions register, the mother-to-child channel at lag
{best_lag} ({o2:+.3f}, p = {p2:.3f}). The dyad is genuinely coupled --
and the lag at which it appears says the influence acts over several
exchanges rather than turn to turn, which is why the lag-1 test found
nothing.""")
    elif p2 < 0.05:
        print(f"""
Mother-to-child registers at lag {best_lag} ({o2:+.3f}, p = {p2:.3f})
while the control does not. Given that the child-to-mother channel
exists by construction, a failed control means the partial correlation
cannot see it -- r_eff is too autocorrelated -- rather than that it is
absent. Read direction 2 on its own terms.""")
    elif control_ok:
        print(f"""
The control fires but mother-to-child does not, at any lag tested. The
child drives the mother's availability -- by construction -- and the
mother's availability does not measurably drive the child back, even
allowing several exchanges for the influence to act.

An asymmetric coupling is a substantive finding: whatever the
caregiver manipulation does, it does not act through a channel
detectable at these timescales.""")
    else:
        print(f"""
Nothing registers at any lag, including a channel that exists by
construction. That means the test cannot resolve coupling in this
data, not that coupling is absent -- most likely because r_eff is
smoothed over a 20-turn window and moves too slowly for {len(runs)}
runs of 180 turns to constrain. Shortening the feedback window
(ResponsivenessFeedback(window=5)) would make availability respond
fast enough for this design to work.""")

    print("""
CAVEATS. r_eff moves through a running average of bidding, so it
changes slowly and its autocorrelation is high -- circular shifts
preserve that, but the effective number of independent samples is far
below the pair count. Partial correlation removes only linear
dependence on the child's previous state. And one turn may be the
wrong lag if influence acts over several exchanges.""")


if __name__ == "__main__":
    main()