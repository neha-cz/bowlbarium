"""
Testing the double-well prediction: is there a critical KL strength?

THE PREDICTION
--------------
double_well.py derives that with behaviour-dependent caregiver
availability -- r_eff = r0 + c * bid_rate -- the free energy gains a
quadratic self-interaction and becomes bistable below

    T_c = c / 2

Since toy_model.py established T = beta (the KL coefficient), this
predicts a CRITICAL KL STRENGTH. Below it, identical conditions should
support two stable outcomes. Above it, one.

WHY THIS IS A REAL TEST
-----------------------
Until now the mechanism was absent from the system: the event
generator drew availability independently of the child's behaviour, so
c = 0, T_c = 0, and nothing the double-well model says could apply.
learning_loop.py --feedback c supplies it.

The signature is BIMODALITY. Run many seeds at fixed r0, c and beta,
and look at the distribution of final protest levels:

    bimodal   -> two basins; the same conditions support a bidding
                 child and an accommodating child, which is what a
                 discrete attachment style would mean mechanically
    unimodal  -> one basin; the model is wrong, or beta is above T_c

The test is the CONTRAST across beta, not the shape at any one value.
Bimodal at low beta and unimodal at high beta, with the switch near
c/2, is the prediction. Unimodal everywhere falsifies it. Bimodal
everywhere means something other than the predicted mechanism is
producing the split.

A CHEAPER PRELIMINARY
---------------------
Bistability also implies sensitivity to initial conditions: two
children started from different states under identical conditions
should stay apart if bistable and converge if not. --mode divergence
runs that with two seeds per beta instead of eight, and is worth
running first.

COST
----
--mode bimodal: n_seeds x len(betas) runs. At ~20 min per 15-episode
run, 8 seeds x 2 betas is about 5 hours.
--mode divergence: 2 runs per beta, so under an hour for two betas.

Usage:
    python bistability_test.py --mode divergence --c 1.0
    python bistability_test.py --mode bimodal --c 1.0 --seeds 8
"""

import argparse
import json
import os
import subprocess
import sys


def run(cmd):
    print(f"    $ {' '.join(str(x) for x in cmd)}")
    r = subprocess.run([str(x) for x in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:])
        print(r.stderr[-1500:])
        raise SystemExit("command failed")
    return r.stdout


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def tail_protest(path, frac=0.3):
    rows = load(path)
    q = max(5, int(len(rows) * frac))
    return mean([r["activations"]["protesting"] for r in rows[-q:]])


def bimodality_coefficient(xs):
    """Sarle's bimodality coefficient: (skew^2 + 1) / kurtosis.

    Above 5/9 = 0.555 suggests bimodality; a normal distribution gives
    0.333. Crude on small samples, but it is a standard summary and
    beats eyeballing.
    """
    n = len(xs)
    if n < 4:
        return float("nan")
    m = mean(xs)
    s = stdev(xs)
    if s == 0:
        return float("nan")
    z = [(x - m) / s for x in xs]
    skew = sum(v ** 3 for v in z) / n
    kurt = sum(v ** 4 for v in z) / n
    if kurt == 0:
        return float("nan")
    return (skew ** 2 + 1) / kurt


def gap_statistic(xs):
    """Largest gap between consecutive sorted values, as a fraction of
    the total range. A clean split shows one dominant gap."""
    if len(xs) < 3:
        return 0.0
    s = sorted(xs)
    rng = s[-1] - s[0]
    if rng == 0:
        return 0.0
    gaps = [s[i + 1] - s[i] for i in range(len(s) - 1)]
    return max(gaps) / rng


def make_events(seed, episodes, workdir, r0):
    prefix = os.path.join(workdir, f"B{seed}_events")
    # Availability in the file is overridden by the feedback loop, so
    # the reliability here only shapes which EVENTS occur.
    run([sys.executable, "event_generator.py", "--seed", seed,
         "--episodes", episodes, "--prefix", prefix, "--quiet",
         "--high", r0, "--low", r0])
    return f"{prefix}_high_reliability.jsonl"


def one_run(events, out, seed, args, beta, start_bias=None):
    cmd = [sys.executable, "learning_loop.py", events, "--out", out,
           "--episodes", args.episodes, "--threshold", args.threshold,
           "--batch-size", args.batch_size, "--seed", seed,
           "--feedback", args.c, "--base-reliability", args.r0]
    if start_bias is not None:
        cmd += ["--child-adapter", start_bias]
    env = dict(os.environ, MARV_KL_BETA=str(beta))
    print(f"    $ KL_BETA={beta} " + " ".join(str(x) for x in cmd))
    r = subprocess.run([str(x) for x in cmd], capture_output=True,
                       text=True, env=env)
    if r.returncode != 0:
        print(r.stdout[-1500:])
        print(r.stderr[-1500:])
        raise SystemExit("command failed")
    return out


def mode_divergence(args):
    """Two children, same conditions, different starting adapters.
    Bistable -> they stay apart. Single-well -> they converge."""
    print("=" * 74)
    print("DIVERGENCE TEST  (cheap preliminary)")
    print("=" * 74)
    print(f"""
Two children under identical conditions (c={args.c}, r0={args.r0}),
started from different adapters. If the potential is bistable they
settle into different basins and stay apart; if it has one minimum
they converge regardless of where they began.

predicted T_c = c/2 = {args.c / 2:.3f}; compare against each beta below.""")

    os.makedirs(args.workdir, exist_ok=True)
    events = make_events(args.seed, args.episodes, args.workdir, args.r0)

    results = {}
    for beta in args.betas:
        print(f"\nbeta = {beta}  ({'below' if beta < args.c/2 else 'above'} "
              f"predicted T_c)")
        outs = []
        for i, adapter in enumerate(args.start_adapters):
            out = os.path.join(args.workdir,
                               f"div_b{beta}_{i}.jsonl")
            one_run(events, out, args.seed + i, args, beta,
                    start_bias=adapter)
            outs.append(out)
        run([sys.executable, "extract_emotions.py", "--score-all"] + outs)
        vals = [tail_protest(o) for o in outs]
        results[beta] = vals
        print(f"    final protest: " + ", ".join(f"{v:+.3f}" for v in vals)
              + f"   |gap| = {abs(vals[0] - vals[1]):.3f}")

    print(f"\n{'=' * 74}")
    print(f"{'beta':>8} {'below T_c':>11} {'gap':>9}")
    print("-" * 74)
    for beta, vals in results.items():
        print(f"{beta:>8.2f} {str(beta < args.c/2):>11} "
              f"{abs(vals[0] - vals[1]):>9.3f}")
    print("""
Prediction: a large gap below T_c (the children stay in different
basins) and a small one above it (they converge). A gap that does not
depend on beta means something other than bistability is keeping them
apart.""")


def mode_bimodal(args):
    """Many seeds, same conditions. Bistable -> bimodal outcomes."""
    print("=" * 74)
    print("BIMODALITY TEST")
    print("=" * 74)
    print(f"""
{args.seeds} seeds at identical conditions (c={args.c}, r0={args.r0}),
across KL strengths. Bistability shows as a BIMODAL distribution of
final protest levels -- two clusters rather than one spread.

predicted T_c = c/2 = {args.c / 2:.3f}""")

    os.makedirs(args.workdir, exist_ok=True)

    all_res = {}
    for beta in args.betas:
        print(f"\nbeta = {beta}  ({'below' if beta < args.c/2 else 'above'} "
              f"predicted T_c)")
        outs = []
        for s in range(args.seeds):
            seed = args.seed + s
            events = make_events(seed, args.episodes, args.workdir, args.r0)
            out = os.path.join(args.workdir, f"bim_b{beta}_s{seed}.jsonl")
            one_run(events, out, seed, args, beta)
            outs.append(out)
        run([sys.executable, "extract_emotions.py", "--score-all"] + outs)
        vals = sorted(tail_protest(o) for o in outs)
        all_res[beta] = vals
        print("    final protest: " + " ".join(f"{v:+.2f}" for v in vals))
        print(f"    bimodality coefficient {bimodality_coefficient(vals):.3f} "
              f"(>0.555 suggests bimodal)")
        print(f"    largest gap {gap_statistic(vals):.3f} of range")

    print(f"\n{'=' * 74}")
    print(f"{'beta':>8} {'below T_c':>11} {'sd':>8} {'bimod coef':>12} "
          f"{'max gap':>9}")
    print("-" * 74)
    for beta, vals in all_res.items():
        print(f"{beta:>8.2f} {str(beta < args.c/2):>11} "
              f"{stdev(vals):>8.3f} {bimodality_coefficient(vals):>12.3f} "
              f"{gap_statistic(vals):>9.3f}")

    print("""
The prediction is the CONTRAST: bimodal below T_c, unimodal above,
switching near c/2.

  unimodal everywhere -> the predicted mechanism is not operating,
                         which falsifies the double-well account for
                         this system
  bimodal everywhere  -> a split exists but is not beta-dependent, so
                         something other than the critical temperature
                         is producing it
  contrast as predicted -> the first evidence that the KL coefficient
                         acts as a critical parameter here

Sarle's coefficient is crude below about 20 samples. Read it alongside
the raw values and the gap statistic rather than on its own.""")

    with open(os.path.join(args.workdir, "bistability.json"), "w") as f:
        json.dump({"c": args.c, "r0": args.r0, "T_c": args.c / 2,
                   "results": {str(k): v for k, v in all_res.items()}},
                  f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["divergence", "bimodal"],
                    default="divergence")
    ap.add_argument("--c", type=float, default=1.0,
                    help="feedback strength; predicted T_c = c/2")
    ap.add_argument("--r0", type=float, default=0.15,
                    help="floor availability when the child never bids")
    ap.add_argument("--betas", nargs="+", type=float, default=[0.2, 2.0],
                    help="KL strengths to compare; span c/2")
    ap.add_argument("--seeds", type=int, default=8,
                    help="seeds per beta (bimodal mode)")
    ap.add_argument("--seed", type=int, default=900)
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--threshold", type=float, default=1.2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--start-adapters", nargs="+",
                    default=["adapters_child_r32", "adapters_child_v5"],
                    help="divergence mode: two different starting states")
    ap.add_argument("--workdir", default="runs_bistability")
    args = ap.parse_args()

    if args.mode == "divergence":
        mode_divergence(args)
    else:
        mode_bimodal(args)


if __name__ == "__main__":
    main()