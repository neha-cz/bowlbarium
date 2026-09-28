"""
Multi-seed replication of the learning-vs-frozen drift result.

WHAT IS BEING TESTED
--------------------
A single run showed the learning child drifting away from protest and
toward withdrawal (drift differences of -0.80 and +0.84 against the
frozen control), while its frozen twin -- same caregiver, same events,
same gating, no gradient steps -- did not.

Three earlier findings in this project looked equally convincing on one
seed and did not survive replication: a 22-point gate-rate effect that
averaged to exactly zero, a "withdrawn" effect consistent across three
seeds that reversed on the next five, and a "distressed" effect that
also appeared in the null control. This runs the same test that killed
those.

WHAT EACH SEED DOES
-------------------
  1. generate events at that seed
  2. run the learning loop with weight updates
  3. run the SAME loop with --no-update (matched control)
  4. score both with shared normalisation
  5. compute per-concept drift, early quartile vs late quartile

Then it reports the drift difference per seed and whether the sign
holds across all of them.

COST
----
Each seed is two full loop runs (~15 and ~14 min) plus scoring. Budget
roughly 35 minutes per seed; three seeds is about 1h45m.

Usage:
    python multi_seed_learning.py --seeds 3
    python multi_seed_learning.py --seeds 3 --threshold 1.2 --batch-size 2
"""

import argparse
import json
import os
import subprocess
import sys

from fep_layer import CONCEPT_WEIGHTS, net_affect
from drift_analysis import quartile_drift, load


def run(cmd):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit(f"failed: {' '.join(str(c) for c in cmd)}")
    return r.stdout


def run_seed(seed, args, workdir):
    prefix = os.path.join(workdir, f"L{seed}_events")
    run([sys.executable, "event_generator.py", "--seed", seed,
         "--episodes", args.episodes, "--prefix", prefix, "--quiet",
         "--high", args.high, "--low", args.low])

    events = f"{prefix}_low_reliability.jsonl"
    on = os.path.join(workdir, f"L{seed}_on.jsonl")
    off = os.path.join(workdir, f"L{seed}_off.jsonl")

    common = ["--episodes", args.episodes, "--threshold", args.threshold,
              "--batch-size", args.batch_size, "--seed", seed]

    print("  [learning]")
    out = run([sys.executable, "learning_loop.py", events,
               "--out", on] + common)
    gated_on = _grep_int(out, "gate fired on ")

    print("  [frozen control]")
    out = run([sys.executable, "learning_loop.py", events,
               "--out", off, "--no-update"] + common)
    gated_off = _grep_int(out, "gate fired on ")

    # Shared normalisation across both, so the two are comparable.
    run([sys.executable, "extract_emotions.py", "--score-all", on, off])

    return on, off, gated_on, gated_off


def _grep_int(text, marker):
    for line in text.splitlines():
        if marker in line:
            try:
                return int(line.split(marker)[1].split("/")[0])
            except (IndexError, ValueError):
                return None
    return None


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def report(results):
    concepts = list(CONCEPT_WEIGHTS)

    print(f"\n{'=' * 76}")
    print(f"AGGREGATE OVER {len(results)} SEEDS")
    print(f"{'=' * 76}")
    print("\nDrift difference = (learning late-early) - (frozen late-early).")
    print("The frozen control saw the same caregiver, events and gating,")
    print("so this isolates the gradient steps.")

    print(f"\n{'concept':<14} {'mean diff':>12} {'sd':>10} "
          f"{'per-seed':>34}")
    print("-" * 76)
    summary = {}
    for c in concepts:
        diffs = [r["drift_diff"][c] for r in results]
        summary[c] = diffs
        shown = " ".join(f"{d:+.2f}" for d in diffs)[:32]
        print(f"{c:<14} {mean(diffs):>+12.3f} {stdev(diffs):>10.3f} "
              f"{shown:>34}")

    print(f"\n{'=' * 76}")
    print("SIGN CONSISTENCY")
    print(f"{'=' * 76}")
    print(f"\n{'concept':<14} {'consistent':>12} {'mean/sd':>12}")
    print("-" * 76)
    n_consistent = 0
    for c in concepts:
        diffs = summary[c]
        signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in diffs]
        ok = all(s == signs[0] for s in signs) and signs[0] != 0
        n_consistent += ok
        sd = stdev(diffs)
        ratio = abs(mean(diffs)) / sd if sd > 0 else float("inf")
        print(f"{c:<14} {'YES' if ok else 'no':>12} {ratio:>12.2f}")

    print(f"\nconsistent: {n_consistent}/{len(concepts)}")
    print("\nWith 4 concepts and few seeds, one or two consistent signs is")
    print("expected by chance -- at 3 seeds each has a 25% chance alone.")
    print("What matters is several concepts consistent together, with")
    print("mean/sd well above 1, and the direction matching the")
    print("single-run result (protesting down, withdrawn up).")

    print(f"\n{'=' * 76}")
    print("GATE RATES  (should be similar -- a large gap means the")
    print("conditions were not matched)")
    print(f"{'=' * 76}")
    print(f"\n{'seed':>6} {'learning':>10} {'frozen':>10}")
    print("-" * 76)
    for r in results:
        print(f"{r['seed']:>6} {str(r['gated_on']):>10} "
              f"{str(r['gated_off']):>10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--start-seed", type=int, default=400)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=1.2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--high", type=float, default=0.9)
    ap.add_argument("--low", type=float, default=0.15)
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--workdir", default="runs_learning")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    seeds = list(range(args.start_seed, args.start_seed + args.seeds))

    results = []
    for i, seed in enumerate(seeds, 1):
        print(f"\n{'=' * 76}")
        print(f"SEED {seed}  ({i}/{len(seeds)})")
        print(f"{'=' * 76}")

        on, off, g_on, g_off = run_seed(seed, args, args.workdir)

        d_learn = quartile_drift(load(on), args.frac)
        d_froz = quartile_drift(load(off), args.frac)
        drift_diff = {c: d_learn[c]["drift"] - d_froz[c]["drift"]
                      for c in CONCEPT_WEIGHTS}

        results.append({
            "seed": seed, "gated_on": g_on, "gated_off": g_off,
            "drift_diff": drift_diff,
            "learning_drift": {c: d_learn[c]["drift"] for c in CONCEPT_WEIGHTS},
            "frozen_drift": {c: d_froz[c]["drift"] for c in CONCEPT_WEIGHTS},
        })

        print("  drift diff: " + "  ".join(
            f"{c}={drift_diff[c]:+.2f}" for c in CONCEPT_WEIGHTS))

        if not args.keep:
            for f in os.listdir(args.workdir):
                if f.startswith(f"L{seed}_events"):
                    os.remove(os.path.join(args.workdir, f))

    report(results)

    out = os.path.join(args.workdir, "learning_multi_seed.json")
    with open(out, "w") as f:
        json.dump({"seeds": seeds, "threshold": args.threshold,
                   "batch_size": args.batch_size, "episodes": args.episodes,
                   "results": results}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()