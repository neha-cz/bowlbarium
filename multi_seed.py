"""
Run the full pipeline across several seeds and aggregate.

WHY
---
Every result so far is single-seed, n=60 per condition. The two effects
that survived scrutiny -- a higher gate fire rate under low reliability,
and a shift toward withdrawal rather than protest -- cannot be
distinguished from noise without repeated runs.

This varies BOTH sources of randomness per seed:
  - which events occur, and when the mother is available (event seed)
  - what the agents actually say (sampling seed)

Sharing one seed for both is deliberate: each seed is one complete
independent replication of the experiment.

WHAT IT REPORTS
---------------
Per-seed values plus mean and standard deviation across seeds for each
metric. An effect whose per-seed sign is consistent is worth something;
one that flips sign between seeds is noise regardless of how large the
pooled difference looks.

COST
----
Roughly 8-10 minutes per seed on an M5 Air: two 60-event transcript runs
(~3 min each) plus scoring 120 turns. Five seeds is about 45 minutes.
Start with --seeds 3 to check it works end to end.

Usage:
    python multi_seed.py --seeds 5
    python multi_seed.py --seeds 5 --threshold 2.2 --residualize
    python multi_seed.py --seeds 5 --keep    # keep intermediate files
"""

import argparse
import json
import os
import subprocess
import sys

from fep_layer import AffectTracker, PopulationBaseline, CONCEPT_WEIGHTS

CONDITIONS = ["high_reliability", "low_reliability"]


def run(cmd):
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return result.stdout


def run_one_seed(seed, episodes, residualize, workdir,
                 high=0.9, low=0.15, null_control=False):
    prefix = os.path.join(workdir, f"s{seed}_events")

    # In a null control both reliabilities are equal, so sharing one seed
    # would make both arms draw the SAME events and sample the SAME
    # completions -- identical transcripts, difference trivially zero,
    # nothing tested. Offsetting the low arm gives independent draws from
    # the same distribution, which is what a null control needs.
    offset = 7919 if null_control else 0

    run([sys.executable, "event_generator.py",
         "--seed", str(seed), "--episodes", str(episodes),
         "--prefix", prefix, "--quiet",
         "--high", str(high), "--low", str(low),
         "--low-seed-offset", str(offset)])

    transcripts = []
    for cond in CONDITIONS:
        ev = f"{prefix}_{cond}.jsonl"
        tr = os.path.join(workdir, f"s{seed}_transcript_{cond}.jsonl")
        sample_seed = seed + (offset if cond == "low_reliability" else 0)
        run([sys.executable, "run_episodes.py", ev,
             "--out", tr, "--seed", str(sample_seed)])
        transcripts.append(tr)

    score_cmd = [sys.executable, "extract_emotions.py",
                 "--score-all"] + transcripts
    if residualize:
        score_cmd.append("--residualize-length")
    run(score_cmd)

    return [t.replace("transcript_", "affect_") for t in transcripts]


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def analyze_pair(high_path, low_path, threshold):
    high_rows, low_rows = load(high_path), load(low_path)

    population = PopulationBaseline(min_n=30)
    for r in high_rows + low_rows:
        population.update(sum(CONCEPT_WEIGHTS.get(c, 0) * v
                              for c, v in r["activations"].items()))

    out = {}
    for label, rows in (("high", high_rows), ("low", low_rows)):
        tracker = AffectTracker(population=population,
                                surprise_threshold=threshold)
        readings, current_ep = [], None
        for r in rows:
            if current_ep is not None and r["episode"] != current_ep:
                tracker.episode_boundary()
            current_ep = r["episode"]
            readings.append(tracker.observe(r["activations"],
                                            episode=r["episode"]))

        n = len(readings)
        out[label] = {
            "gate_rate": sum(x.is_surprising for x in readings) / n,
            "mean_surprise": sum(abs(x.surprise) for x in readings) / n,
            "mean_affect": sum(x.net_affect for x in readings) / n,
            "final_mood": readings[-1].mood,
        }
        for c in CONCEPT_WEIGHTS:
            out[label][c] = sum(r["activations"][c] for r in rows) / len(rows)
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def report(results, threshold, null_control=False):
    metrics = ["gate_rate", "mean_surprise", "withdrawn", "protesting",
               "comforted", "distressed", "mean_affect", "final_mood"]

    print(f"\n{'=' * 74}")
    print(f"AGGREGATE OVER {len(results)} SEEDS  "
          f"(surprise_threshold={threshold})")
    if null_control:
        print("*** NULL CONTROL -- both arms at the same reliability. ***")
        print("*** Any metric marked consistent here is a FALSE POSITIVE ***")
        print("*** produced by the pipeline, not by the manipulation.    ***")
    print(f"{'=' * 74}")

    print(f"\n{'metric':<16} {'high':>16} {'low':>16} {'diff':>16}")
    print("-" * 74)
    summary = {}
    for m in metrics:
        highs = [r["high"][m] for r in results]
        lows = [r["low"][m] for r in results]
        diffs = [l - h for h, l in zip(highs, lows)]
        summary[m] = diffs
        print(f"{m:<16} "
              f"{mean(highs):>+9.3f}+-{stdev(highs):<5.3f} "
              f"{mean(lows):>+9.3f}+-{stdev(lows):<5.3f} "
              f"{mean(diffs):>+9.3f}+-{stdev(diffs):<5.3f}")

    print(f"\n{'=' * 74}")
    print("SIGN CONSISTENCY  (does the effect hold in every seed?)")
    print(f"{'=' * 74}")
    print(f"\n{'metric':<16} {'per-seed diffs':<40} {'consistent':>12}")
    print("-" * 74)
    for m in metrics:
        diffs = summary[m]
        signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in diffs]
        consistent = all(s == signs[0] for s in signs) and signs[0] != 0
        shown = " ".join(f"{d:+.2f}" for d in diffs)[:38]
        print(f"{m:<16} {shown:<40} {'YES' if consistent else 'no':>12}")

    print("\nAn effect that flips sign between seeds is noise no matter how")
    print("large the pooled difference looks. Consistent sign across every")
    print("seed, with a mean well outside the per-seed spread, is the")
    print("minimum bar for treating a difference as real.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--start-seed", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="surprise threshold; run calibrate_threshold.py "
                         "first to pick this")
    ap.add_argument("--high", type=float, default=0.9,
                    help="reliability for the high condition")
    ap.add_argument("--low", type=float, default=0.15,
                    help="reliability for the low condition. Set EQUAL to "
                         "--high to run a null control: with no "
                         "manipulation, any 'consistent' effect that still "
                         "appears is manufactured by the pipeline.")
    ap.add_argument("--residualize", action="store_true")
    ap.add_argument("--workdir", default="runs")
    ap.add_argument("--keep", action="store_true",
                    help="keep intermediate transcripts")
    ap.add_argument("--reuse", action="store_true",
                    help="skip generation, re-analyse existing affect files")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    is_null = abs(args.high - args.low) < 1e-9
    if is_null:
        print("NULL CONTROL: both arms at reliability "
              f"{args.high}, independent seeds.\n")
    seeds = list(range(args.start_seed, args.start_seed + args.seeds))

    results = []
    for i, seed in enumerate(seeds, 1):
        print(f"\n{'=' * 74}")
        print(f"SEED {seed}  ({i}/{len(seeds)})")
        print(f"{'=' * 74}")

        paths = [os.path.join(args.workdir, f"s{seed}_affect_{c}.jsonl")
                 for c in CONDITIONS]

        if args.reuse and all(os.path.exists(p) for p in paths):
            print("  reusing existing affect files")
        else:
            paths = run_one_seed(seed, args.episodes,
                                 args.residualize, args.workdir,
                                 high=args.high, low=args.low,
                                 null_control=is_null)

        results.append(analyze_pair(paths[0], paths[1], args.threshold))
        r = results[-1]
        print(f"  gate: high {r['high']['gate_rate']:.0%} "
              f"low {r['low']['gate_rate']:.0%}   "
              f"withdrawn diff {r['low']['withdrawn'] - r['high']['withdrawn']:+.3f}")

        if not args.keep:
            for f in os.listdir(args.workdir):
                if f.startswith(f"s{seed}_") and (
                        "events" in f or "transcript" in f):
                    os.remove(os.path.join(args.workdir, f))

    report(results, args.threshold,
           null_control=is_null)

    out = os.path.join(args.workdir, "multi_seed_results.json")
    with open(out, "w") as f:
        json.dump({"threshold": args.threshold, "seeds": seeds,
                   "high": args.high, "low": args.low,
                   "null_control": is_null,
                   "residualized": args.residualize, "results": results}, f,
                  indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()