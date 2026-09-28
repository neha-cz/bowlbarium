"""
Reliability sweep: is the collapse a bifurcation or a slope?

THE QUESTION
------------
The dissipative-structure framing says caregiver responsiveness is the
flux that maintains the child's behavioural repertoire. Bénard cells do
not fade in gradually as a fluid is heated -- there is a critical
gradient below which there is no structure and above which there is.

If responsiveness works like flux, there should be a CRITICAL
RELIABILITY: above it the repertoire is maintained, below it it
collapses, with a relatively sharp boundary.

The boring alternative -- the child is trained on compliant targets, so
it becomes compliant -- predicts a smooth monotonic decline instead.
Nothing thermodynamic is needed for that, and the framing would be
decoration.

This sweep distinguishes them. It is the same shape as the dose sweep
that worked before: vary one parameter, look for structure in the
response.

NOTE ON THE MEASURE
-------------------
Unique-reaction counts were adequate as a collapse alarm but are a poor
order parameter -- they saturate and ignore how mass is distributed. So
this reports Shannon entropy over reaction types and distinct-1/2
n-gram diversity alongside them. Entropy collapse is a well-studied
failure mode in RL for language models, and these measures connect this
work to it rather than to an ad-hoc diversity count.

Every run here uses the LEARNING loop with weights updating. A frozen
run at each level is optional (--with-frozen) and doubles the cost, but
is what separates "the environment produces less varied behaviour" from
"learning under this environment collapses the repertoire".

COST
----
One learning run per reliability level. At roughly 45 min per
30-episode run, four levels is about 3 hours; --episodes 15 halves it.
Start with --levels 0.15 0.9 --episodes 10 to check the machinery.

Usage:
    python reliability_sweep.py --seed 900
    python reliability_sweep.py --seed 900 --levels 0.15 0.4 0.65 0.9
    python reliability_sweep.py --seed 900 --with-frozen
"""

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter


def run(cmd):
    print(f"    $ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:])
        print(r.stderr[-1500:])
        raise SystemExit("command failed")
    return r.stdout


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def shannon_entropy(items):
    """Entropy in bits over the distribution of item types.

    Better than a unique count as an order parameter: a child saying
    one thing 90% of the time and nine others once each has the same
    unique count as one spread evenly, but far lower entropy.
    """
    if not items:
        return 0.0
    counts = Counter(items)
    n = len(items)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def distinct_n(texts, n):
    """Ratio of unique n-grams to total n-grams."""
    grams = []
    for t in texts:
        toks = t.lower().split()
        grams += [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def measure(rows, frac=0.3):
    """Diversity of the LAST `frac` of a run -- the end state is what
    the order parameter should describe."""
    q = max(5, int(len(rows) * frac))
    late = [r["child_reaction"] for r in rows[-q:]]
    early = [r["child_reaction"] for r in rows[:q]]
    return {
        "n": q,
        "entropy_late": shannon_entropy(late),
        "entropy_early": shannon_entropy(early),
        "unique_late": len(set(late)) / len(late),
        "distinct1": distinct_n(late, 1),
        "distinct2": distinct_n(late, 2),
        "mean_words": sum(len(t.split()) for t in late) / len(late),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=900)
    ap.add_argument("--levels", nargs="+", type=float,
                    default=[0.15, 0.4, 0.65, 0.9],
                    help="caregiver reliability levels to sweep")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=1.2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--with-frozen", action="store_true",
                    help="also run a frozen control at each level, which "
                         "separates environment effects from learning "
                         "effects but doubles the cost")
    ap.add_argument("--workdir", default="runs_reliability")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    W = args.workdir
    common = ["--episodes", args.episodes, "--threshold", args.threshold,
              "--batch-size", args.batch_size, "--seed", args.seed]

    results = []
    for level in sorted(args.levels):
        print(f"\n{'=' * 70}")
        print(f"RELIABILITY {level}")
        print(f"{'=' * 70}")

        prefix = os.path.join(W, f"R{args.seed}_{level:g}")
        # Both arms at the same level; only the "high" file is used.
        run([sys.executable, "event_generator.py", "--seed", args.seed,
             "--episodes", args.episodes, "--prefix", prefix, "--quiet",
             "--high", level, "--low", level])
        events = f"{prefix}_high_reliability.jsonl"

        out = os.path.join(W, f"R{args.seed}_{level:g}_learn.jsonl")
        print("  learning")
        run([sys.executable, "learning_loop.py", events, "--out", out]
            + common)
        entry = {"level": level, "learn": measure(load(out))}

        if args.with_frozen:
            out_f = os.path.join(W, f"R{args.seed}_{level:g}_frozen.jsonl")
            print("  frozen control")
            run([sys.executable, "learning_loop.py", events, "--out", out_f,
                 "--no-update"] + common)
            entry["frozen"] = measure(load(out_f))

        results.append(entry)
        m = entry["learn"]
        print(f"  entropy {m['entropy_early']:.2f} -> {m['entropy_late']:.2f}"
              f"   distinct-2 {m['distinct2']:.2f}"
              f"   words {m['mean_words']:.1f}")

    print(f"\n{'=' * 78}")
    print(f"RELIABILITY SWEEP  (seed {args.seed}, "
          f"{args.episodes} episodes)")
    print(f"{'=' * 78}")

    print(f"\n{'level':>7} {'H early':>9} {'H late':>8} {'dH':>8} "
          f"{'unique':>8} {'dist-1':>8} {'dist-2':>8} {'words':>7}")
    print("-" * 78)
    for e in results:
        m = e["learn"]
        print(f"{e['level']:>7.2f} {m['entropy_early']:>9.2f} "
              f"{m['entropy_late']:>8.2f} "
              f"{m['entropy_late'] - m['entropy_early']:>+8.2f} "
              f"{m['unique_late']:>8.2f} {m['distinct1']:>8.2f} "
              f"{m['distinct2']:>8.2f} {m['mean_words']:>7.1f}")

    if args.with_frozen:
        print(f"\nfrozen controls (no weight updates)")
        print(f"{'level':>7} {'H late':>8} {'dist-2':>8}")
        print("-" * 78)
        for e in results:
            m = e["frozen"]
            print(f"{e['level']:>7.2f} {m['entropy_late']:>8.2f} "
                  f"{m['distinct2']:>8.2f}")

    # --- shape of the response ---
    print(f"\n{'=' * 78}")
    print("IS IT A THRESHOLD OR A SLOPE?")
    print(f"{'=' * 78}")

    levels = [e["level"] for e in results]
    ent = [e["learn"]["entropy_late"] for e in results]

    if len(levels) < 3:
        print("\nNeed at least three levels to judge the shape.")
    else:
        # Largest single step vs the average step. A bifurcation shows
        # one dominant jump; a slope spreads the change evenly.
        steps = [abs(ent[i + 1] - ent[i]) for i in range(len(ent) - 1)]
        total = abs(ent[-1] - ent[0])
        biggest = max(steps)
        share = biggest / sum(steps) if sum(steps) else 0
        where = steps.index(biggest)

        print(f"\nentropy across levels: "
              + " -> ".join(f"{e:.2f}" for e in ent))
        print(f"total change: {total:.2f}")
        print(f"largest single step: {biggest:.2f} "
              f"(between {levels[where]:.2f} and {levels[where + 1]:.2f})")
        print(f"that step is {share:.0%} of all movement")

        if share > 0.6 and total > 0.5:
            print(f"""
One step accounts for most of the movement, which is what a bifurcation
looks like: the repertoire is maintained above roughly
{levels[where + 1]:.2f} and collapses below it. The dissipative framing
earns its place -- a smooth training effect would not produce a
threshold.

Next: sample finely around {levels[where]:.2f}-{levels[where + 1]:.2f}
to locate the critical point, and check for critical slowing down
(runs near the boundary taking longer to settle, or varying more
between seeds).""")
        elif total < 0.3:
            print("""
Entropy barely changes across the whole reliability range. The
repertoire is not responding to caregiver responsiveness at all, which
undercuts the flux account -- and also suggests the collapse seen
earlier was driven by the learning rate rather than by the
environment.""")
        else:
            print("""
The change is spread fairly evenly across levels: a slope, not a
threshold. That is the boring account -- more unresponsive care means
more compliant training targets means less varied behaviour, no
bifurcation and no critical point. The thermodynamic framing would be
redescription rather than explanation, and should not be claimed.""")

    print("\nSingle seed. The shape of a curve from one run is suggestive "
          "only.")

    with open(os.path.join(W, f"sweep_{args.seed}.json"), "w") as f:
        json.dump({"seed": args.seed, "episodes": args.episodes,
                   "results": results}, f, indent=2)
    print(f"\nwrote {os.path.join(W, f'sweep_{args.seed}.json')}")


if __name__ == "__main__":
    main()