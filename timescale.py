"""
Timescale asymmetry: how long does responsive care take to undo
deprivation?

THE QUESTION
------------
Result 3 showed that twenty episodes of responsive care did not close a
gap opened by twenty episodes of unresponsive care. That is an
observation, not a measurement. The claim worth testing is an
ASYMMETRY: that undoing the effect takes substantially longer than
creating it.

WHAT THIS DOES DIFFERENTLY
--------------------------
Rather than lengthening phase B and checking the endpoint, it BINS
phase B and tracks the LH-HH gap across it. That yields a decay curve
instead of a single number, which answers a better question: not
"had it closed by episode N" but "what is the shape and timescale of
the decay".

Three outcomes are distinguishable this way, and only the first two are
visible from an endpoint alone:

  gap decays to zero at bin k   -> the timescale is k bins; compare
                                   against the phase-A length to get
                                   the asymmetry ratio
  gap decays but plateaus       -> partial recovery with a permanent
                                   residue, which is the stronger
                                   version of the claim
  gap flat across all bins      -> no recovery process at all; the
                                   earlier "did not close in 20" was
                                   not slow decay but no decay

DESIGN
------
  phase A   20 episodes, LH unresponsive / HH responsive
  phase B   60 episodes, both responsive, same events

Phase B is three times phase A so there is room for the gap to close
well before the end. Both conditions run the identical phase-B event
file, so the comparison within each bin is clean.

COST
----
80 episodes per condition, two conditions, per seed -- roughly four
hours per seed at recent timings. Run ONE seed first to see the curve
shape; replicate only if there is a curve worth replicating.

Usage:
    python timescale.py --seed 850
    python timescale.py --seed 850 --episodes-a 20 --episodes-b 60 --bins 6
"""

import argparse
import json
import os
import subprocess
import sys

from fep_layer import CONCEPT_WEIGHTS

CONDITIONS = {"LH": ("low", "high"), "HH": ("high", "high")}


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


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def tail_mean(rows, concept, frac=0.3):
    q = max(5, int(len(rows) * frac))
    return mean([r["activations"][concept] for r in rows[-q:]])


def binned(rows, concept, bins):
    """Mean activation per bin of episodes, in order."""
    episodes = sorted({r["episode"] for r in rows})
    if not episodes:
        return []
    per_bin = max(1, len(episodes) // bins)
    out = []
    for b in range(bins):
        lo = b * per_bin
        hi = (b + 1) * per_bin if b < bins - 1 else len(episodes)
        eps = set(episodes[lo:hi])
        vals = [r["activations"][concept] for r in rows if r["episode"] in eps]
        out.append(mean(vals) if vals else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=850)
    ap.add_argument("--episodes-a", type=int, default=20,
                    help="phase A length -- the deprivation that has to "
                         "be undone")
    ap.add_argument("--episodes-b", type=int, default=60,
                    help="phase B length -- make this well longer than "
                         "phase A so the gap has room to close")
    ap.add_argument("--bins", type=int, default=6,
                    help="bins to split phase B into for the decay curve")
    ap.add_argument("--threshold", type=float, default=1.2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--base-adapter", default="adapters_child_r32")
    ap.add_argument("--workdir", default="runs_timescale")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    W = args.workdir
    S = args.seed

    print(f"generating events "
          f"(phase A {args.episodes_a}, phase B {args.episodes_b})")
    run([sys.executable, "event_generator.py", "--seed", S,
         "--episodes", args.episodes_a,
         "--prefix", os.path.join(W, f"T{S}_A"), "--quiet"])
    run([sys.executable, "event_generator.py", "--seed", S + 1000,
         "--episodes", args.episodes_b,
         "--prefix", os.path.join(W, f"T{S}_B"), "--quiet"])

    def events_for(phase, level):
        name = "low_reliability" if level == "low" else "high_reliability"
        return os.path.join(W, f"T{S}_{phase}_{name}.jsonl")

    paths = {}
    for cond, (lvl_a, lvl_b) in CONDITIONS.items():
        print(f"\n{'=' * 70}")
        print(f"CONDITION {cond}")
        print(f"{'=' * 70}")

        common_a = ["--episodes", args.episodes_a,
                    "--threshold", args.threshold,
                    "--batch-size", args.batch_size, "--seed", S]
        common_b = ["--episodes", args.episodes_b,
                    "--threshold", args.threshold,
                    "--batch-size", args.batch_size, "--seed", S]

        out_a = os.path.join(W, f"T{S}_{cond}_A.jsonl")
        adapter_a = os.path.join(W, f"T{S}_{cond}_adapter_A")
        print(f"  phase A ({lvl_a}), {args.episodes_a} episodes")
        run([sys.executable, "learning_loop.py", events_for("A", lvl_a),
             "--out", out_a, "--child-adapter", args.base_adapter,
             "--save-adapter", adapter_a] + common_a)

        out_b = os.path.join(W, f"T{S}_{cond}_B.jsonl")
        print(f"  phase B ({lvl_b}), {args.episodes_b} episodes, "
              f"continuing from A")
        run([sys.executable, "learning_loop.py", events_for("B", lvl_b),
             "--out", out_b, "--child-adapter", adapter_a] + common_b)

        paths[cond] = (out_a, out_b)

    print("\nscoring all phases with shared normalisation")
    flat = [p for pair in paths.values() for p in pair]
    run([sys.executable, "extract_emotions.py", "--score-all"] + flat)

    rows = {c: (load(paths[c][0]), load(paths[c][1])) for c in CONDITIONS}

    print(f"\n{'=' * 74}")
    print(f"TIMESCALE  (seed {S}, phase A {args.episodes_a} ep, "
          f"phase B {args.episodes_b} ep)")
    print(f"{'=' * 74}")

    for concept in ("protesting", "comforted"):
        lh_a = tail_mean(rows["LH"][0], concept)
        hh_a = tail_mean(rows["HH"][0], concept)
        gap_a = lh_a - hh_a

        lh_bins = binned(rows["LH"][1], concept, args.bins)
        hh_bins = binned(rows["HH"][1], concept, args.bins)
        gaps = [l - h for l, h in zip(lh_bins, hh_bins)]

        per_bin = args.episodes_b / args.bins
        print(f"\n{concept.upper()}")
        print(f"  gap at end of phase A: {gap_a:+.3f}")
        print(f"\n  {'bin':>4} {'episodes':>12} {'LH':>9} {'HH':>9} "
              f"{'gap':>9} {'% of A':>9}")
        print("  " + "-" * 60)
        for i, (l, h, g) in enumerate(zip(lh_bins, hh_bins, gaps)):
            lo = int(i * per_bin)
            hi = int((i + 1) * per_bin)
            frac = g / gap_a if gap_a else 0.0
            print(f"  {i:>4} {f'{lo}-{hi}':>12} {l:>+9.3f} {h:>+9.3f} "
                  f"{g:>+9.3f} {frac:>8.0%}")

        if concept == "protesting":
            p_gap_a, p_gaps = gap_a, gaps

    print(f"\n{'=' * 74}")
    print("READING THIS")
    print(f"{'=' * 74}")

    per_bin = args.episodes_b / args.bins
    fracs = [g / p_gap_a if p_gap_a else 0.0 for g in p_gaps]

    # First bin where the gap falls below 25% of its phase-A size and
    # stays there.
    closed_at = None
    for i in range(len(fracs)):
        if all(abs(f) < 0.25 for f in fracs[i:]):
            closed_at = i
            break

    print(f"\nprotesting gap as a fraction of its phase-A size, by bin:")
    print("  " + " -> ".join(f"{f:.0%}" for f in fracs))

    if closed_at is not None:
        eps = int(closed_at * per_bin)
        ratio = eps / args.episodes_a if args.episodes_a else 0
        print(f"""
The gap closes at roughly episode {eps} of phase B.

  {args.episodes_a} episodes of deprivation
  {eps} episodes of responsive care to undo it
  ratio {ratio:.1f}x

That ratio is the result. Above 1 means undoing costs more than
creating -- an asymmetry in timescales, which is the psychologically
interesting version of the claim. Near 1 means the process is roughly
symmetric and there is nothing special about the direction.""")
    elif abs(fracs[-1]) > 0.75:
        print(f"""
The gap is still at {fracs[-1]:.0%} of its phase-A size after
{args.episodes_b} episodes -- three times as long as the deprivation
that produced it -- and shows no downward trend.

That is not slow decay, it is NO decay. Responsive care is not
reversing the effect at all on this timescale, which is a stronger
claim than Result 3 made and worth stating as such.""")
    else:
        print(f"""
The gap declined to {fracs[-1]:.0%} of its phase-A size but has not
closed. Partial recovery with a residue. Either the decay is slower
than {args.episodes_b} episodes can show, or there is a permanent
component. Extending phase B further would distinguish those.""")

    print("\nSingle seed, and weights persist across phases by "
          "construction.\nReplicate before relying on the number.")

    with open(os.path.join(W, f"timescale_{S}.json"), "w") as f:
        json.dump({"seed": S, "episodes_a": args.episodes_a,
                   "episodes_b": args.episodes_b,
                   "protesting_gap_a": p_gap_a,
                   "protesting_gaps_b": p_gaps}, f, indent=2)
    print(f"\nwrote {os.path.join(W, f'timescale_{S}.json')}")


if __name__ == "__main__":
    main()