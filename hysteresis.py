"""
Hysteresis: does protest recover when the caregiver becomes responsive?

THE QUESTION
------------
Protest reduction under an unresponsive caregiver replicated across
seeds and survived length correction. What is not known is whether it
is REVERSIBLE.

Thermodynamically that is the question of whether the process is
reversible or path-dependent. Psychologically it is the whole clinical
point about attachment styles: they persist after the conditions that
formed them have changed. If the child recovers as soon as the mother
becomes responsive, what was measured is a response to current
conditions. If it does not recover, or recovers only partly, or needs
far more responsive episodes than it took unresponsive ones to lose,
that is hysteresis.

DESIGN
------
Each condition runs two phases with weights PERSISTING across the
switch -- phase B continues training from phase A's learned adapter.

    LH   phase A low reliability, phase B high      the test
    HH   phase A high, phase B high                 never-depressed control
    LL   phase A low, phase B low                   no-recovery control

The signature is a comparison at the END of phase B:

    LH ~= HH   -> recovered; the change was reversible
    LH ~= LL   -> no recovery; fully path-dependent
    in between -> partial recovery, which is still hysteresis

LL matters because without it, a flat LH could mean either "did not
recover" or "protest was already drifting down for unrelated reasons".
HH matters because it establishes what protest looks like after the
same total number of episodes without ever being suppressed.

COST
----
Three conditions, each two phases. At roughly 45 minutes per 30-episode
phase, 20 episodes per phase is about 2.5 hours total. Start with
--episodes 10 --conditions LH HH to check the machinery end to end
before committing.

Usage:
    python hysteresis.py --seed 800 --episodes 20
    python hysteresis.py --seed 800 --episodes 10 --conditions LH HH
"""

import argparse
import json
import os
import subprocess
import sys

from fep_layer import CONCEPT_WEIGHTS

CONDITIONS = {
    "LH": ("low", "high"),
    "HH": ("high", "high"),
    "LL": ("low", "low"),
}


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


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def tail_mean(rows, concept, frac=0.3):
    """Mean activation over the last `frac` of a phase."""
    q = max(5, int(len(rows) * frac))
    return mean([r["activations"][concept] for r in rows[-q:]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=800)
    ap.add_argument("--episodes", type=int, default=20,
                    help="episodes PER PHASE")
    ap.add_argument("--threshold", type=float, default=1.2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--conditions", nargs="+", default=["LH", "HH", "LL"],
                    choices=list(CONDITIONS))
    ap.add_argument("--workdir", default="runs_hysteresis")
    ap.add_argument("--base-adapter", default="adapters_child_r32")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    W = args.workdir

    # Two independent event streams per reliability level, so phase A
    # and phase B are not the same events replayed.
    print("generating events")
    for phase, offset in (("A", 0), ("B", 1000)):
        run([sys.executable, "event_generator.py",
             "--seed", args.seed + offset, "--episodes", args.episodes,
             "--prefix", os.path.join(W, f"H{args.seed}_{phase}"),
             "--quiet"])

    def events_for(phase, level):
        name = "low_reliability" if level == "low" else "high_reliability"
        return os.path.join(W, f"H{args.seed}_{phase}_{name}.jsonl")

    common = ["--episodes", args.episodes, "--threshold", args.threshold,
              "--batch-size", args.batch_size, "--seed", args.seed]

    results = {}
    for cond in args.conditions:
        lvl_a, lvl_b = CONDITIONS[cond]
        print(f"\n{'=' * 70}")
        print(f"CONDITION {cond}:  phase A {lvl_a}  ->  phase B {lvl_b}")
        print(f"{'=' * 70}")

        # --- phase A, from the cold-start adapter ---
        out_a = os.path.join(W, f"{cond}_A.jsonl")
        adapter_a = os.path.join(W, f"{cond}_adapter_A")
        print(f"  phase A ({lvl_a})")
        run([sys.executable, "learning_loop.py", events_for("A", lvl_a),
             "--out", out_a, "--child-adapter", args.base_adapter,
             "--save-adapter", adapter_a] + common)

        # --- phase B, CONTINUING from phase A's weights ---
        out_b = os.path.join(W, f"{cond}_B.jsonl")
        print(f"  phase B ({lvl_b}), continuing from phase A weights")
        run([sys.executable, "learning_loop.py", events_for("B", lvl_b),
             "--out", out_b, "--child-adapter", adapter_a] + common)

        results[cond] = (out_a, out_b)

    # Score everything together, so all phases share one normalisation
    # and are directly comparable.
    print("\nscoring all phases with shared normalisation")
    all_paths = [p for pair in results.values() for p in pair]
    run([sys.executable, "extract_emotions.py", "--score-all"] + all_paths)

    print(f"\n{'=' * 78}")
    print(f"HYSTERESIS  (seed {args.seed}, {args.episodes} episodes per phase)")
    print(f"{'=' * 78}")

    table = {}
    for cond, (pa, pb) in results.items():
        rows_a, rows_b = load(pa), load(pb)
        table[cond] = {
            c: {"end_A": tail_mean(rows_a, c), "end_B": tail_mean(rows_b, c)}
            for c in CONCEPT_WEIGHTS
        }

    for concept in ("protesting", "comforted"):
        print(f"\n{concept.upper()}  (mean over the last 30% of each phase)")
        print(f"{'cond':<6} {'A->B':<12} {'end of A':>10} {'end of B':>10} "
              f"{'change':>9}")
        print("-" * 78)
        for cond in args.conditions:
            a = table[cond][concept]["end_A"]
            b = table[cond][concept]["end_B"]
            path = f"{CONDITIONS[cond][0]}->{CONDITIONS[cond][1]}"
            print(f"{cond:<6} {path:<12} {a:>+10.3f} {b:>+10.3f} "
                  f"{b - a:>+9.3f}")

    print(f"\n{'=' * 78}")
    print("READING THIS")
    print(f"{'=' * 78}")

    if "LH" in table and "HH" in table:
        lh_a = table["LH"]["protesting"]["end_A"]
        lh_b = table["LH"]["protesting"]["end_B"]
        hh_b = table["HH"]["protesting"]["end_B"]

        print(f"\nprotesting at end of phase B:")
        print(f"  LH (was suppressed, then responsive care)  {lh_b:+.3f}")
        print(f"  HH (never suppressed)                      {hh_b:+.3f}")
        if "LL" in table:
            ll_b = table["LL"]["protesting"]["end_B"]
            print(f"  LL (suppressed throughout)                 {ll_b:+.3f}")

        gap_to_hh = abs(lh_b - hh_b)
        moved = abs(lh_b - lh_a)
        print(f"\nLH moved {moved:+.3f} from end of A to end of B, "
              f"and sits {gap_to_hh:.3f} from HH.")

        if gap_to_hh < 0.15:
            print("""
LH ends up where HH does. Protest recovered once care became
responsive, so the change is a reversible response to current
conditions rather than an acquired disposition. No hysteresis.""")
        elif moved < 0.15:
            print("""
LH barely moved during responsive care and remains far from HH. The
suppression persisted after the conditions that produced it were
removed -- path dependence, and the thermodynamic signature of an
irreversible process. This is also the clinical claim about attachment
styles, which is what makes it worth reporting.""")
        else:
            print("""
LH recovered partially: it moved toward HH without reaching it. Partial
recovery is still hysteresis -- the return path differs from the
outgoing one. Worth measuring how many responsive episodes would be
needed to close the remaining gap, since an asymmetry in timescales is
itself the result.""")

    print("\nSingle seed. Replicate before relying on it -- four findings "
          "in\nthis project looked convincing at n=1 and did not survive.")

    with open(os.path.join(W, f"hysteresis_{args.seed}.json"), "w") as f:
        json.dump({"seed": args.seed, "episodes": args.episodes,
                   "table": table}, f, indent=2)
    print(f"\nwrote {os.path.join(W, f'hysteresis_{args.seed}.json')}")


if __name__ == "__main__":
    main()