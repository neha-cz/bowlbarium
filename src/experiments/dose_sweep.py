"""
Dose-response sweep for the learning loop.

THE LOGIC
---------
The environment effect was null at 10 episodes and clear at 30 -- the
manipulation needed enough dose to show up. The learning effect may be
in the same position: 39 gradient steps on 5.24M LoRA parameters
produced a large drift on one seed and nothing consistent across three.

Rather than replicate at a dose that may be too low, this sweeps the
dose first on a single seed and looks for a MONOTONIC relationship
between how much learning happens and how much the child drifts. A
monotonic dose-response is far more convincing than any single
comparison, and it identifies which setting is worth the expense of
replicating.

THE EFFICIENCY THAT MAKES THIS AFFORDABLE
-----------------------------------------
The frozen control does not depend on the learning rate -- no gradient
steps happen in it. So it runs ONCE and is reused as the baseline for
every dose. That halves the wall-clock cost.

WHAT TO LOOK FOR
----------------
  - drift difference growing with dose  -> the effect is real but was
    under-dosed; replicate at the highest safe setting
  - drift flat across doses             -> more learning does not
    produce more change; the single-run result was noise
  - collapse at higher doses            -> there is a narrow window,
    and its edges are themselves worth reporting

Usage:
    python src/experiments/dose_sweep.py --seed 500
    python src/experiments/dose_sweep.py --seed 500 --lrs 1e-6 3e-6 1e-5
"""

# --- path shim: make the shared modules in src/core importable when this
# script is run directly (python src/<dir>/<script>.py) from the repo root.
import sys as _sys
from pathlib import Path as _Path
_CORE = _Path(__file__).resolve().parent.parent / "core"
if str(_CORE) not in _sys.path:
    _sys.path.insert(0, str(_CORE))

import argparse
import json
import os
import subprocess
import sys

from fep_layer import CONCEPT_WEIGHTS
from drift_analysis import quartile_drift, load


def run(cmd):
    print(f"    $ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit("command failed")
    return r.stdout


def parse_run(out):
    """Pull gate count, update count and diversity trend from stdout."""
    info = {}
    for line in out.splitlines():
        if "gate fired on" in line:
            try:
                info["gated"] = int(line.split("gate fired on ")[1].split("/")[0])
            except (IndexError, ValueError):
                pass
        elif "updates in" in line and "events," in line:
            try:
                info["updates"] = int(
                    line.split("gated, ")[1].split(" updates")[0])
            except (IndexError, ValueError):
                pass
        elif "unique reactions:" in line and "first" in line:
            try:
                tail = line.split("|")[1]
                info["early_unique"] = int(tail.split("first")[1]
                                           .split(":")[1].split()[0])
                info["late_unique"] = int(tail.split("last")[1]
                                          .split(":")[1].split()[0])
            except (IndexError, ValueError):
                pass
        elif "COLLAPSE" in line:
            info["collapsed"] = True
    info.setdefault("collapsed", False)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=1.2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lrs", nargs="+", type=float,
                    default=[1e-6, 3e-6, 1e-5],
                    help="learning rates to sweep, ascending")
    ap.add_argument("--workdir", default="runs/dose")
    ap.add_argument("--frac", type=float, default=0.25)
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    prefix = os.path.join(args.workdir, f"D{args.seed}_events")

    print(f"generating events (seed {args.seed}, {args.episodes} episodes)")
    run([sys.executable, str(_CORE / "event_generator.py"), "--seed", args.seed,
         "--episodes", args.episodes, "--prefix", prefix, "--quiet"])
    events = f"{prefix}_low_reliability.jsonl"

    common = ["--episodes", args.episodes, "--threshold", args.threshold,
              "--batch-size", args.batch_size, "--seed", args.seed]

    # --- frozen control, once ---
    print("\nfrozen control (reused for every dose)")
    off = os.path.join(args.workdir, f"D{args.seed}_off.jsonl")
    off_info = parse_run(run([sys.executable, str(_CORE / "learning_loop.py"), events,
                              "--out", off, "--no-update"] + common))
    print(f"    gated {off_info.get('gated')}, "
          f"diversity {off_info.get('early_unique')} -> "
          f"{off_info.get('late_unique')}")

    # --- one learning run per dose ---
    runs = []
    for lr in args.lrs:
        print(f"\nlearning at lr={lr:g}")
        on = os.path.join(args.workdir, f"D{args.seed}_on_{lr:g}.jsonl")
        info = parse_run(run([sys.executable, str(_CORE / "learning_loop.py"), events,
                              "--out", on, "--lr", lr] + common))
        info["lr"] = lr
        info["path"] = on
        runs.append(info)
        print(f"    gated {info.get('gated')}, "
              f"updates {info.get('updates')}, "
              f"diversity {info.get('early_unique')} -> "
              f"{info.get('late_unique')}"
              + ("  !! COLLAPSED" if info["collapsed"] else ""))

    # --- score each learning run against the shared control ---
    print("\nscoring")
    for info in runs:
        run([sys.executable, str(_CORE / "extract_emotions.py"), "--score-all",
             info["path"], off])
        d_learn = quartile_drift(load(info["path"]), args.frac)
        d_froz = quartile_drift(load(off), args.frac)
        info["drift_diff"] = {c: d_learn[c]["drift"] - d_froz[c]["drift"]
                              for c in CONCEPT_WEIGHTS}
        info["abs_drift"] = sum(abs(v) for v in info["drift_diff"].values())

    # --- report ---
    print(f"\n{'=' * 76}")
    print(f"DOSE-RESPONSE  (seed {args.seed}, {args.episodes} episodes)")
    print(f"{'=' * 76}")
    print(f"\n{'lr':>10} {'updates':>9} {'diversity':>12} "
          f"{'total |drift|':>14} {'collapse':>10}")
    print("-" * 76)
    for info in runs:
        div = (f"{info.get('early_unique')}->{info.get('late_unique')}")
        print(f"{info['lr']:>10.1e} {str(info.get('updates')):>9} "
              f"{div:>12} {info['abs_drift']:>14.3f} "
              f"{'YES' if info['collapsed'] else '-':>10}")

    print(f"\n{'lr':>10} " + " ".join(f"{c:>12}" for c in CONCEPT_WEIGHTS))
    print("-" * 76)
    for info in runs:
        print(f"{info['lr']:>10.1e} " + " ".join(
            f"{info['drift_diff'][c]:>+12.3f}" for c in CONCEPT_WEIGHTS))

    print(f"\n{'=' * 76}")
    print("READING THIS")
    print(f"{'=' * 76}")
    usable = [i for i in runs if not i["collapsed"]]
    if len(usable) < 2:
        print("\nToo few usable doses -- most collapsed. Raise KL_BETA or")
        print("tighten --diversity-threshold before sweeping further.")
    else:
        drifts = [i["abs_drift"] for i in usable]
        rising = all(b >= a for a, b in zip(drifts, drifts[1:]))
        print(f"\ntotal |drift| across usable doses: "
              + " -> ".join(f"{d:.2f}" for d in drifts))
        if rising and drifts[-1] > drifts[0] * 1.5:
            print("\nMonotonic and substantially rising. The effect scales")
            print("with dose, which is the pattern that made the")
            print("environment result credible. Replicate at the highest")
            print("non-collapsing setting across seeds.")
        elif rising:
            print("\nRising but weakly. Extend the sweep upward before")
            print("committing to a replication.")
        else:
            print("\nNot monotonic. More learning does not produce more")
            print("drift, so the single-run effect was probably noise and")
            print("a higher dose will not rescue it.")
        print("\nSingle seed either way -- this sweep only picks which")
        print("setting is worth replicating, it does not establish "
              "anything.")

    out = os.path.join(args.workdir, f"dose_sweep_{args.seed}.json")
    with open(out, "w") as f:
        json.dump({"seed": args.seed, "episodes": args.episodes,
                   "threshold": args.threshold,
                   "batch_size": args.batch_size,
                   "runs": [{k: v for k, v in i.items() if k != "path"}
                            for i in runs]}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()