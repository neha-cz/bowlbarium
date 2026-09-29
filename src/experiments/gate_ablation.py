"""
Gate ablation: does the FEP layer actually do anything?

WHAT IS BEING TESTED
--------------------
The project's premise is that emotional introspection selects WHICH
moments are worth learning from. That has been assumed throughout and
never tested. The evidence so far is not encouraging for it: across
every multi-seed run the FEP surprise signal itself was null, and the
null control caught gate_rate and mean_surprise as consistent artifacts
with no manipulation present.

The protest-reduction result was obtained with an FEP-gated learner.
But nothing so far distinguishes:

    (a) "learning from emotionally surprising moments works"
    (b) "learning from ~50 moments works, whichever ones"

The matched-rate random gate separates them. It fires on the same
NUMBER of turns, chosen at random. If it reproduces protest reduction,
the FEP layer is decoration and the finding is about update count. If
it does not, emotion-derived surprise is selecting the right moments,
and the project's central claim is demonstrated rather than assumed.

The other modes narrow the question further:
  naive       -- is Holt's trend model needed, or would any change
                 detector do?
  percentile  -- would a dumb adaptive threshold with no FEP machinery
                 suffice?

THE SECOND QUESTION: KL DIRECTION
---------------------------------
Whether positive mood should loosen or tighten the constraint has been
open since the mania/depression mapping was first discussed, and
--invert-kl has never been run. If the two directions produce different
outcomes, mood modulation matters. If they are indistinguishable, the
mood->KL machinery is inert and should be reported as such.

COST
----
Each condition is one full learning run (~15 min) plus scoring. The
frozen control is shared across all of them, since no weights move in
it. Six conditions plus one control is roughly two hours per seed.

Usage:
    python src/experiments/gate_ablation.py --seed 700
    python src/experiments/gate_ablation.py --seed 700 --modes fep random
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
        print(r.stdout[-1500:])
        print(r.stderr[-1500:])
        raise SystemExit("command failed")
    return r.stdout


def parse(out):
    info = {"collapsed": False}
    for line in out.splitlines():
        if "fired on" in line:
            try:
                info["gated"] = int(line.split("fired on ")[1].split("/")[0])
            except (IndexError, ValueError):
                pass
        elif "updates in" in line and "gated," in line:
            try:
                info["updates"] = int(
                    line.split("gated, ")[1].split(" updates")[0])
            except (IndexError, ValueError):
                pass
        elif "unique reactions:" in line and "first" in line:
            try:
                tail = line.split("|")[1]
                info["early"] = int(tail.split("first")[1].split(":")[1].split()[0])
                info["late"] = int(tail.split("last")[1].split(":")[1].split()[0])
            except (IndexError, ValueError):
                pass
        elif "COLLAPSE" in line:
            info["collapsed"] = True
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=700)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=1.2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--modes", nargs="+",
                    default=["fep", "naive", "percentile", "random"],
                    help="gate modes to compare")
    ap.add_argument("--test-kl-direction", action="store_true",
                    help="also run fep with --invert-kl, testing whether "
                         "the mood->KL mapping direction matters")
    ap.add_argument("--workdir", default="runs/ablation")
    ap.add_argument("--frac", type=float, default=0.25)
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    prefix = os.path.join(args.workdir, f"A{args.seed}_events")
    run([sys.executable, str(_CORE / "event_generator.py"), "--seed", args.seed,
         "--episodes", args.episodes, "--prefix", prefix, "--quiet"])
    events = f"{prefix}_low_reliability.jsonl"

    common = ["--episodes", args.episodes, "--threshold", args.threshold,
              "--batch-size", args.batch_size, "--seed", args.seed]

    # Frozen control, once -- no weights move, so it is the same
    # baseline for every condition.
    print("\nfrozen control")
    off = os.path.join(args.workdir, f"A{args.seed}_off.jsonl")
    run([sys.executable, str(_CORE / "learning_loop.py"), events,
         "--out", off, "--no-update"] + common)

    # FEP first, so its fire rate can be matched by the random gate.
    conditions = []
    fep_rate = None
    ordered = (["fep"] + [m for m in args.modes if m != "fep"]
               if "fep" in args.modes else list(args.modes))

    for mode in ordered:
        print(f"\ngate = {mode}")
        out = os.path.join(args.workdir, f"A{args.seed}_{mode}.jsonl")
        cmd = [sys.executable, str(_CORE / "learning_loop.py"), events,
               "--out", out, "--gate-mode", mode] + common
        if mode == "random":
            if fep_rate is None:
                print("    (no fep run to match; using 0.5)")
            else:
                cmd += ["--match-rate", f"{fep_rate:.4f}"]
                print(f"    matching fep rate {fep_rate:.0%}")
        info = parse(run(cmd))
        info["mode"] = mode
        info["path"] = out
        if mode == "fep" and info.get("gated"):
            fep_rate = info["gated"] / (args.episodes * 6)
        conditions.append(info)
        print(f"    gated {info.get('gated')}, updates {info.get('updates')}, "
              f"diversity {info.get('early')}->{info.get('late')}"
              + ("  !! COLLAPSED" if info["collapsed"] else ""))

    if args.test_kl_direction:
        print("\ngate = fep, KL direction INVERTED")
        out = os.path.join(args.workdir, f"A{args.seed}_fep_invert.jsonl")
        info = parse(run([sys.executable, str(_CORE / "learning_loop.py"), events,
                          "--out", out, "--gate-mode", "fep",
                          "--invert-kl"] + common))
        info["mode"] = "fep_invert"
        info["path"] = out
        conditions.append(info)
        print(f"    gated {info.get('gated')}, updates {info.get('updates')}, "
              f"diversity {info.get('early')}->{info.get('late')}")

    # Score each against the shared frozen control.
    print("\nscoring")
    for info in conditions:
        run([sys.executable, str(_CORE / "extract_emotions.py"), "--score-all",
             info["path"], off])
        d_on = quartile_drift(load(info["path"]), args.frac)
        d_off = quartile_drift(load(off), args.frac)
        info["drift"] = {c: d_on[c]["drift"] - d_off[c]["drift"]
                         for c in CONCEPT_WEIGHTS}

    print(f"\n{'=' * 78}")
    print(f"GATE ABLATION  (seed {args.seed}, {args.episodes} episodes)")
    print(f"{'=' * 78}")
    print(f"\n{'mode':<14} {'gated':>7} {'updates':>8} {'diversity':>11} "
          + " ".join(f"{c[:9]:>10}" for c in CONCEPT_WEIGHTS))
    print("-" * 78)
    for info in conditions:
        div = f"{info.get('early')}->{info.get('late')}"
        flag = " !!" if info["collapsed"] else ""
        print(f"{info['mode']:<14} {str(info.get('gated')):>7} "
              f"{str(info.get('updates')):>8} {div:>11} "
              + " ".join(f"{info['drift'][c]:>+10.3f}"
                         for c in CONCEPT_WEIGHTS) + flag)

    print(f"\n{'=' * 78}")
    print("READING THIS")
    print(f"{'=' * 78}")

    by_mode = {i["mode"]: i for i in conditions}
    if "fep" in by_mode and "random" in by_mode:
        f = by_mode["fep"]["drift"]["protesting"]
        r = by_mode["random"]["drift"]["protesting"]
        print(f"\nprotesting drift -- fep {f:+.3f}, random {r:+.3f}")
        if abs(r) >= abs(f) * 0.7:
            print("""
The matched-rate random gate reproduces most of the effect. What
mattered was how MANY updates happened, not which moments they came
from. The FEP layer is not doing the selecting, and the honest framing
is that protest reduction follows from reflection-based updating in an
unresponsive environment -- with emotion-derived gating serving only to
keep the update count in a survivable range.""")
        else:
            print("""
The FEP gate produces a substantially larger effect than random gating
at a matched update count. The moments it selects are the ones that
carry the signal, which is the project's central claim shown rather
than assumed. Replicate across seeds before relying on it.""")

    if "fep" in by_mode and "fep_invert" in by_mode:
        a = by_mode["fep"]["drift"]["protesting"]
        b = by_mode["fep_invert"]["drift"]["protesting"]
        print(f"\nKL direction -- default {a:+.3f}, inverted {b:+.3f}")
        if abs(a - b) < 0.15:
            print("The two directions are indistinguishable, so the "
                  "mood->KL\nmodulation is inert at this scale and should "
                  "be reported as such.")
        else:
            print("The direction matters, which is the first evidence that "
                  "mood\nmodulation does real work.")

    print("\nSINGLE SEED. Four findings in this project looked convincing "
          "at n=1\nand did not survive replication.")

    with open(os.path.join(args.workdir,
                           f"ablation_{args.seed}.json"), "w") as f:
        json.dump({"seed": args.seed,
                   "conditions": [{k: v for k, v in i.items() if k != "path"}
                                  for i in conditions]}, f, indent=2)


if __name__ == "__main__":
    main()