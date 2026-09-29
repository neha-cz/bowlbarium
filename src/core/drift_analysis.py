"""
Within-condition drift analysis.

WHY THIS AND NOT A POOLED COMPARISON
------------------------------------
analyze_affect.py compares pooled means between two conditions. That is
the wrong instrument here, for two reasons.

First, the learning child's weights change DURING the run, so its own
distribution is non-stationary. A pooled mean mixes pre-learning and
post-learning behaviour into one number and cannot distinguish "the
child changed over time" from "the child was different from the start".

Second, the pooled z-scoring in extract_emotions.py forces the two
files' means to be exact negatives of each other, so between-condition
differences are partly an artifact of normalisation.

The claim being tested is about PERSISTENCE -- that weight updates
produce a shift which accumulates and survives, rather than a
context-window effect that resets. That is a claim about drift WITHIN a
run, and it has a clean control: the frozen child saw the same
caregiver, the same events, and the same gating, but its weights never
moved. It should not drift. If it does, the drift is not learning.

WHAT IT REPORTS
---------------
  - each condition's first vs last quartile, per concept
  - drift = late - early, per condition
  - the difference in drift between conditions (the actual result)
  - a per-episode trajectory, so gradual change can be told apart
    from a single step

Usage:
    python src/core/drift_analysis.py learn_on.jsonl learn_off.jsonl
    python src/core/drift_analysis.py learn_on.jsonl learn_off.jsonl --bins 6
"""

import argparse
import json
from collections import defaultdict

from fep_layer import CONCEPT_WEIGHTS, net_affect


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


def sem(xs):
    """Standard error, for judging whether a drift exceeds noise."""
    if len(xs) < 2:
        return 0.0
    return stdev(xs) / (len(xs) ** 0.5)


def quartile_drift(rows, frac=0.25):
    """Per-concept mean in the first and last `frac` of the run."""
    q = max(5, int(len(rows) * frac))
    early, late = rows[:q], rows[-q:]

    out = {}
    for c in list(CONCEPT_WEIGHTS) + ["_net"]:
        if c == "_net":
            e = [net_affect(r["activations"]) for r in early]
            l = [net_affect(r["activations"]) for r in late]
        else:
            e = [r["activations"][c] for r in early]
            l = [r["activations"][c] for r in late]
        out[c] = {
            "early": mean(e), "late": mean(l),
            "drift": mean(l) - mean(e),
            "sem": (sem(e) ** 2 + sem(l) ** 2) ** 0.5,
            "n": q,
        }
    return out


def episode_trajectory(rows, bins):
    """Mean net affect per bin of episodes, to see the shape of any
    change -- gradual accumulation looks different from a single step."""
    episodes = sorted({r["episode"] for r in rows})
    if not episodes:
        return []
    per_bin = max(1, len(episodes) // bins)
    grouped = defaultdict(list)
    for r in rows:
        idx = min(episodes.index(r["episode"]) // per_bin, bins - 1)
        grouped[idx].append(r)
    return [(i, len(grouped[i]),
             mean([net_affect(x["activations"]) for x in grouped[i]]))
            for i in sorted(grouped)]


def report(name, drift):
    print(f"\n{'=' * 74}")
    print(f"{name}")
    print(f"{'=' * 74}")
    print(f"\n{'concept':<14} {'early':>10} {'late':>10} {'drift':>10} "
          f"{'+-sem':>9} {'signif':>9}")
    print("-" * 74)
    for c, d in drift.items():
        label = "NET AFFECT" if c == "_net" else c
        # Rough: drift exceeding twice its combined standard error.
        sig = "yes" if abs(d["drift"]) > 2 * d["sem"] and d["sem"] > 0 else "-"
        print(f"{label:<14} {d['early']:>+10.3f} {d['late']:>+10.3f} "
              f"{d['drift']:>+10.3f} {d['sem']:>9.3f} {sig:>9}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("learning", help="affect file, weights UPDATED")
    ap.add_argument("frozen", help="affect file, weights FROZEN (control)")
    ap.add_argument("--bins", type=int, default=4,
                    help="episode bins for the trajectory")
    ap.add_argument("--frac", type=float, default=0.25,
                    help="fraction of the run treated as early/late")
    args = ap.parse_args()

    learn = load(args.learning)
    froz = load(args.frozen)

    d_learn = quartile_drift(learn, args.frac)
    d_froz = quartile_drift(froz, args.frac)

    report(f"LEARNING (weights updated)  --  {args.learning}", d_learn)
    report(f"FROZEN control (weights fixed)  --  {args.frozen}", d_froz)

    print(f"\n{'=' * 74}")
    print("DRIFT DIFFERENCE  (learning drift minus frozen drift)")
    print(f"{'=' * 74}")
    print("\nThe frozen child saw the same caregiver, the same events and")
    print("the same gating. Only the gradient steps differ, so this column")
    print("isolates what the weight updates did.")
    print(f"\n{'concept':<14} {'learning':>12} {'frozen':>12} "
          f"{'difference':>13}")
    print("-" * 74)
    for c in d_learn:
        label = "NET AFFECT" if c == "_net" else c
        dl, df = d_learn[c]["drift"], d_froz[c]["drift"]
        print(f"{label:<14} {dl:>+12.3f} {df:>+12.3f} {dl - df:>+13.3f}")

    print(f"\n{'=' * 74}")
    print("TRAJECTORY  (mean net affect by episode bin)")
    print(f"{'=' * 74}")
    print(f"\n{'bin':>5} {'n':>6} {'learning':>12} {'frozen':>12}")
    print("-" * 74)
    tl = episode_trajectory(learn, args.bins)
    tf = episode_trajectory(froz, args.bins)
    for (i, n1, m1), (_, _, m2) in zip(tl, tf):
        print(f"{i:>5} {n1:>6} {m1:>+12.3f} {m2:>+12.3f}")

    print(f"\n{'=' * 74}")
    print("READING THIS")
    print(f"{'=' * 74}")
    # Judge on the CONCEPTS, not on net affect. Net affect is a signed
    # weighted sum, so concepts drifting in opposite directions cancel:
    # in testing, an injected drift of -0.22 on protesting showed up
    # clearly per-concept while net affect read as "no drift".
    concept_drifts = {c: d_learn[c]["drift"] - d_froz[c]["drift"]
                      for c in CONCEPT_WEIGHTS}
    biggest = max(concept_drifts.items(), key=lambda kv: abs(kv[1]))
    n_moved = sum(1 for c, v in concept_drifts.items()
                  if abs(v) > 2 * d_learn[c]["sem"])

    print(f"\nconcepts drifting more than 2 sem beyond the control: "
          f"{n_moved}/{len(concept_drifts)}")
    print(f"largest: {biggest[0]} {biggest[1]:+.3f}")

    net_l = d_learn["_net"]["drift"]
    net_f = d_froz["_net"]["drift"]
    print(f"\nlearning child drifted  {net_l:+.3f} in net affect")
    print(f"frozen child drifted    {net_f:+.3f}")

    # Judge on CONCEPTS. The earlier version compared net affect first
    # and reported "the frozen child drifted more" on a run where all
    # four concepts had drifted beyond the control -- net affect is a
    # signed weighted sum, so comforted rising while protesting falls
    # cancels out in it.
    if n_moved == 0:
        print("\nNo concept drifted beyond the control. The updates were")
        print("too few or too small to move behaviour -- more episodes or")
        print("a higher learning rate is the next thing to vary.")
    else:
        control_moved = sum(
            1 for c in CONCEPT_WEIGHTS
            if abs(d_froz[c]["drift"]) > 2 * d_froz[c]["sem"]
            and d_froz[c]["sem"] > 0)
        print(f"concepts drifting in the FROZEN control: "
              f"{control_moved}/{len(CONCEPT_WEIGHTS)}")

        if control_moved >= n_moved:
            print("\nThe control drifted on as many concepts as the")
            print("learning condition. Something other than weight updates")
            print("produces drift across a run -- event ordering, the")
            print("standardiser warming up, or the mother's behaviour")
            print("varying by episode. Not attributable to learning.")
        else:
            print(f"\n{n_moved} concept(s) drifted in the learning "
                  f"condition against {control_moved} in the frozen")
            print("control, which saw the same caregiver, the same events")
            print("and the same gating. The difference is attributable to")
            print("the gradient steps.")
            print("\nSingle run, single seed. Three earlier findings in")
            print("this project looked this good and did not survive")
            print("repeated seeds. Replicate before claiming it.")

    print(f"\nnet affect drift: learning {net_l:+.3f}, "
          f"frozen {net_f:+.3f}")
    print("(net affect is a signed sum -- read the per-concept table)")


if __name__ == "__main__":
    main()