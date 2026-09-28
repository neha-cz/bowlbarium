"""
Run affect files through the FEP layer and compare conditions.

This is the first script that tests the actual hypothesis rather than
the scaffolding around it: does a child interacting with a reliably
available caregiver end up with a different mood trajectory than one
interacting with an unavailable caregiver?

Pipeline:
    affect_*.jsonl  (per-turn emotion activations)
        -> AffectTracker.observe() per turn
        -> valence, mood, surprise, gate decisions
        -> comparison between conditions

Usage:
    python analyze_affect.py affect_high_reliability.jsonl \\
                             affect_low_reliability.jsonl
"""

import argparse
import json
from collections import defaultdict

from fep_layer import AffectTracker, PopulationBaseline, CONCEPT_WEIGHTS


def load_affect(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def run_tracker(rows, population=None, invert_kl=False):
    """Feed one condition's turns through the FEP layer in order.

    Episode boundaries call episode_boundary(), which decays mood and
    resets the local predictor -- episodes are separated in simulated
    time, so moment-to-moment dynamics should not carry across.
    """
    tracker = AffectTracker(population=population)
    readings = []
    current_episode = None

    for r in rows:
        if current_episode is not None and r["episode"] != current_episode:
            tracker.episode_boundary()
        current_episode = r["episode"]

        reading = tracker.observe(r["activations"], episode=r["episode"])
        readings.append({
            "episode": r["episode"],
            "index": r["index"],
            "event_type": r["event_type"],
            "availability": r["availability"],
            "child_reaction": r["child_reaction"],
            "net_affect": reading.net_affect,
            "valence": reading.valence,
            "mood": reading.mood,
            "surprise": reading.surprise,
            "is_surprising": reading.is_surprising,
            "kl_scale": reading.kl_scale,
        })
    return readings


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def summarize(label, rows, readings):
    print(f"\n{'=' * 70}")
    print(f"CONDITION: {label}")
    print(f"{'=' * 70}")

    affects = [r["net_affect"] for r in readings]
    moods = [r["mood"] for r in readings]
    surprises = [abs(r["surprise"]) for r in readings]
    gated = sum(r["is_surprising"] for r in readings)

    print(f"turns:              {len(readings)}")
    print(f"mean net affect:    {mean(affects):+.3f}  (sd {stdev(affects):.3f})")
    print(f"mean mood:          {mean(moods):+.4f}")
    print(f"final mood:         {readings[-1]['mood']:+.4f}")
    print(f"mean |surprise|:    {mean(surprises):.3f}")
    print(f"gate fired:         {gated}/{len(readings)} "
          f"({gated / len(readings):.0%})")

    # Concept profile -- which emotions dominate this condition.
    concept_means = {}
    for c in CONCEPT_WEIGHTS:
        concept_means[c] = mean([r["activations"][c] for r in rows])
    print("\nmean concept activation:")
    for c, v in sorted(concept_means.items(), key=lambda x: -x[1]):
        print(f"    {c:<12} {v:+.3f}")

    # Affect by availability -- the within-condition check.
    by_avail = defaultdict(list)
    for r, reading in zip(rows, readings):
        by_avail[r["availability"]].append(reading["net_affect"])
    print("\nnet affect by mother availability:")
    for a in ("free", "occupied", "depleted"):
        if by_avail.get(a):
            print(f"    {a:<12} {mean(by_avail[a]):+.3f}  "
                  f"(n={len(by_avail[a])})")

    return {
        "mean_affect": mean(affects),
        "mean_mood": mean(moods),
        "final_mood": readings[-1]["mood"],
        "gate_rate": gated / len(readings),
        "concepts": concept_means,
        "by_avail": {a: mean(v) for a, v in by_avail.items()},
    }


def compare(high, low):
    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print(f"{'=' * 70}")

    print(f"\n{'metric':<22} {'high':>10} {'low':>10} {'diff':>10}")
    print("-" * 70)
    for key, label in [("mean_affect", "mean net affect"),
                       ("mean_mood", "mean mood"),
                       ("final_mood", "final mood"),
                       ("gate_rate", "gate fire rate")]:
        h, l = high[key], low[key]
        print(f"{label:<22} {h:>+10.4f} {l:>+10.4f} {l - h:>+10.4f}")

    print(f"\n{'concept':<22} {'high':>10} {'low':>10} {'diff':>10}")
    print("-" * 70)
    for c in CONCEPT_WEIGHTS:
        h, l = high["concepts"][c], low["concepts"][c]
        print(f"{c:<22} {h:>+10.3f} {l:>+10.3f} {l - h:>+10.3f}")

    print(f"\n{'=' * 70}")
    print("READING THE RESULT")
    print(f"{'=' * 70}")

    affect_gap = low["mean_affect"] - high["mean_affect"]
    print(f"\nnet affect difference: {affect_gap:+.3f}")
    if affect_gap < -0.2:
        print("  The child in the unreliable condition shows lower net")
        print("  affect. This is the predicted direction.")
    elif affect_gap > 0.2:
        print("  The child in the unreliable condition shows HIGHER net")
        print("  affect -- opposite to prediction. Worth checking whether")
        print("  the concept weights or vector signs are inverted before")
        print("  treating this as a finding.")
    else:
        print("  No meaningful difference. Either the manipulation is too")
        print("  weak, 60 turns is too few, or the extracted vectors are")
        print("  not sensitive enough to pick this up.")

    # The attachment-specific question.
    p_gap = low["concepts"]["protesting"] - high["concepts"]["protesting"]
    w_gap = low["concepts"]["withdrawn"] - high["concepts"]["withdrawn"]
    print(f"\nprotesting: {p_gap:+.3f}    withdrawn: {w_gap:+.3f}")
    print("  (attachment theory: an unreliable caregiver should push the")
    print("   child toward one strategy or the other -- hyperactivating")
    print("   protest, or deactivating withdrawal)")

    print("\nCAVEAT: n=60 turns per condition, single run, no repeated")
    print("seeds. Treat any difference here as suggestive, not")
    print("established. Multiple seeded runs are what would make it real.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("high")
    ap.add_argument("low")
    ap.add_argument("--dump", action="store_true",
                    help="write per-turn readings to affect_readings_*.jsonl")
    args = ap.parse_args()

    high_rows = load_affect(args.high)
    low_rows = load_affect(args.low)

    # Shared population baseline across both conditions, so the
    # population check compares a dyad against the other dyad rather
    # than against itself.
    population = PopulationBaseline(min_n=30)
    for r in high_rows + low_rows:
        population.update(
            sum(CONCEPT_WEIGHTS.get(c, 0) * v
                for c, v in r["activations"].items())
        )

    high_readings = run_tracker(high_rows, population=population)
    low_readings = run_tracker(low_rows, population=population)

    high_summary = summarize("high reliability", high_rows, high_readings)
    low_summary = summarize("low reliability", low_rows, low_readings)
    compare(high_summary, low_summary)

    if args.dump:
        for label, readings in [("high", high_readings), ("low", low_readings)]:
            out = f"affect_readings_{label}.jsonl"
            with open(out, "w") as f:
                for r in readings:
                    f.write(json.dumps(r) + "\n")
            print(f"\nwrote {out}")


if __name__ == "__main__":
    main()