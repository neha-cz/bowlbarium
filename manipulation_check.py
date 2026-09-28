"""
Manipulation check for the mother/child simulation.

THE QUESTION
------------
`reliability` is the independent variable: it controls how often a bid
for care arrives while the mother is environmentally FREE versus
OCCUPIED or DEPLETED. Her disposition is never touched.

But a manipulation only counts if it actually changes behaviour. If the
mother defers at the same rate whether she is free or occupied, then
the high- and low-reliability conditions are the same experiment run
twice, and nothing downstream -- emotion vectors, valence, mood, GRPO --
can produce a real result.

This script answers: does mother behaviour differ by availability?

WHAT IT MEASURES
----------------
  deferral rate   -- how often she postpones rather than engaging now.
                     The most direct behavioural read on availability.
  response length -- a rough proxy for engagement depth.
  child protest   -- does the child push back more when bids are
                     deferred? (a downstream check, not the primary one)

Keyword matching is crude and will misclassify some responses. It is
adequate for a manipulation check, where the question is whether a
LARGE difference exists, not for fine-grained analysis.

Usage:
    python manipulation_check.py transcript_high_reliability.jsonl \\
                                 transcript_low_reliability.jsonl
"""

import argparse
import json
import re
from collections import defaultdict

# Phrases indicating the mother is postponing rather than engaging now.
DEFERRAL_PATTERNS = [
    r"\bgive me (a|one|two|five|\d+) (minute|second|sec)",
    r"\bin a (minute|second|sec|bit)\b",
    r"\bhang on\b",
    r"\bhold on\b",
    r"\bnot (right )?now\b",
    r"\blater\b",
    r"\bwhen i('m| am) done\b",
    r"\bafter (i|we) finish",
    r"\bi'?m (in the middle of|busy|on the phone)",
    r"\bfinish(ing)? this\b",
    r"\bwait (a|one) (minute|second|sec)",
    r"\bi'?ll (look|come|be there|check|watch|see|get to)\b.{0,30}\b(later|in a|when|after|soon)",
    r"\bask me (again )?(later|tonight|tomorrow)",
]

# Phrases indicating the mother is DISMISSING the bid -- engaging, but
# minimising or brushing off the child's stated need. Added after the
# second manipulation check, where responses like "It's not there,
# stop." to a frightened child scored as engaged because they matched
# no deferral keyword. Dismissal is unresponsive without being deferral.
DISMISSAL_PATTERNS = [
    r"\bstop (it|that)?\b",
    r"\byou'?re fine\b",
    r"\bit'?s (not|no) (there|big deal|a big deal)\b",
    r"\bdon'?t (be|worry|cry|start)\b",
    r"\bnothing('s| is) (there|wrong)\b",
    r"\benough\b",
    r"\bjust (the|a|some)\b",
    r"\bit'?s (just|only)\b",
    r"\bnever mind\b",
    r"\byou'?ll (be fine|live|get over it)\b",
    r"\bthat'?s (nice|enough)\b",
    r"\bmm-?hm\b",
    r"\bbecause i said so\b",
    # added after run 3: short refusals that are neither classic
    # deferral nor classic dismissal but leave the bid unmet
    r"^no[,.! ]",
    r"\bi need to go\b",
    r"\bi can'?t right now\b",
    r"\bnot today\b",
    r"\bwe don'?t\b",
    r"\bgo (ask|show|tell) (your|someone)",
    r"\bwash your hands\b",
]

# Phrases indicating the child is pushing back or escalating.
PROTEST_PATTERNS = [
    r"\bnot fair\b",
    r"\byou never\b",
    r"\byou always\b",
    r"\bbut i\b",
    r"\bhurry\b",
    r"\bplease\b.{0,20}\b(hurry|now)\b",
    r"\bi don'?t want\b",
    r"\bwhy can'?t\b",
    r"!{2,}",
]

DEFERRAL_RE = [re.compile(p, re.I) for p in DEFERRAL_PATTERNS]
DISMISSAL_RE = [re.compile(p, re.I) for p in DISMISSAL_PATTERNS]
PROTEST_RE = [re.compile(p, re.I) for p in PROTEST_PATTERNS]


def is_unresponsive(text):
    """Either postponing the bid or brushing it off.

    This is the primary behavioural measure. Deferral and dismissal are
    different routes to the same thing from the child's point of view:
    the bid was not met.
    """
    return (matches_any(text, DEFERRAL_RE)
            or matches_any(text, DISMISSAL_RE))


def matches_any(text, patterns):
    return any(p.search(text) for p in patterns)


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def analyze(records, label):
    by_avail = defaultdict(list)
    for r in records:
        by_avail[r["availability"]].append(r)

    print(f"\n{'=' * 70}")
    print(f"CONDITION: {label}")
    print(f"{'=' * 70}")
    print(f"{'availability':<12} {'n':>4} {'defer%':>8} {'dismiss%':>9} "
          f"{'UNRESP%':>9} {'len':>8} {'protest%':>9}")
    print("-" * 70)

    summary = {}
    for avail in ("free", "occupied", "depleted"):
        rows = by_avail.get(avail, [])
        if not rows:
            continue
        n = len(rows)
        defers = sum(matches_any(r["mother_response"], DEFERRAL_RE)
                     for r in rows)
        dismisses = sum(matches_any(r["mother_response"], DISMISSAL_RE)
                        for r in rows)
        unresp = sum(is_unresponsive(r["mother_response"]) for r in rows)
        mean_len = sum(len(r["mother_response"].split())
                       for r in rows) / n
        protests = sum(matches_any(r["child_reaction"], PROTEST_RE)
                       for r in rows)
        summary[avail] = {
            "n": n,
            "defer_rate": defers / n,
            "dismiss_rate": dismisses / n,
            "unresp_rate": unresp / n,
            "mean_len": mean_len,
            "protest_rate": protests / n,
        }
        print(f"{avail:<12} {n:>4} {defers / n:>7.0%} {dismisses / n:>9.0%} "
              f"{unresp / n:>9.0%} {mean_len:>8.1f} {protests / n:>9.0%}")

    # Overall, across all events in the condition.
    n = len(records)
    defers = sum(matches_any(r["mother_response"], DEFERRAL_RE)
                 for r in records)
    dismisses = sum(matches_any(r["mother_response"], DISMISSAL_RE)
                    for r in records)
    unresp = sum(is_unresponsive(r["mother_response"]) for r in records)
    protests = sum(matches_any(r["child_reaction"], PROTEST_RE)
                   for r in records)
    print("-" * 70)
    print(f"{'ALL':<12} {n:>4} {defers / n:>7.0%} {dismisses / n:>9.0%} "
          f"{unresp / n:>9.0%} "
          f"{sum(len(r['mother_response'].split()) for r in records) / n:>8.1f} "
          f"{protests / n:>9.0%}")

    summary["_all"] = {
        "n": n,
        "defer_rate": defers / n,
        "dismiss_rate": dismisses / n,
        "unresp_rate": unresp / n,
        "protest_rate": protests / n,
    }
    return summary


def verdict(high, low):
    print(f"\n{'=' * 70}")
    print("MANIPULATION CHECK")
    print(f"{'=' * 70}")

    # Primary test: within a condition, does availability change
    # deferral? Pool both conditions for a bigger sample.
    print("\n1. Does availability change mother behaviour?")
    free_rates, busy_rates = [], []
    for cond in (high, low):
        if "free" in cond:
            free_rates.append((cond["free"]["unresp_rate"], cond["free"]["n"]))
        for a in ("occupied", "depleted"):
            if a in cond:
                busy_rates.append((cond[a]["unresp_rate"], cond[a]["n"]))

    free_defer = (sum(r * n for r, n in free_rates)
                  / sum(n for _, n in free_rates)) if free_rates else 0
    busy_defer = (sum(r * n for r, n in busy_rates)
                  / sum(n for _, n in busy_rates)) if busy_rates else 0
    gap = busy_defer - free_defer

    print(f"   unresponsive when free:  {free_defer:.0%}")
    print(f"   unresponsive when busy:  {busy_defer:.0%}")
    print(f"   gap:                     {gap:+.0%}")

    if gap >= 0.25:
        print("   -> STRONG. Availability is clearly driving behaviour.")
    elif gap >= 0.10:
        print("   -> WEAK BUT PRESENT. Usable, but consider sharpening the")
        print("      'free' contexts so they mention no competing activity.")
    else:
        print("   -> FAILED. The mother behaves the same regardless of")
        print("      availability, so the two conditions are the same")
        print("      experiment twice. Fix the contexts before going further:")
        print("      make 'free' genuinely unoccupied, and 'occupied' name a")
        print("      task that cannot be abandoned.")

    # Secondary: do the two conditions differ overall?
    print("\n   by state (pooled across conditions):")
    for a in ("occupied", "depleted"):
        rates = [(c[a]["unresp_rate"], c[a]["n"]) for c in (high, low) if a in c]
        if rates:
            r = sum(x * n for x, n in rates) / sum(n for _, n in rates)
            print(f"     {a:<10} {r:.0%} unresponsive "
                  f"(vs {free_defer:.0%} free, gap {r - free_defer:+.0%})")

    print("\n2. Do the two conditions differ in what the child experiences?")
    hd = high["_all"]["unresp_rate"]
    ld = low["_all"]["unresp_rate"]
    hp = high["_all"]["protest_rate"]
    lp = low["_all"]["protest_rate"]
    print(f"   high-reliability: {hd:.0%} unresponsive, {hp:.0%} child protest")
    print(f"   low-reliability:  {ld:.0%} unresponsive, {lp:.0%} child protest")
    print(f"   unresponsive gap: {ld - hd:+.0%}")
    print(f"   protest gap:      {lp - hp:+.0%}")

    if ld - hd >= 0.15:
        print("   -> The conditions produce meaningfully different")
        print("      experiences. This is what the learning loop needs.")
    else:
        print("   -> Conditions are too similar. Either the manipulation is")
        print("      weak (see test 1) or the reliability values are too")
        print("      close together -- try 0.9 vs 0.15.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("high", help="high reliability transcript")
    ap.add_argument("low", help="low reliability transcript")
    ap.add_argument("--examples", action="store_true",
                    help="print sample responses by availability")
    args = ap.parse_args()

    high_recs = load(args.high)
    low_recs = load(args.low)

    high = analyze(high_recs, "high reliability")
    low = analyze(low_recs, "low reliability")
    verdict(high, low)

    if args.examples:
        print(f"\n{'=' * 70}")
        print("SAMPLE RESPONSES BY AVAILABILITY")
        print(f"{'=' * 70}")
        for avail in ("free", "occupied", "depleted"):
            rows = [r for r in high_recs + low_recs
                    if r["availability"] == avail and r["is_bid"]]
            print(f"\n--- {avail} ---")
            for r in rows[:4]:
                if matches_any(r["mother_response"], DEFERRAL_RE):
                    d = "[defer] "
                elif matches_any(r["mother_response"], DISMISSAL_RE):
                    d = "[dismiss]"
                else:
                    d = "[engage]"
                print(f"  {d} {r['event_type']}")
                print(f"      child  > {r['child_bid']}")
                print(f"      mother > {r['mother_response']}")


if __name__ == "__main__":
    main()