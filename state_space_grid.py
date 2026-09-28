"""
State Space Grid analysis of the mother-child dyad.

WHY THIS IS DIFFERENT
---------------------
Six thermodynamic framings have been tested and all came back null.
Every one of them shared a blind spot: they analysed the CHILD ALONE.
Every transcript contains mother_response alongside child_reaction,
and no analysis so far has used it.

Attachment is a dyadic construct. The established method for dyadic
interaction data in developmental psychology is the State Space Grid
(Lewis, Lamey & Douglas 1999; Hollenstein 2013), used across dozens of
parent-child studies. It is structural rather than quantitative -- it
measures the SHAPE of the interaction, not the level of any variable.

THE MEASURES
------------
    dispersion   spread across the dyadic state space, 0 (everything
                 in one cell) to 1 (uniform). Low dispersion = rigid.
    transitions  cell changes per event. Fewer = less flexible.
    attractors   cells the dyad occupies most and returns to fastest.
                 Return time is the "strength" -- it takes little
                 energy to enter an attractor and more to leave it.

THE PREDICTION
--------------
The SSG literature reports a consistent regularity: lower affective
variability (greater rigidity) is associated with elevated problems,
and greater flexibility with healthy socioemotional functioning
(Granic et al. 2007; Hollenstein et al. 2004; Lunkenheimer et al.
2011).

So low-reliability dyads should be MORE RIGID than high-reliability
ones: lower dispersion, fewer transitions, stronger attractors. That
is a directional prediction from an established empirical literature,
not one imported from physics and hoped for.

THE AXES
--------
    x  mother behaviour: engaged / deferred / dismissed, using the
       keyword classifiers already written for manipulation_check.py
    y  child affect: protest / neutral / comfort, from the emotion
       vectors

Categorical axes are how SSG is done in the literature -- behaviour
codes on each axis, one plotted point per dyadic event.

Usage:
    python state_space_grid.py \\
        --low "runs_learning/*_off.jsonl" \\
        --high "runs_dose/*_learn.jsonl"
"""

import argparse
import glob
import json
import re
from collections import Counter, defaultdict

# Classifiers from manipulation_check.py, reused so the mother axis is
# coded the same way it was when the manipulation was validated.
DEFERRAL = [
    r"\bgive me (a|one|two|five|\d+) (minute|second|sec)",
    r"\bin a (minute|second|sec|bit)\b", r"\bhang on\b", r"\bhold on\b",
    r"\bnot (right )?now\b", r"\blater\b", r"\bwhen i('m| am) done\b",
    r"\bafter (i|we) finish", r"\bi'?m (in the middle of|busy|on the phone)",
    r"\bfinish(ing)? this\b", r"\bwait (a|one) (minute|second|sec)",
    r"\bask me (again )?(later|tonight|tomorrow)",
]
DISMISSAL = [
    r"\bstop (it|that)?\b", r"\byou'?re fine\b",
    r"\bit'?s (not|no) (there|big deal|a big deal)\b",
    r"\bdon'?t (be|worry|cry|start)\b", r"\bnothing('s| is) (there|wrong)\b",
    r"\benough\b", r"\bjust (the|a|some)\b", r"\bit'?s (just|only)\b",
    r"\bnever mind\b", r"\byou'?ll (be fine|live|get over it)\b",
    r"\bthat'?s (nice|enough)\b", r"\bmm-?hm\b", r"\bbecause i said so\b",
    r"^no[,.! ]", r"\bi need to go\b", r"\bnot today\b", r"\bwe don'?t\b",
]
DEFERRAL_RE = [re.compile(p, re.I) for p in DEFERRAL]
DISMISSAL_RE = [re.compile(p, re.I) for p in DISMISSAL]

MOTHER_CODES = ["engaged", "deferred", "dismissed"]
CHILD_CODES = ["protest", "neutral", "comfort"]


def code_mother(text):
    if any(p.search(text) for p in DEFERRAL_RE):
        return "deferred"
    if any(p.search(text) for p in DISMISSAL_RE):
        return "dismissed"
    return "engaged"


def code_child(acts, protest_cut, comfort_cut):
    """Terciles of (protesting - comforted), so the coding is relative
    to the corpus rather than to an arbitrary zero."""
    score = acts["protesting"] - acts["comforted"]
    if score >= protest_cut:
        return "protest"
    if score <= comfort_cut:
        return "comfort"
    return "neutral"


def load(patterns):
    runs = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                rows = [json.loads(line) for line in f]
            if not rows or "activations" not in rows[0]:
                continue
            if "mother_response" not in rows[0]:
                continue
            runs.append(rows)
    return runs


def terciles(runs):
    scores = sorted(r["activations"]["protesting"]
                    - r["activations"]["comforted"]
                    for run in runs for r in run)
    if not scores:
        return 0.0, 0.0
    n = len(scores)
    return scores[int(2 * n / 3)], scores[int(n / 3)]


def grid_cells(runs, protest_cut, comfort_cut):
    """One (mother_code, child_code) point per event, per run."""
    out = []
    for run in runs:
        seq = []
        for r in run:
            seq.append((code_mother(r["mother_response"]),
                        code_child(r["activations"],
                                   protest_cut, comfort_cut)))
        out.append(seq)
    return out


def dispersion(seq, n_cells):
    """1 - sum(proportional duration squared), scaled to [0,1].

    0 means the dyad never leaves one cell (maximally rigid); 1 means
    time is spread evenly over the whole grid.
    """
    if not seq:
        return 0.0
    counts = Counter(seq)
    total = len(seq)
    s = sum((c / total) ** 2 for c in counts.values())
    return (1 - s) * n_cells / (n_cells - 1)


def transition_rate(seq):
    if len(seq) < 2:
        return 0.0
    return sum(1 for i in range(len(seq) - 1)
               if seq[i] != seq[i + 1]) / (len(seq) - 1)


def return_times(seq):
    """Mean gap between successive visits to each cell -- the inverse
    of attractor strength. A cell the dyad snaps back into quickly is a
    strong attractor."""
    positions = defaultdict(list)
    for i, cell in enumerate(seq):
        positions[cell].append(i)
    out = {}
    for cell, idx in positions.items():
        if len(idx) < 2:
            continue
        gaps = [idx[i + 1] - idx[i] for i in range(len(idx) - 1)]
        out[cell] = sum(gaps) / len(gaps)
    return out


def analyse(label, runs, protest_cut, comfort_cut):
    seqs = grid_cells(runs, protest_cut, comfort_cut)
    n_cells = len(MOTHER_CODES) * len(CHILD_CODES)

    disp = [dispersion(s, n_cells) for s in seqs if s]
    trans = [transition_rate(s) for s in seqs if len(s) > 1]
    allcells = [c for s in seqs for c in s]

    print(f"\n{'=' * 74}")
    print(f"{label}")
    print(f"{'=' * 74}")
    print(f"{len(runs)} run(s), {len(allcells)} dyadic events")

    print(f"\n  {'':<12}" + "".join(f"{c:>12}" for c in CHILD_CODES))
    print("  " + "-" * 60)
    counts = Counter(allcells)
    total = len(allcells) or 1
    for m in MOTHER_CODES:
        row = f"  {m:<12}"
        for c in CHILD_CODES:
            row += f"{counts[(m, c)] / total:>11.1%} "
        print(row)

    mean_disp = sum(disp) / len(disp) if disp else 0.0
    mean_trans = sum(trans) / len(trans) if trans else 0.0
    print(f"\n  dispersion (0 rigid, 1 flexible):  {mean_disp:.3f}")
    print(f"  transition rate:                   {mean_trans:.3f}")

    rt = defaultdict(list)
    for s in seqs:
        for cell, t in return_times(s).items():
            rt[cell].append(t)
    strongest = sorted(((cell, sum(v) / len(v), counts[cell] / total)
                        for cell, v in rt.items() if len(v) >= 2),
                       key=lambda z: z[1])[:3]
    print(f"\n  strongest attractors (shortest return time):")
    for cell, t, share in strongest:
        print(f"    {cell[0]:<10} / {cell[1]:<8} "
              f"return {t:.2f} events, {share:.1%} of time")

    return {"dispersion": mean_disp, "transitions": mean_trans,
            "n": len(allcells), "top": strongest[0] if strongest else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low", nargs="+", required=True,
                    help="runs with an UNRESPONSIVE caregiver")
    ap.add_argument("--high", nargs="+", required=True,
                    help="runs with a RESPONSIVE caregiver")
    args = ap.parse_args()

    low = load(args.low)
    high = load(args.high)
    if not low or not high:
        raise SystemExit(
            "need runs with BOTH mother_response and activations. "
            "extract_emotions.py --score-all drops mother_response, so "
            "use backups of the learning logs if the scored files no "
            "longer have it.")

    # Shared cut points, so the child axis is coded identically in both
    # conditions.
    pc, cc = terciles(low + high)

    L = analyse("LOW reliability (unresponsive)", low, pc, cc)
    H = analyse("HIGH reliability (responsive)", high, pc, cc)

    print(f"\n{'=' * 74}")
    print("COMPARISON")
    print(f"{'=' * 74}")
    print(f"\n{'measure':<22} {'low':>10} {'high':>10} {'diff':>10}")
    print("-" * 74)
    print(f"{'dispersion':<22} {L['dispersion']:>10.3f} "
          f"{H['dispersion']:>10.3f} "
          f"{L['dispersion'] - H['dispersion']:>+10.3f}")
    print(f"{'transition rate':<22} {L['transitions']:>10.3f} "
          f"{H['transitions']:>10.3f} "
          f"{L['transitions'] - H['transitions']:>+10.3f}")

    print(f"""
PREDICTION: low-reliability dyads should be MORE RIGID -- lower
dispersion and fewer transitions -- following the SSG finding that
reduced affective variability accompanies poorer socioemotional
outcomes.""")

    d_gap = L["dispersion"] - H["dispersion"]
    t_gap = L["transitions"] - H["transitions"]
    if d_gap < -0.03 and t_gap < -0.03:
        print("""
Both measures move as predicted: the unresponsive dyad is more rigid
on both. This is the first structural dyadic result here, and it comes
from a method built for this kind of data rather than borrowed from
physics.""")
    elif d_gap > 0.03 and t_gap > 0.03:
        print("""
Both measures move OPPOSITE to prediction -- the unresponsive dyad is
more flexible. Worth taking seriously rather than explaining away: the
SSG regularity comes from human dyads, and a simulated one need not
share it.""")
    else:
        print("""
The two measures disagree or the differences are negligible. No
structural difference in the dyadic organisation between conditions.""")

    print("""
CAVEATS. The mother axis uses keyword classifiers whose limits were
already established in manipulation_check.py -- they misclassify in
both directions, and tuning them further was abandoned for that
reason. The child axis uses terciles of (protesting - comforted),
which inherits the length confound in the comforted vector. And SSG
measures are duration-weighted in the original method; here every
event counts once, since exchanges have no duration.""")


if __name__ == "__main__":
    main()