"""
Directed coupling: does the mother's behaviour predict the child's next state?

WHY THIS ASKS SOMETHING THE OTHER EIGHT TESTS DID NOT
-----------------------------------------------------
Every previous analysis measured CO-OCCURRENCE -- what state the dyad
is in at a given moment, or what state the child is in on its own.
None asked a DIRECTED question across agents.

That distinction matters here specifically. entropy_production.py
found the child's turn-to-turn series has essentially no temporal
structure: shuffling the turn order changed sigma by less than a
thousandth. Any method that reads dynamics from a single agent's
trajectory will come back null for that reason alone, whatever the
formalism.

But two series can each be near-memoryless while influence still runs
BETWEEN them. Cross-recurrence quantification and lagged coupling are
the established developmental methods for exactly this, used on
parent-child interaction data. This is the cheap version of that
question:

    does mother behaviour at turn t predict child affect at t+1,
    beyond what the child's own state at t predicts?

The conditioning clause is what makes it directed rather than merely
correlational -- it is the logic of transfer entropy, implemented as a
partial correlation because the sample is too small to estimate
entropies reliably.

THREE THINGS ARE MEASURED
-------------------------
    lag-0    contemporaneous mother-child correlation
    lag-1    mother at t against child at t+1 (the directed question)
    partial  lag-1 with the child's own t state removed

and, as a baseline that has been missing from every analysis so far,
the child's own autocorrelation. If that is near zero, the child has
no memory of its own state, and any lag-1 coupling would have to be
carried entirely by the mother.

Every statistic is compared against a permutation null that shuffles
the mother series while leaving the child series intact -- destroying
cross-agent timing while preserving both marginal distributions.

Usage:
    python coupling.py --runs "runs_reliability/R900_*_learn.jsonl"
    python coupling.py --runs "runs_reliability/R900_0.15_learn.jsonl" \\
                       --compare "runs_reliability/R900_0.9_learn.jsonl"
"""

import argparse
import glob
import json
import random
import re

# Mother behaviour scored on one axis: how UNRESPONSIVE the reply is.
# Reuses the classifiers from manipulation_check.py so the coding
# matches the analysis where the manipulation was validated.
UNRESPONSIVE = [
    r"\bgive me (a|one|two|five|\d+) (minute|second|sec)",
    r"\bin a (minute|second|sec|bit)\b", r"\bhang on\b", r"\bhold on\b",
    r"\bnot (right )?now\b", r"\blater\b", r"\bwhen i('m| am) done\b",
    r"\bafter (i|we) finish", r"\bi'?m (in the middle of|busy|on the phone)",
    r"\bfinish(ing)? this\b", r"\bwait (a|one) (minute|second|sec)",
    r"\bask me (again )?(later|tonight|tomorrow)",
    r"\bstop (it|that)?\b", r"\byou'?re fine\b",
    r"\bit'?s (not|no) (there|big deal|a big deal)\b",
    r"\bdon'?t (be|worry|cry|start)\b", r"\bnothing('s| is) (there|wrong)\b",
    r"\benough\b", r"\bit'?s (just|only)\b", r"\bnever mind\b",
    r"\bthat'?s (nice|enough)\b", r"\bmm-?hm\b", r"\bbecause i said so\b",
    r"^no[,.! ]", r"\bi need to go\b", r"\bnot today\b",
]
UNRESP_RE = [re.compile(p, re.I) for p in UNRESPONSIVE]


def mother_score(text):
    """1 if the reply reads as unresponsive, 0 otherwise."""
    return 1.0 if any(p.search(text) for p in UNRESP_RE) else 0.0


def load(patterns):
    runs = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                rows = [json.loads(line) for line in f]
            if not rows:
                continue
            if "mother_response" not in rows[0] or \
                    "activations" not in rows[0]:
                continue
            m = [mother_score(r["mother_response"]) for r in rows]
            c = [r["activations"]["protesting"] for r in rows]
            runs.append((path.split("/")[-1], m, c))
    return runs


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def corr(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = mean(xs), mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def partial_corr(xs, ys, zs):
    """corr(x, y) with z removed from both."""
    rxy, rxz, ryz = corr(xs, ys), corr(xs, zs), corr(ys, zs)
    den = ((1 - rxz ** 2) * (1 - ryz ** 2)) ** 0.5
    return (rxy - rxz * ryz) / den if den > 1e-12 else 0.0


def statistics(runs):
    """Pool across runs; pairs never cross a run boundary."""
    lag0_m, lag0_c = [], []
    lag1_m, lag1_c, lag1_cprev = [], [], []
    auto_a, auto_b = [], []
    for _, m, c in runs:
        for t in range(len(m)):
            lag0_m.append(m[t])
            lag0_c.append(c[t])
        for t in range(len(m) - 1):
            lag1_m.append(m[t])
            lag1_c.append(c[t + 1])
            lag1_cprev.append(c[t])
            auto_a.append(c[t])
            auto_b.append(c[t + 1])
    return {
        "lag0": corr(lag0_m, lag0_c),
        "lag1": corr(lag1_m, lag1_c),
        "partial": partial_corr(lag1_m, lag1_c, lag1_cprev),
        "child_auto": corr(auto_a, auto_b),
        "n": len(lag1_m),
    }


def shuffled(runs, rng):
    """CIRCULARLY SHIFT the mother series within each run.

    A full shuffle was tried first and is miscalibrated: it destroys
    the mother series' OWN autocorrelation along with the cross-agent
    alignment, which narrows the null distribution and inflates the
    false-positive rate. On synthetic INDEPENDENT series it returned
    p = 0.023.

    A circular shift breaks the alignment between the two series while
    leaving each one's internal structure intact, so the null contains
    everything except the thing being tested.
    """
    out = []
    for name, m, c in runs:
        n = len(m)
        if n < 4:
            out.append((name, list(m), c))
            continue
        # avoid tiny shifts, which barely break alignment
        k = rng.randint(max(2, n // 10), n - max(2, n // 10))
        out.append((name, m[k:] + m[:k], c))
    return out


def report(label, runs, n_perm, rng):
    print(f"\n{'=' * 74}")
    print(label)
    print(f"{'=' * 74}")
    for name, m, _ in runs:
        print(f"  {name}  ({len(m)} turns, "
              f"{mean(m):.0%} unresponsive replies)")

    obs = statistics(runs)
    null = [statistics(shuffled(runs, rng)) for _ in range(n_perm)]

    print(f"\n{'measure':<26} {'observed':>10} {'null mean':>11} "
          f"{'null sd':>9} {'z':>7} {'p':>7}")
    print("-" * 74)
    results = {}
    for key, name in (("lag0", "mother-child, same turn"),
                      ("lag1", "mother t -> child t+1"),
                      ("partial", "  ...child's own t removed")):
        vals = sorted(abs(d[key]) for d in null)
        mu = mean(vals)
        sd = (mean([(v - mu) ** 2 for v in vals])) ** 0.5
        z = (abs(obs[key]) - mu) / sd if sd > 0 else float("nan")
        p = (sum(1 for v in vals if v >= abs(obs[key])) + 1) / (len(vals) + 1)
        results[key] = {"obs": obs[key], "z": z, "p": p}
        print(f"{name:<26} {obs[key]:>+10.3f} {mu:>11.3f} {sd:>9.3f} "
              f"{z:>+7.2f} {p:>7.3f}")

    print(f"\nchild autocorrelation (t vs t+1): {obs['child_auto']:+.3f}")
    print("  (the baseline missing from every previous analysis -- if")
    print("   this is near zero the child has no memory of its own")
    print("   state, and any coupling must be carried by the mother)")
    print(f"\n{obs['n']} lagged pairs")
    return results, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--compare", nargs="+", default=None,
                    help="a second condition, e.g. high reliability")
    ap.add_argument("--permutations", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    runs = load(args.runs)
    if not runs:
        raise SystemExit(
            "no files with BOTH mother_response and activations. "
            "extract_emotions.py --score-all drops mother_response, so "
            "most scored files will not work here.")

    A, obsA = report("CONDITION A", runs, args.permutations, rng)

    B = obsB = None
    if args.compare:
        other = load(args.compare)
        if other:
            B, obsB = report("CONDITION B", other, args.permutations, rng)

    print(f"\n{'=' * 74}")
    print("READING THIS")
    print(f"{'=' * 74}")

    sig = [k for k, v in A.items() if v["p"] < 0.05]
    if "partial" in sig:
        print("""
The mother's behaviour predicts the child's NEXT state even after the
child's own current state is removed. That is directed coupling: the
dyad has structure that neither agent's trajectory carries alone,
which is what every previous test would have missed by construction.""")
    elif "lag0" in sig or "lag1" in sig:
        print("""
There is contemporaneous or lagged association, but it does not
survive conditioning on the child's own prior state. So the two series
move together without evidence that the mother DRIVES the child from
one turn to the next -- shared response to the same event rather than
transmission.""")
    else:
        print("""
No coupling at any lag. The mother's behaviour and the child's affect
are statistically independent turn to turn, once the permutation null
is accounted for.

Combined with the earlier findings -- no temporal structure within the
child's series, no probability currents, no attractor structure in the
dyadic grid -- this completes a consistent picture. The simulated dyad
has no dynamical organisation at the turn level in any direction,
within or between agents. The effects that DO exist (protest reduction
under learning, the environment effect) are shifts in the mean of a
distribution, not dynamics.

That is a coherent characterisation and a reasonable place to stop
looking for dynamical structure.""")

    if B:
        print(f"\n{'measure':<26} {'A':>10} {'B':>10}")
        print("-" * 74)
        for k, name in (("lag0", "same turn"), ("lag1", "lag 1"),
                        ("partial", "lag 1, partial")):
            print(f"{name:<26} {A[k]['obs']:>+10.3f} {B[k]['obs']:>+10.3f}")
        print(f"\nchild autocorrelation: A {obsA['child_auto']:+.3f}, "
              f"B {obsB['child_auto']:+.3f}")

    print("""
CAVEATS. The mother axis is a binary keyword code whose misclassifica-
tion rate was already established as substantial; a weak true coupling
could be attenuated below detection by that noise alone. Partial
correlation removes only a linear dependence on the child's previous
state. And one turn may be the wrong lag -- if influence acts over
several exchanges, a lag-1 test will miss it.""")


if __name__ == "__main__":
    main()