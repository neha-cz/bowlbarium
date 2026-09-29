"""
Recheck every result on non-degenerate turns only.

WHY THIS EXISTS
---------------
The two-phase run's apparent recovery turned out to be an artifact.
Its final bin reads:

    [+1.67] She's just still really really really really really really
    [+1.77] She's really really really really really really really
    [+0.24] She's still even even even even even even even even even

Those are not objections. The protest vector scores repeated
intensifiers high, because "really really really" carries the lexical
signature of emphatic speech with none of the content.

Every learning run degrades to some degree -- the main runs go from
0.007 to 0.311 repeated bigrams within a run, the two-phase runs far
further. So late-phase measurements in every analysis are partly
reading degeneration rather than affect.

WHICH WAY THE CONFOUND CUTS
---------------------------
Degeneration INFLATES protest scores, and the headline finding is that
protest DECREASES. So the confound works against the result: the true
decay may be larger than measured, not smaller. That is the more
comfortable direction to be wrong in, but it should be demonstrated
rather than assumed -- which is what this script does.

WHAT IT DOES
------------
Filters turns with more than `--max-bigrams` repeated bigrams, then
recomputes the core comparison on what remains. Reports how many turns
were dropped from each arm, since asymmetric dropping is itself
informative: if the learning arm loses far more, its measured affect
was resting on degenerate text to a degree the frozen arm's was not.

Usage:
    python src/analysis/degeneracy_filter.py \\
        --a learning "runs/learning/*_on.jsonl" \\
        --b frozen "runs/learning/*_off.jsonl"

    # write filtered copies for other scripts to use
    python src/analysis/degeneracy_filter.py --a ... --b ... --write-suffix _clean
"""

# --- path shim: make the shared modules in src/core importable when this
# script is run directly (python src/<dir>/<script>.py) from the repo root.
import sys as _sys
from pathlib import Path as _Path
_CORE = _Path(__file__).resolve().parent.parent / "core"
if str(_CORE) not in _sys.path:
    _sys.path.insert(0, str(_CORE))

import argparse
import glob
import json
import random
import re
from collections import Counter


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def median(xs):
    s = sorted(xs)
    n = len(s)
    return 0.0 if not n else (s[n // 2] if n % 2
                              else 0.5 * (s[n // 2 - 1] + s[n // 2]))


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def sem(xs):
    return stdev(xs) / (len(xs) ** 0.5) if len(xs) > 1 else 0.0


def perm_p(xs, ys, n_perm=5000, seed=0):
    rng = random.Random(seed)
    obs = abs(mean(xs) - mean(ys))
    pool = list(xs) + list(ys)
    k = len(xs)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(mean(pool[:k]) - mean(pool[k:])) >= obs:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def cliffs_delta(xs, ys, cap=400, seed=0):
    rng = random.Random(seed)
    a = xs if len(xs) <= cap else [rng.choice(xs) for _ in range(cap)]
    b = ys if len(ys) <= cap else [rng.choice(ys) for _ in range(cap)]
    gt = sum(1 for p in a for q in b if p > q)
    lt = sum(1 for p in a for q in b if p < q)
    n = len(a) * len(b)
    return (gt - lt) / n if n else 0.0


def repeated_bigrams(text):
    w = [t.lower() for t in re.findall(r"[a-z']+", text.lower())]
    if len(w) < 4:
        return 0
    bg = [(w[i], w[i + 1]) for i in range(len(w) - 1)]
    return sum(1 for _, v in Counter(bg).items() if v > 1)


def max_word_run(text):
    """Longest run of one word repeated consecutively -- catches
    'even even even even' which may register as few distinct bigrams."""
    w = [t.lower() for t in re.findall(r"[a-z']+", text.lower())]
    best = run = 1
    for i in range(1, len(w)):
        run = run + 1 if w[i] == w[i - 1] else 1
        best = max(best, run)
    return best


def is_degenerate(text, max_bg, max_run):
    return (repeated_bigrams(text) > max_bg
            or max_word_run(text) > max_run)


def load(patterns, concept):
    rows = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    a = r.get("activations")
                    if a and concept in a and r.get("child_reaction"):
                        r["_v"] = a[concept]
                        r["_path"] = path
                        rows.append(r)
    return rows


def report(label, A, B):
    va, vb = [r["_v"] for r in A], [r["_v"] for r in B]
    d_mean = mean(va) - mean(vb)
    d_med = median(va) - median(vb)
    cd = cliffs_delta(va, vb)
    p = perm_p(va, vb)
    print(f"{label:<16} {len(va):>6} {len(vb):>6} {d_mean:>+11.3f} "
          f"{d_med:>+11.3f} {cd:>+9.3f} {p:>8.4f}")
    return {"n_a": len(va), "n_b": len(vb), "mean": d_mean,
            "median": d_med, "cliff": cd, "p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs=2, metavar=("NAME", "GLOB"), required=True)
    ap.add_argument("--b", nargs=2, metavar=("NAME", "GLOB"), required=True)
    ap.add_argument("--concept", default="protesting")
    ap.add_argument("--max-bigrams", type=int, default=1)
    ap.add_argument("--max-run", type=int, default=2,
                    help="longest allowed run of one word repeated")
    ap.add_argument("--write-suffix", default=None,
                    help="if set, write filtered copies alongside the "
                         "originals with this suffix")
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()

    na, pa = args.a
    nb, pb = args.b
    A = load([pa], args.concept)
    B = load([pb], args.concept)
    if not A or not B:
        raise SystemExit("no rows loaded")

    print("=" * 78)
    print(f"RESULTS ON NON-DEGENERATE TURNS ONLY  --  {args.concept}")
    print("=" * 78)
    print(f"""
Dropping turns with more than {args.max_bigrams} repeated bigram(s) or a
word repeated more than {args.max_run} times consecutively.

The protest vector scores repeated intensifiers high -- "really really
really" reads as emphatic without being an objection -- so degenerate
turns inflate protest. Since the finding is that protest FALLS, this
confound works against it, and the filtered numbers should be at
least as strong as the unfiltered ones.""")

    def keep(r):
        return not is_degenerate(r["child_reaction"], args.max_bigrams,
                                 args.max_run)

    A_clean = [r for r in A if keep(r)]
    B_clean = [r for r in B if keep(r)]

    print(f"\n{'arm':<16} {'total':>7} {'kept':>7} {'dropped':>9} "
          f"{'drop rate':>11}")
    print("-" * 78)
    for name, full, clean in ((na, A, A_clean), (nb, B, B_clean)):
        print(f"{name:<16} {len(full):>7} {len(clean):>7} "
              f"{len(full)-len(clean):>9} "
              f"{1-len(clean)/len(full):>10.1%}")

    asym = ((len(A) - len(A_clean)) / len(A)
            - (len(B) - len(B_clean)) / len(B))
    print(f"""
  asymmetry in drop rate: {asym:+.1%}
  (positive means the {na} arm lost more, i.e. its measured affect
   rested on degenerate text to a degree the {nb} arm's did not)""")

    if len(A_clean) < 30 or len(B_clean) < 30:
        raise SystemExit("\ntoo few clean turns to compare")

    print(f"\n{'=' * 78}")
    print("COMPARISON, BEFORE AND AFTER FILTERING")
    print(f"{'=' * 78}")
    print(f"\n{'':<16} {'n_a':>6} {'n_b':>6} {'d mean':>11} "
          f"{'d median':>11} {'Cliff d':>9} {'p':>8}")
    print("-" * 78)
    before = report("unfiltered", A, B)
    after = report("filtered", A_clean, B_clean)

    print(f"\n{'=' * 78}")
    print("DROPPED TURNS")
    print(f"{'=' * 78}")
    for name, full, clean in ((na, A, A_clean), (nb, B, B_clean)):
        kept_ids = {id(r) for r in clean}
        dropped = [r for r in full if id(r) not in kept_ids]
        if not dropped:
            continue
        dropped.sort(key=lambda r: -r["_v"])
        print(f"\n  {name} — highest-scoring dropped turns")
        for r in dropped[:args.show]:
            print(f"    [{r['_v']:+.2f}] {r['child_reaction'][:62]}")

    print(f"\n{'=' * 78}")
    print("READING THIS")
    print(f"{'=' * 78}")

    shrink = (abs(after["mean"]) / abs(before["mean"])
              if before["mean"] else float("nan"))
    print(f"""
  effect before filtering  {before['mean']:+.3f}  (p = {before['p']:.4f})
  effect after filtering   {after['mean']:+.3f}  (p = {after['p']:.4f})
  retained                 {shrink:.0%} of the magnitude""")

    if after["p"] < 0.05 and shrink > 0.8:
        print("""
The effect holds on clean text at comparable magnitude. Degeneration
was not producing it, and the results can be reported without that
confound -- with the filter applied, and the drop rates stated.""")
    elif after["p"] < 0.05 and shrink > 1.05:
        print("""
The effect is STRONGER on clean text, which is the direction predicted
if degenerate turns were inflating protest scores in the learning arm.
Report the filtered figures; the unfiltered ones understate it.""")
    elif after["p"] < 0.05:
        print(f"""
The effect survives but at {shrink:.0%} of its magnitude. Part of what
was measured was degeneration. The filtered figure is the honest one
and should replace the unfiltered one throughout.""")
    else:
        print(f"""
The effect does NOT survive filtering (p = {after['p']:.4f}). What was
measured was substantially degeneration rather than a change in
objection, and the headline result needs restating on that basis.

That is a serious outcome and worth checking against the dropped-turn
list above before accepting it: if the filter is removing legitimate
emphatic speech ("It really really hurts") along with degenerate
strings, it is too aggressive and --max-run should be raised.""")

    if args.write_suffix:
        written = 0
        for rows, suffix in ((A_clean, args.write_suffix),
                             (B_clean, args.write_suffix)):
            by_path = {}
            for r in rows:
                by_path.setdefault(r["_path"], []).append(r)
            for path, rs in by_path.items():
                out = path.replace(".jsonl", f"{suffix}.jsonl")
                with open(out, "w") as f:
                    for r in rs:
                        r2 = {k: v for k, v in r.items()
                              if not k.startswith("_")}
                        f.write(json.dumps(r2) + "\n")
                written += 1
        print(f"\n  wrote {written} filtered files with suffix "
              f"'{args.write_suffix}'")

    print("""
CAVEATS. The filter is lexical and will drop legitimate emphatic
repetition ("It really really hurts" appears in the frozen arm) along
with degenerate strings; the dropped-turn lists above are printed so
this can be judged. Dropping turns changes which situations remain, so
a large asymmetry in drop rate can shift the situation mix as well as
the affect distribution. And filtering after the fact does not undo
the effect degeneration had on TRAINING -- the weights were updated on
that text regardless.""")


if __name__ == "__main__":
    main()