"""
Did the learned adapters degrade?

WHY THIS MATTERS BEYOND ONE PROBE
---------------------------------
The refusal probe produced responses like:

    "I'm not even touching it, I just said him his name."
    "You're really hurtin' and I'm just gonna sit here fo..."
    "You don't even get a piece of junk, I'm doing it myself."

These are not refusals or compliances. They are incoherent, and they
address the mother rather than answering the request that was put to
them. The diversity gate in the learning loop checks for REPETITION,
not coherence, so a run can pass its diversity check while producing
this.

If the degradation is general rather than specific to the ceiling
adapters, several results need re-examining:

  the refusal probe        may be measuring incoherence, not compliance
  weight-space convergence may partly reflect convergence toward
                           degeneracy rather than toward a learning
                           direction
  the objection decay      a measure built on activations of degraded
                           text is harder to interpret

If it is specific to those runs, the fix is local: use different
adapters and note it.

WHAT IS MEASURED
----------------
Text-level signals that need no model and no judge, computed per
quartile of a run so drift within the run is visible:

    type-token ratio      lexical diversity within a response
    repeated bigrams      "even even", "you just you just"
    truncation rate       responses ending mid-word or mid-clause
    2nd-person rate       share of responses addressing the partner
                          rather than answering -- the specific
                          pattern noticed in the probe
    function-word ratio   a crude fluency proxy; degenerate text
                          drifts away from normal English proportions
    mean length

Frozen twins are the control: they experience identical events with no
gradient step, so any divergence between arms is attributable to
learning rather than to the environment.

Usage:
    python coherence_check.py \\
        --run learning "runs_learning/*_on.jsonl" \\
        --run frozen "runs_learning/*_off.jsonl" \\
        --run ceiling "runs_ceil/*.jsonl"
"""

import argparse
import glob
import json
import re
from collections import Counter

FUNCTION_WORDS = {
    "the", "a", "an", "and", "but", "or", "if", "of", "to", "in", "on",
    "at", "for", "with", "is", "was", "are", "were", "be", "been",
    "it", "that", "this", "you", "i", "me", "my", "your", "not",
    "do", "did", "does", "have", "has", "had", "will", "would", "can",
}
SECOND_PERSON = re.compile(r"\b(you|your|you're|youre|you'll|yourself)\b",
                           re.I)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def type_token(text):
    w = [t.lower() for t in re.findall(r"[a-z']+", text.lower())]
    return len(set(w)) / len(w) if w else 1.0


def repeated_bigram(text):
    w = [t.lower() for t in re.findall(r"[a-z']+", text.lower())]
    if len(w) < 4:
        return 0
    bg = [(w[i], w[i + 1]) for i in range(len(w) - 1)]
    c = Counter(bg)
    return sum(1 for _, v in c.items() if v > 1)


def looks_truncated(text):
    """Heuristic: no terminal punctuation and ends on a short fragment."""
    t = text.strip()
    if not t:
        return True
    if t[-1] in '.!?"\'':
        return False
    last = t.split()[-1] if t.split() else ""
    return len(last) <= 2


def function_ratio(text):
    w = [t.lower() for t in re.findall(r"[a-z']+", text.lower())]
    if not w:
        return 0.0
    return sum(1 for t in w if t in FUNCTION_WORDS) / len(w)


def load(patterns):
    rows = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            src = path.split("/")[-1]
            with open(path) as f:
                for i, line in enumerate(f):
                    r = json.loads(line)
                    if r.get("child_reaction"):
                        r["_src"] = src
                        r["_i"] = i
                        rows.append(r)
    return rows


def metrics(texts):
    return {
        "n": len(texts),
        "words": mean([len(t.split()) for t in texts]),
        "ttr": mean([type_token(t) for t in texts]),
        "rep_bg": mean([repeated_bigram(t) for t in texts]),
        "trunc": mean([1 if looks_truncated(t) else 0 for t in texts]),
        "you": mean([1 if SECOND_PERSON.search(t) else 0 for t in texts]),
        "func": mean([function_ratio(t) for t in texts]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=2, action="append",
                    metavar=("NAME", "GLOB"), required=True)
    ap.add_argument("--quartiles", action="store_true", default=True)
    ap.add_argument("--show", type=int, default=4)
    args = ap.parse_args()

    print("=" * 78)
    print("COHERENCE CHECK")
    print("=" * 78)
    print("""
The diversity gate checks repetition, not coherence, so a run can pass
it while degenerating. Frozen twins are the control: identical events,
no gradient step.""")

    groups = {}
    for name, pattern in args.run:
        rows = load([pattern])
        if not rows:
            print(f"\n  {name}: no rows")
            continue
        groups[name] = rows

    print(f"\n{'run':<12} {'n':>5} {'words':>7} {'TTR':>7} {'rep bg':>8} "
          f"{'trunc':>7} {'2nd-p':>7} {'func':>7}")
    print("-" * 78)
    for name, rows in groups.items():
        m = metrics([r["child_reaction"] for r in rows])
        print(f"{name:<12} {m['n']:>5} {m['words']:>7.1f} {m['ttr']:>7.3f} "
              f"{m['rep_bg']:>8.2f} {m['trunc']:>6.0%} {m['you']:>6.0%} "
              f"{m['func']:>7.3f}")

    print("""
  TTR near 1.0 means no word repeats within a response. rep bg counts
  bigrams occurring more than once -- "even even", "you just you
  just". 2nd-p is the share of responses containing you/your, the
  pattern noticed in the refusal probe where the agent addressed the
  partner instead of answering.""")

    # ---- within-run drift ----
    print(f"\n{'=' * 78}")
    print("DRIFT WITHIN RUN  (first vs last quartile)")
    print(f"{'=' * 78}")
    for name, rows in groups.items():
        by_src = {}
        for r in rows:
            by_src.setdefault(r["_src"], []).append(r)
        firsts, lasts = [], []
        for src, rs in by_src.items():
            rs.sort(key=lambda x: x["_i"])
            q = max(1, len(rs) // 4)
            firsts += [x["child_reaction"] for x in rs[:q]]
            lasts += [x["child_reaction"] for x in rs[-q:]]
        mf, ml = metrics(firsts), metrics(lasts)
        print(f"\n  {name}")
        print(f"    {'metric':<12} {'first Q':>10} {'last Q':>10} "
              f"{'change':>10}")
        for k, label in (("words", "words"), ("ttr", "TTR"),
                         ("rep_bg", "rep bigram"), ("trunc", "truncated"),
                         ("you", "2nd person"), ("func", "function")):
            print(f"    {label:<12} {mf[k]:>10.3f} {ml[k]:>10.3f} "
                  f"{ml[k]-mf[k]:>+10.3f}")

    # ---- worst examples ----
    print(f"\n{'=' * 78}")
    print("LEAST COHERENT RESPONSES BY RUN")
    print(f"{'=' * 78}")
    for name, rows in groups.items():
        scored = sorted(
            rows,
            key=lambda r: (-repeated_bigram(r["child_reaction"]),
                           type_token(r["child_reaction"])))
        print(f"\n  {name}")
        for r in scored[:args.show]:
            t = r["child_reaction"]
            print(f"    [ttr {type_token(t):.2f} bg {repeated_bigram(t)}] "
                  f"{t[:62]}")

    # ---- verdict ----
    print(f"\n{'=' * 78}")
    print("READING THIS")
    print(f"{'=' * 78}")

    if "learning" in groups and "frozen" in groups:
        ml = metrics([r["child_reaction"] for r in groups["learning"]])
        mfz = metrics([r["child_reaction"] for r in groups["frozen"]])
        d_ttr = ml["ttr"] - mfz["ttr"]
        d_you = ml["you"] - mfz["you"]
        d_bg = ml["rep_bg"] - mfz["rep_bg"]
        print(f"""
  learning vs frozen
    TTR          {d_ttr:+.3f}
    2nd person   {d_you:+.0%}
    rep bigrams  {d_bg:+.2f}""")

        degraded = d_ttr < -0.05 or d_bg > 0.5
        if degraded:
            print("""
The learning arm is measurably less coherent than its frozen twin on
identical events. That is a consequence of the loop and belongs in the
paper -- but it also means results computed on these responses carry a
confound: an activation measure applied to degraded text is harder to
interpret, and the refusal probe in particular may be reading
incoherence rather than compliance.""")
        elif abs(d_you) > 0.15:
            print(f"""
Coherence is comparable but the learning arm addresses the partner
{d_you:+.0%} more often. That is a stylistic shift rather than
degradation -- the agent talks ABOUT the exchange rather than
answering it, which would explain the refusal probe's responses
without implying the text is degenerate.""")
        else:
            print("""
No coherence difference between the learning arm and its frozen twin.
The odd responses in the refusal probe are then specific to the
probe's out-of-distribution requests rather than a property of the
adapters, and the other results are unaffected.""")

    if "ceiling" in groups and "learning" in groups:
        mc = metrics([r["child_reaction"] for r in groups["ceiling"]])
        ml = metrics([r["child_reaction"] for r in groups["learning"]])
        print(f"""
  ceiling adapters vs main learning runs
    TTR {mc['ttr']:.3f} against {ml['ttr']:.3f}
    rep bigrams {mc['rep_bg']:.2f} against {ml['rep_bg']:.2f}

  If the ceiling runs are worse, the degradation is specific to them
  and the weight-space convergence result should be recomputed on
  adapters that are not degraded.""")

    print("""
CAVEATS -- AND A SPECIFIC ONE THAT LIMITS THIS SCRIPT.

Surface metrics detect DEGENERACY (repetition, truncation) but not
INCOHERENCE. Tested directly: "I'm not even touching it, I just said
him his name" -- the clearest incoherence in the refusal probe --
scores TTR 1.00, zero repeated bigrams, not truncated. It is
grammatical and meaningless, and nothing here catches that.

So a clean result on TTR and bigrams does NOT establish the responses
make sense. The 2nd-person rate is the metric that tracks the pattern
actually observed, since it fires on responses that address the
partner instead of answering. Semantic incoherence needs a judge model
or human reading, and if these metrics come back clean while the
responses still look wrong, that is the next step rather than a
reason to conclude nothing is wrong.

Short child utterances also have high TTR by default, so the metric is
informative as a between-arm comparison rather than in absolute
terms.""")


if __name__ == "__main__":
    main()