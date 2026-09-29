"""
Resolving Result 3: saturation, length, and per-seed replication.

WHAT NEEDS RESOLVING
--------------------
Result 3 has been stated three ways today and each was undercut:

  1. "learning reduces protest"        -- true but the per-turn effect
                                          is small (Cliff's d -0.099,
                                          CI crossing zero)
  2. "occupancy changed, states did"   -- the two-state fit; replicated
     "not"                                per-seed, but the fit only
                                          ever saw a scalar
  3. reading the actual turns showed the two conditions' high-protest
     bands are almost disjoint in SITUATION (frozen 42% conflict,
     learning ~0% conflict, 23% achievement) and in FORM (frozen: bare
     "No!"; learning: qualified argument, "But there's nothing else!")

If the bands contain different situations and different language, they
are not one state entered at different rates, and the occupancy
framing overstates what a scalar projection can support.

Three things have to be checked before Result 3 can be stated at all.

CHECK 1 -- SATURATION
Eight frozen turns read exactly +2.300, all of them "No!". If the
protest vector has a ceiling, it cannot rank intense turns against
each other, and every analysis using the upper range inherits that --
the mixture fit, the percentiles, the potential reconstruction. This
counts how much mass sits at the ceiling in each condition.

CHECK 2 -- LENGTH
The learning child's reactions grew from 5.4 to 14.5 words because it
was TRAINED on long revision targets. Both conditions generate
reactions from the same system prompt, so this is a real consequence
of learning rather than a prompt difference -- but it is mediated by a
design choice, and long turns may simply project differently. Matching
on word count separates "learning changed which situations produce
protest" from "longer turns land differently on the vector".

CHECK 3 -- PER SEED
The 42%-vs-0% conflict split is pooled. Four findings in this project
died at exactly this step and a fifth was downgraded.

Usage:
    python src/analysis/resolve_result3.py \\
        --a learning "runs/learning/*_on.jsonl" \\
        --b frozen   "runs/learning/*_off.jsonl"
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
from collections import Counter, defaultdict

from fep_layer import CONCEPT_WEIGHTS


def load_rows(pattern, concept):
    rows = []
    for path in sorted(glob.glob(pattern)):
        src = path.split("/")[-1]
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                a = r.get("activations")
                if not a or concept not in a:
                    continue
                r["_v"] = a[concept]
                r["_src"] = src
                r["_len"] = len(r.get("child_reaction", "").split())
                rows.append(r)
    return rows


def quantile(vals, q):
    s = sorted(vals)
    i = q * (len(s) - 1)
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (i - lo) * (s[hi] - s[lo])


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


# ------------------------------------------------------------------
# CHECK 1: saturation
# ------------------------------------------------------------------

def saturation(label, rows, concept, tol=0.005):
    """Find repeated values ANYWHERE in the distribution.

    A first version looked only near the maximum, which was wrong: in
    the real data the eight identical +2.300 turns are not the maximum
    (that is +3.835) but a SPIKE in the middle of the distribution.
    A spike anywhere means the measure cannot distinguish the turns
    that produce it -- here, every bare "No!" scoring the same.
    """
    from collections import Counter
    vals = [r["_v"] for r in rows]
    # bin at tol and find the heaviest bins
    binned = Counter(round(v / tol) for v in vals)
    top = binned.most_common(4)

    print(f"\n  {label}   n={len(rows)}, range "
          f"{min(vals):+.3f} to {max(vals):+.3f}")
    print(f"    heaviest {tol}-wide bins:")
    spike_frac = 0.0
    for b, c in top:
        centre = b * tol
        if c < 3:
            continue
        share = c / len(rows)
        spike_frac = max(spike_frac, share)
        texts = Counter(r.get("child_reaction", "")
                        for r in rows if abs(r["_v"] - centre) < tol)
        distinct = len(texts)
        common = texts.most_common(1)[0] if texts else ("", 0)
        print(f"      {centre:+.3f}: {c:>4} turns ({share:.1%}), "
              f"{distinct} distinct text(s)"
              + (f'  most common: {common[1]}x "{common[0][:40]}"'
                 if distinct <= 3 else ""))
    if spike_frac == 0:
        print("      (no bin holds 3 or more turns -- no spike)")
    return spike_frac


# ------------------------------------------------------------------
# CHECK 2: length-matched comparison
# ------------------------------------------------------------------

def length_matched(A, B, concept, bucket=3, min_per=15):
    """Compare protest and event mix within word-count buckets."""
    def by_bucket(rows):
        d = defaultdict(list)
        for r in rows:
            d[r["_len"] // bucket].append(r)
        return d

    ba, bb = by_bucket(A), by_bucket(B)
    shared = sorted(set(ba) & set(bb))

    print(f"\n{'words':>10} {'n_A':>6} {'n_B':>6} {'mean_A':>9} "
          f"{'mean_B':>9} {'diff':>8} {'conflict_A':>12} {'conflict_B':>12}")
    print("-" * 78)
    rows_out = []
    for b in shared:
        a, bb_ = ba[b], bb[b]
        if len(a) < min_per or len(bb_) < min_per:
            continue
        ma, mb = mean([r["_v"] for r in a]), mean([r["_v"] for r in bb_])
        ca = sum(1 for r in a if r.get("event_type") == "conflict") / len(a)
        cb = sum(1 for r in bb_ if r.get("event_type") == "conflict") / len(bb_)
        rows_out.append((b, len(a), len(bb_), ma, mb, ca, cb))
        print(f"{f'{b*bucket}-{(b+1)*bucket-1}':>10} {len(a):>6} {len(bb_):>6} "
              f"{ma:>+9.3f} {mb:>+9.3f} {ma-mb:>+8.3f} "
              f"{ca:>11.0%} {cb:>11.0%}")
    return rows_out


# ------------------------------------------------------------------
# CHECK 3: per-seed event composition
# ------------------------------------------------------------------

def seed_of(src):
    import re
    m = re.search(r"[LDA](\d{3})", src)
    return m.group(1) if m else src


def per_seed_events(A, B, concept, q=0.85):
    """Conflict share of the high-protest band, per seed."""
    def band(rows):
        vals = [r["_v"] for r in rows]
        if len(vals) < 20:
            return None
        cut = quantile(vals, q)
        return [r for r in rows if r["_v"] >= cut]

    seeds_a = defaultdict(list)
    seeds_b = defaultdict(list)
    for r in A:
        seeds_a[seed_of(r["_src"])].append(r)
    for r in B:
        seeds_b[seed_of(r["_src"])].append(r)

    shared = sorted(set(seeds_a) & set(seeds_b))
    print(f"\n{'seed':>8} {'conflict% A':>13} {'conflict% B':>13} "
          f"{'diff':>9} {'nA':>5} {'nB':>5}")
    print("-" * 78)
    diffs = []
    for s in shared:
        ba, bb = band(seeds_a[s]), band(seeds_b[s])
        if not ba or not bb:
            continue
        ca = sum(1 for r in ba if r.get("event_type") == "conflict") / len(ba)
        cb = sum(1 for r in bb if r.get("event_type") == "conflict") / len(bb)
        diffs.append(ca - cb)
        print(f"{s:>8} {ca:>12.0%} {cb:>12.0%} {ca-cb:>+9.0%} "
              f"{len(ba):>5} {len(bb):>5}")
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs=2, metavar=("NAME", "GLOB"), required=True)
    ap.add_argument("--b", nargs=2, metavar=("NAME", "GLOB"), required=True)
    ap.add_argument("--concept", default="protesting",
                    choices=list(CONCEPT_WEIGHTS))
    ap.add_argument("--band", type=float, default=0.85)
    args = ap.parse_args()

    na, pa = args.a
    nb, pb = args.b
    A = load_rows(pa, args.concept)
    B = load_rows(pb, args.concept)
    if not A or not B:
        raise SystemExit("no rows loaded")

    print("=" * 78)
    print(f"RESOLVING RESULT 3  --  {args.concept}")
    print("=" * 78)
    print(f"\n{na}: {len(A)} turns, mean length {mean([r['_len'] for r in A]):.1f} words")
    print(f"{nb}: {len(B)} turns, mean length {mean([r['_len'] for r in B]):.1f} words")

    # ---- 1 ----
    print(f"\n{'=' * 78}")
    print("CHECK 1 — IS THE VECTOR SATURATING?")
    print(f"{'=' * 78}")
    ta = saturation(na, A, args.concept)
    tb = saturation(nb, B, args.concept)
    print(f"""
  A measure pinned at a ceiling cannot rank intense turns against each
  other. If a large share of one condition's top decile sits there and
  the other's does not, the two conditions are being measured with
  different resolution and the mixture fit is comparing unlike things.""")

    # ---- 2 ----
    print(f"\n{'=' * 78}")
    print("CHECK 2 — DOES IT SURVIVE LENGTH MATCHING?")
    print(f"{'=' * 78}")
    print("""
Within each word-count bucket the two conditions are compared on equal
terms. If the protest difference and the conflict-share difference
persist inside buckets, they are not length effects.""")
    lm = length_matched(A, B, args.concept)

    # ---- 3 ----
    print(f"\n{'=' * 78}")
    print("CHECK 3 — DOES THE SITUATIONAL SHIFT HOLD PER SEED?")
    print(f"{'=' * 78}")
    print(f"\nconflict share of the top {1-args.band:.0%} of protest turns:")
    diffs = per_seed_events(A, B, args.concept, args.band)

    # ---- verdict ----
    print(f"\n{'=' * 78}")
    print("READING THIS")
    print(f"{'=' * 78}")

    if max(ta, tb) > 0.03 and abs(ta - tb) > 0.02:
        print(f"""
SATURATION IS A PROBLEM. The largest single-value spike holds {ta:.1%}
of {na}'s turns and {tb:.1%} of {nb}'s. A spike means the measure
cannot distinguish the turns producing it, and the two conditions are
affected unequally -- so the mixture fit was comparing distributions
of different effective resolution.""")
    elif max(ta, tb) > 0.03:
        print(f"""
Both conditions show spikes of similar size ({ta:.1%} and {tb:.1%}).
Resolution is limited but symmetrically, so between-condition
comparisons are less affected than they would otherwise be.""")
    else:
        print("""
No substantial spike in either condition. The identical values noticed
by eye were a small share of the whole.""")

    if lm:
        within = [r[3] - r[4] for r in lm]
        conf = [r[5] - r[6] for r in lm]
        print(f"""
LENGTH MATCHING: protest difference within buckets averages
{mean(within):+.3f} (pooled difference was -0.234); conflict-share
difference averages {mean(conf):+.0%}.""")
        if abs(mean(within)) < 0.1:
            print("""
  The protest difference nearly vanishes at matched length. Most of
  Result 2's effect is that the learning child says MORE, not that it
  protests less at comparable length.""")
        else:
            print("""
  The protest difference survives length matching, so it is not
  explained by the learning child's longer turns alone.""")

    if diffs:
        signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in diffs]
        consistent = all(s == signs[0] for s in signs) and signs[0] != 0
        print(f"""
PER SEED: conflict-share difference {mean(diffs):+.0%} on average,
sign consistent across {len(diffs)} seeds: {'YES' if consistent else 'no'}""")
        if consistent:
            print("""
  The situational shift holds per seed. Learning changes WHICH events
  produce high protest, not only how often protest occurs -- and that
  is a different claim from the occupancy one, which should be
  restated.""")
        else:
            print("""
  The sign flips between seeds. The pooled 42%-vs-0% conflict split
  was driven by particular runs, and the situational reading does not
  survive -- as four earlier findings did not.""")

    print("""
WHAT TO CONCLUDE. Result 3 can only be stated as an occupancy change
if saturation is symmetric, the protest difference survives length
matching, and the situational shift does NOT hold per seed. If the
situational shift does hold, the honest statement is that learning
changed which situations produce protest, and the two-state fit was
describing a scalar collapse of two different things.""")


if __name__ == "__main__":
    main()