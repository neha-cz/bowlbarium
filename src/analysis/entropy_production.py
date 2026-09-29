"""
Entropy production and dissipated work, measured from the data.

WHY THIS AND NOT ANOTHER POTENTIAL
----------------------------------
reconstruct_potential.py estimated U(x) = -integral D1/D2 dx. That
formula only means anything if a potential EXISTS -- if the dynamics
are a gradient flow. When a system carries probability currents,
cycling through states rather than relaxing toward one, there is no
potential to find, and fitting one produces exactly what was observed:
a shape indistinguishable from the frozen baseline.

So the prior question is whether the assumption held. That is
measurable without assuming anything.

ENTROPY PRODUCTION
------------------
At equilibrium, detailed balance holds: a trajectory is as likely
forward as backward, so P(i->j) = P(j->i) for every pair of states.
Any asymmetry is a probability current, and its magnitude is the
entropy production rate:

    sigma = (1/2) * sum_ij (N_ij - N_ji) * ln(N_ij / N_ji) / N_total

sigma = 0 means equilibrium. sigma > 0 means the system is driven --
being held away from equilibrium by something. It is the defining
non-equilibrium quantity and it needs no model at all, just transition
counts.

The control is built in. Frozen runs have FIXED WEIGHTS, so they
cannot be driven by learning and should sit near equilibrium. Learning
runs are driven by the gradient updates. If sigma is the same in both,
learning is not driving the system on this coordinate.

CIRCULATION
-----------
In two dimensions the same asymmetry shows up as a net rotation --
the system cycling through, say, protest -> withdrawal -> comfort ->
protest. A cycle is impossible under gradient flow, since a potential
has no closed downhill loops. Detecting circulation would prove no
potential exists, which is a stronger statement than any particular
well shape.

DISSIPATED WORK
---------------
learning_loop.py already logs the KL divergence at each update. By the
Crooks fluctuation theorem, W_dissipated = W - dF = T * D(forward ||
reverse), so a logged KL IS dissipated work in units of temperature.
That is an identity from the fluctuation theorems, not an analogy, and
it means total dissipation per run can simply be summed.

THE NULL
--------
Finite samples produce apparent asymmetry by chance. Every statistic
here is compared against a permutation null: shuffle the time order
within each run, destroying temporal structure while preserving the
distribution of states, and recompute. A real current has to exceed
what shuffling produces.

Usage:
    python src/analysis/entropy_production.py --learning "runs/learning/*_on.jsonl" \\
                                 --frozen "runs/learning/*_off.jsonl"
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
import math
import random

from fep_layer import CONCEPT_WEIGHTS


def episode_means(rows, concepts):
    """One point per episode instead of per turn.

    The turn-level test found sigma = 0.05686 (learning) against
    0.05687 (frozen) -- identical to four decimals, and both BELOW
    their shuffled nulls. There is no turn-to-turn temporal structure
    to detect. The systematic change happens across tens of episodes,
    so that is the resolution at which a probability current, if one
    exists, would appear.
    """
    from collections import defaultdict
    by_ep = defaultdict(list)
    for r in rows:
        by_ep[r["episode"]].append([r["activations"][c] for c in concepts])
    out = []
    for _, pts in sorted(by_ep.items()):
        out.append([sum(p[d] for p in pts) / len(pts)
                    for d in range(len(concepts))])
    return out


def load_runs(patterns, concepts, level="turn"):
    runs = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                rows = [json.loads(line) for line in f]
            if not rows or "activations" not in rows[0]:
                continue
            if any(c not in rows[0]["activations"] for c in concepts):
                continue
            if level == "episode":
                runs.append(episode_means(rows, concepts))
            else:
                runs.append([[r["activations"][c] for c in concepts]
                             for r in rows])
    return runs


def load_kl(patterns):
    """Logged KL values from updates -- dissipated work in units of T."""
    out = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                rows = [json.loads(line) for line in f]
            kls = [r["kl"] for r in rows if r.get("kl") is not None]
            if kls:
                out.append((path.split("/")[-1], kls))
    return out


def quantile_edges(values, n_states):
    """Bin by quantile so every state is roughly equally occupied --
    equal-width bins would leave the tails nearly empty and make the
    transition counts meaningless."""
    s = sorted(values)
    return [s[int(i * (len(s) - 1) / n_states)] for i in range(1, n_states)]


def discretize(v, edges):
    for i, e in enumerate(edges):
        if v < e:
            return i
    return len(edges)


def transition_matrix(runs, edges_per_dim, n_states):
    """Counts between discretized states, within runs only."""
    dims = len(edges_per_dim)
    size = n_states ** dims
    N = [[0] * size for _ in range(size)]

    def code(point):
        idx = 0
        for d in range(dims):
            idx = idx * n_states + discretize(point[d], edges_per_dim[d])
        return idx

    for series in runs:
        for t in range(len(series) - 1):
            N[code(series[t])][code(series[t + 1])] += 1
    return N


def entropy_production(N):
    """sigma = (1/2) sum (N_ij - N_ji) ln(N_ij/N_ji) / total."""
    total = sum(sum(row) for row in N)
    if total == 0:
        return 0.0
    sigma = 0.0
    for i in range(len(N)):
        for j in range(i + 1, len(N)):
            a, b = N[i][j], N[j][i]
            if a > 0 and b > 0:
                sigma += (a - b) * math.log(a / b)
            elif a > 0 or b > 0:
                # One-way transitions are maximally irreversible; using
                # a pseudocount keeps the estimate finite.
                a1, b1 = a + 0.5, b + 0.5
                sigma += (a1 - b1) * math.log(a1 / b1)
    return sigma / (2 * total)


def shuffled_runs(runs, rng):
    out = []
    for series in runs:
        s = list(series)
        rng.shuffle(s)
        out.append(s)
    return out


def analyse(label, runs, concepts, n_states, n_perm, rng):
    print(f"\n{'=' * 74}")
    print(f"{label}")
    print(f"{'=' * 74}")
    n_trans = sum(len(s) - 1 for s in runs)
    print(f"{len(runs)} run(s), {n_trans} transitions, "
          f"{n_states}^{len(concepts)} = {n_states ** len(concepts)} states")
    if n_trans < 60:
        print("too few transitions to estimate")
        return None
    if n_trans < 150:
        print("  (few transitions -- the permutation null is doing a lot")
        print("   of work here; treat a positive result cautiously)")

    flat = [[p[d] for s in runs for p in s] for d in range(len(concepts))]
    edges = [quantile_edges(flat[d], n_states) for d in range(len(concepts))]

    N = transition_matrix(runs, edges, n_states)
    sigma = entropy_production(N)

    null = []
    for _ in range(n_perm):
        Ns = transition_matrix(shuffled_runs(runs, rng), edges, n_states)
        null.append(entropy_production(Ns))
    null.sort()
    mu = sum(null) / len(null)
    sd = (sum((v - mu) ** 2 for v in null) / max(1, len(null) - 1)) ** 0.5
    above = sum(1 for v in null if v >= sigma)
    p = (above + 1) / (len(null) + 1)
    z = (sigma - mu) / sd if sd > 0 else float("nan")

    print(f"\nentropy production   sigma = {sigma:.5f}")
    print(f"shuffled null        mean  = {mu:.5f}  sd = {sd:.5f}")
    print(f"                     z = {z:+.2f}, p = {p:.3f} "
          f"({n_perm} permutations)")
    if p < 0.05:
        print("  -> exceeds the null: a real probability current")
    else:
        print("  -> within the null: no detectable irreversibility")
    return {"sigma": sigma, "null_mean": mu, "null_sd": sd, "z": z, "p": p}


def dissipated_work(patterns):
    print(f"\n{'=' * 74}")
    print("DISSIPATED WORK  (from logged KL values)")
    print(f"{'=' * 74}")
    runs = load_kl(patterns)
    if not runs:
        print("\nNo KL values found. These are logged only on turns where")
        print("an update fired, and only in files that still contain the")
        print("learning log rather than the scored affect output --")
        print("extract_emotions.py --score-all overwrites its inputs.")
        return
    print("\nBy Crooks, W_dissipated = T * KL, so these are dissipation")
    print("in units of temperature.\n")
    print(f"{'run':<34} {'updates':>8} {'mean KL':>10} {'total':>10}")
    print("-" * 74)
    for name, kls in runs:
        print(f"{name[:33]:<34} {len(kls):>8} "
              f"{sum(kls)/len(kls):>10.4f} {sum(kls):>10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learning", nargs="+", required=True)
    ap.add_argument("--frozen", nargs="+", default=None)
    ap.add_argument("--concepts", nargs="+", default=["protesting"],
                    choices=list(CONCEPT_WEIGHTS),
                    help="one concept for entropy production; two to "
                         "look for circulation")
    ap.add_argument("--states", type=int, default=5,
                    help="bins per dimension (quantile-spaced)")
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--level", choices=["turn", "episode"], default="turn",
                    help="episode averages within each episode first; the "
                         "turn level has no detectable structure")
    ap.add_argument("--kl-from", nargs="+", default=None,
                    help="learning LOGS (not scored affect files) for "
                         "the dissipated-work section")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    learn = load_runs(args.learning, args.concepts, args.level)
    if not learn:
        raise SystemExit("no usable learning runs found")
    froz = (load_runs(args.frozen, args.concepts, args.level)
            if args.frozen else [])

    L = analyse("LEARNING (weights update)", learn, args.concepts,
                args.states, args.permutations, rng)
    F = analyse("FROZEN (fixed weights)", froz, args.concepts,
                args.states, args.permutations, rng) if froz else None

    if args.kl_from:
        dissipated_work(args.kl_from)

    print(f"\n{'=' * 74}")
    print("READING THIS")
    print(f"{'=' * 74}")
    if L and F:
        print(f"\nsigma: learning {L['sigma']:.5f} (z={L['z']:+.2f}), "
              f"frozen {F['sigma']:.5f} (z={F['z']:+.2f})")
        l_real = L["p"] < 0.05
        f_real = F["p"] < 0.05
        if l_real and not f_real:
            print("""
The learning runs show irreversibility the frozen runs do not. The
weight updates are driving the system away from equilibrium, and
crucially this means the dynamics are NOT a gradient flow -- a
potential has no closed loops, so probability currents rule one out.
That would explain why the potential reconstruction found nothing:
it was fitting an object that does not exist here.""")
        elif l_real and f_real:
            print("""
Both arms show irreversibility, so it is not produced by learning.
Likely sources: the event sequence itself is ordered (episodes have
structure), or the running standardiser introduces a drift over the
course of a run. Worth checking against runs with shuffled event
order before concluding anything about the dynamics.""")
        else:
            print("""
Neither arm shows detectable irreversibility. On this coordinate the
process is consistent with detailed balance, which means the gradient-
flow assumption behind the potential reconstruction was reasonable and
its finding stands: a single well, essentially the same with and
without learning.

That closes the question rather than leaving it open. It is also the
fourth thermodynamic framing to come back null on this system.""")
    print("""
CAVEATS. Entropy production estimated from discretised states is
biased upward at small sample sizes, which is what the permutation
null controls for. Coarse binning can hide currents that exist at
finer resolution, and finer binning empties the cells -- try
--states 4 and --states 6 and see whether the conclusion moves.""")


if __name__ == "__main__":
    main()