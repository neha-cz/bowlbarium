"""
Parameter space: where do the weights actually go?

WHY THIS IS A DIFFERENT MEASUREMENT
-----------------------------------
Every analysis in this project so far has run on a scalar projection of
activations extracted from generated text -- three transformations away
from the 6.8M LoRA parameters actually being optimised:

    weights -> activations -> contrastive projection -> z-score

Eleven thermodynamic and dynamical framings were tested on that
readout, and all eleven failed. The last one failed for a diagnosable
reason: fluctuation-dissipation gave R^2 = 0.469 on linearity and a
susceptibility that flipped sign between adjacent perturbation pairs,
so there is no measurable temperature in that space and every energy
expressed in units of it was decoration.

Parameter space is where the optimisation actually happens. The
adapters are saved on disk, so the displacement is directly
computable -- no projection, no z-scoring, no emotion vectors.

THE QUESTION NOBODY HAS ASKED
-----------------------------
Does caregiver reliability change WHERE the weights go, or only HOW
FAR?

    cos(dtheta_low, dtheta_high) near 1
        the two conditions push the weights along the SAME axis and
        differ only in magnitude. Reliability is a gain on a single
        learning direction.

    cos near 0
        the conditions move the weights in unrelated directions. Low-
        and high-reliability training are doing different things, not
        more or less of one thing.

    cos near -1
        they actively oppose each other.

This has no behavioural proxy. Two adapters could produce similar text
while sitting in quite different places in parameter space, or produce
different text from nearby points. The activation-based analyses could
not distinguish these cases even in principle.

WHAT IS MEASURED
----------------
    displacement    ||dtheta|| from the cold-start adapter
    direction       cosine between condition displacements
    concentration   participation ratio of the displacement -- how
                    many parameters carry the change. A change spread
                    over millions of weights is a different object
                    from one concentrated in a few hundred.
    per-layer       which layers move, and whether conditions differ
                    in that profile

Usage:
    python src/analysis/parameter_space.py \\
        --base adapters/child_r32 \\
        --adapter low_A  runs/hysteresis/LH_adapter_A \\
        --adapter low_B  runs/hysteresis/LL_adapter_A \\
        --adapter high_A runs/hysteresis/HH_adapter_A
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
import math
import os
import re
from collections import defaultdict


def load_adapter(path):
    """Read adapters.safetensors into a dict of flat numpy arrays."""
    import numpy as np
    from safetensors.numpy import load_file

    f = path if path.endswith(".safetensors") else os.path.join(
        path, "adapters.safetensors")
    if not os.path.exists(f):
        raise SystemExit(f"not found: {f}")
    raw = load_file(f)
    return {k: np.asarray(v, dtype=np.float64).ravel() for k, v in raw.items()}


def displacement(base, other):
    """other - base, over the keys they share."""
    import numpy as np
    keys = sorted(set(base) & set(other))
    missing = set(other) - set(base)
    if missing:
        print(f"  note: {len(missing)} keys in adapter not in base, skipped")
    return {k: other[k] - base[k] for k in keys
            if base[k].shape == other[k].shape}


def flatten(d):
    import numpy as np
    return np.concatenate([d[k] for k in sorted(d)])


def norm(v):
    import numpy as np
    return float(np.linalg.norm(v))


def cosine(a, b):
    import numpy as np
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def participation_ratio(v):
    """(sum v^2)^2 / sum v^4.

    The number of parameters effectively carrying the change. Equals n
    when the change is spread evenly and 1 when it sits in a single
    weight, so it says whether an update is diffuse or concentrated
    without depending on the norm.
    """
    import numpy as np
    s2 = float(np.sum(v ** 2))
    s4 = float(np.sum(v ** 4))
    return (s2 ** 2 / s4) if s4 > 0 else 0.0


def layer_of(key):
    m = re.search(r"layers?\.(\d+)", key)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="cold-start adapter, the reference point")
    ap.add_argument("--adapter", nargs=2, action="append",
                    metavar=("NAME", "PATH"), required=True)
    args = ap.parse_args()

    import numpy as np

    print("=" * 78)
    print("PARAMETER SPACE")
    print("=" * 78)
    print(f"\nreference: {args.base}")

    base = load_adapter(args.base)
    n_params = sum(len(v) for v in base.values())
    print(f"{len(base)} tensors, {n_params:,} parameters")

    disps = {}
    for name, path in args.adapter:
        print(f"\nloading {name}: {path}")
        d = displacement(base, load_adapter(path))
        if not d:
            print("  no shared tensors -- skipped")
            continue
        disps[name] = d

    if len(disps) < 2:
        raise SystemExit("need at least two adapters to compare directions")

    # ---------- magnitude and concentration ----------
    print(f"\n{'=' * 78}")
    print("DISPLACEMENT FROM COLD START")
    print(f"{'=' * 78}")
    print(f"\n{'adapter':<12} {'||dtheta||':>12} {'rel. to base':>14} "
          f"{'participation':>15} {'% of params':>12}")
    print("-" * 78)
    base_norm = norm(flatten(base))
    flats = {}
    for name, d in disps.items():
        v = flatten(d)
        flats[name] = v
        pr = participation_ratio(v)
        print(f"{name:<12} {norm(v):>12.4f} {norm(v)/base_norm:>13.1%} "
              f"{pr:>15,.0f} {pr/len(v):>11.2%}")

    print("""
  participation ratio is how many weights effectively carry the
  change: equal to n if spread evenly, 1 if concentrated in a single
  parameter.""")

    # ---------- direction: the headline ----------
    print(f"\n{'=' * 78}")
    print("DIRECTION  --  do the conditions move the weights the same way?")
    print(f"{'=' * 78}")
    names = list(flats)
    print(f"\n{'':<12}" + "".join(f"{n:>12}" for n in names))
    print("-" * 78)
    cos = {}
    for a in names:
        row = f"{a:<12}"
        for b in names:
            c = cosine(flats[a], flats[b])
            cos[(a, b)] = c
            row += f"{c:>12.4f}"
        print(row)

    print("""
  1.0  identical direction -- the conditions differ only in how far
  0.0  orthogonal -- they are doing unrelated things to the weights
 -1.0  opposed""")

    # ---------- per-layer profile ----------
    print(f"\n{'=' * 78}")
    print("PER-LAYER PROFILE")
    print(f"{'=' * 78}")
    layer_norms = {}
    for name, d in disps.items():
        by_layer = defaultdict(float)
        for k, v in d.items():
            by_layer[layer_of(k)] += float(np.sum(v ** 2))
        layer_norms[name] = {l: math.sqrt(s) for l, s in by_layer.items()}

    layers = sorted({l for ln in layer_norms.values() for l in ln})
    print(f"\n{'layer':>6}" + "".join(f"{n:>12}" for n in names))
    print("-" * 78)
    for l in layers:
        row = f"{l:>6}"
        for n in names:
            row += f"{layer_norms[n].get(l, 0.0):>12.4f}"
        print(row)

    # ---------- verdict ----------
    print(f"\n{'=' * 78}")
    print("READING THIS")
    print(f"{'=' * 78}")

    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    same_cond = [(a, b) for a, b in pairs
                 if a.split("_")[0] == b.split("_")[0]]
    diff_cond = [(a, b) for a, b in pairs
                 if a.split("_")[0] != b.split("_")[0]]

    if same_cond:
        w = [cos[p] for p in same_cond]
        print(f"\nwithin-condition cosine:  {min(w):.3f} to {max(w):.3f}")
    if diff_cond:
        a = [cos[p] for p in diff_cond]
        print(f"across-condition cosine:  {min(a):.3f} to {max(a):.3f}")

    if same_cond and diff_cond:
        w_mean = sum(cos[p] for p in same_cond) / len(same_cond)
        a_mean = sum(cos[p] for p in diff_cond) / len(diff_cond)
        print(f"\nmean within {w_mean:.3f}, mean across {a_mean:.3f}, "
              f"gap {w_mean - a_mean:+.3f}")
        if w_mean - a_mean > 0.2:
            print("""
Runs from the SAME condition point in more similar directions than
runs from DIFFERENT conditions. Caregiver reliability determines where
the weights go, not merely how far -- which is a claim about the
learning itself that no behavioural measurement could establish.""")
        elif abs(w_mean - a_mean) < 0.1:
            print("""
Within- and across-condition similarity are indistinguishable. The
direction of weight movement does not depend on the caregiver
condition; whatever differs between conditions is not the axis the
weights travel along.""")
    else:
        print("""
Name adapters as condition_replicate (e.g. low_A, low_B, high_A) and
the within-versus-across comparison will be computed automatically.""")

    all_cos = [cos[p] for p in pairs]
    if all_cos and max(all_cos) < 0.2:
        print(f"""
All pairwise cosines are below 0.2, so every run moves the weights in
a nearly orthogonal direction -- including replicates of the same
condition. In 6.8M dimensions random vectors are almost orthogonal by
default, so this is what NO reproducible learning direction looks
like: the updates are dominated by run-specific noise rather than by
anything the condition determines.""")

    print("""
CAVEATS. Cosine similarity in millions of dimensions is small for
almost any pair, so the within-versus-across CONTRAST carries the
information, not the absolute value. LoRA is a product of two
matrices, so identical effective updates can appear as different
(A, B) factorisations -- a low cosine may understate how similar the
induced weight changes are. Comparing B@A products would be the
stricter test.""")


if __name__ == "__main__":
    main()