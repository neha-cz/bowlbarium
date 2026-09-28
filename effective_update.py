"""
Effective updates: compare B@A, not A and B separately.

WHY THE RAW COMPARISON MAY UNDERSTATE SIMILARITY
------------------------------------------------
parameter_space.py compared the concatenated LoRA factors and found
every pair of adapters at cosine 0.61-0.64, regardless of condition.
That was read as one dominant shared learning direction.

But LoRA does not apply A and B to the model. It applies their
PRODUCT:

    dW = B @ A        (times the scale factor)

and the factorisation is not unique. For any invertible R,

    (B R) @ (R^-1 A) = B @ A

gives an identical effective update from completely different factors.
So two adapters can encode the same weight change while their raw
parameters look only moderately aligned -- the cosine on factors is a
lower bound on the similarity of what the model actually sees.

This recomputes the comparison on dW, which is what the forward pass
uses and therefore what "the same learning direction" should mean.

WHAT CHANGES
------------
If cosine on dW comes out substantially HIGHER than on the factors,
the shared direction was underestimated and the conditions are even
more alike than they appeared. If it comes out LOWER, the factor-level
alignment was partly an artifact of both adapters starting from the
same initialisation, and the effective updates are more distinct than
they looked.

Either way this is the comparison that corresponds to what the model
does, so it supersedes the factor-level one rather than supplementing
it.

ALSO REPORTED
-------------
    effective rank    from the singular values of dW. LoRA caps rank
                      at 32 by construction, but the spectrum says how
                      many directions are actually used -- a rank-32
                      adapter using two directions is a different
                      object from one using thirty.
    spectral overlap  principal angles between the row spaces of two
                      updates: do they act on the same subspace even
                      if the specific directions differ?

Usage:
    python effective_update.py \\
        --base adapters_child_r32 \\
        --adapter low  runs_hysteresis/LH_adapter_A \\
        --adapter lowC runs_timescale/T850_LH_adapter_A \\
        --adapter high runs_hysteresis/HH_adapter_A \\
        --adapter highB runs_timescale/T850_HH_adapter_A
"""

import argparse
import os
import re
from collections import defaultdict


def load_raw(path):
    from safetensors.numpy import load_file
    f = path if path.endswith(".safetensors") else os.path.join(
        path, "adapters.safetensors")
    if not os.path.exists(f):
        raise SystemExit(f"not found: {f}")
    return load_file(f)


def pair_factors(raw):
    """Group tensors into (module -> {a, b}).

    mlx-lm names them lora_a / lora_b; other conventions use
    lora_A / lora_B or .A / .B, so match loosely.
    """
    out = defaultdict(dict)
    for k, v in raw.items():
        m = re.match(r"(.*?)\.?lora[_.]?([abAB])$", k)
        if m:
            out[m.group(1)][m.group(2).lower()] = v
            continue
        m = re.match(r"(.*)\.(a|b)$", k)
        if m:
            out[m.group(1)][m.group(2)] = v
    return {k: v for k, v in out.items() if "a" in v and "b" in v}


def effective_update(raw):
    """dW = B @ A per module, orientation inferred from shapes."""
    import numpy as np
    pairs = pair_factors(raw)
    if not pairs:
        raise SystemExit(
            "could not identify lora_a / lora_b tensors -- inspect the "
            "key names with:\n"
            "  python -c \"from safetensors.numpy import load_file; "
            "print(list(load_file('PATH/adapters.safetensors'))[:6])\"")
    out = {}
    for name, d in pairs.items():
        A = np.asarray(d["a"], dtype=np.float64)
        B = np.asarray(d["b"], dtype=np.float64)
        # find the orientation whose inner dimensions agree
        for X, Y in ((B, A), (A, B), (B.T, A.T), (A.T, B.T)):
            if X.shape[1] == Y.shape[0]:
                out[name] = X @ Y
                break
    return out


def delta_updates(base_raw, other_raw):
    """Change in the EFFECTIVE update relative to the cold start."""
    import numpy as np
    ub = effective_update(base_raw)
    uo = effective_update(other_raw)
    keys = sorted(set(ub) & set(uo))
    return {k: uo[k] - ub[k] for k in keys if ub[k].shape == uo[k].shape}


def flatten(d):
    import numpy as np
    return np.concatenate([d[k].ravel() for k in sorted(d)])


def cosine(a, b):
    import numpy as np
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else float("nan")


def effective_rank(mats, thresh=0.99):
    """Mean number of singular values needed for `thresh` of the
    spectral energy, averaged over modules."""
    import numpy as np
    ranks = []
    for M in mats:
        s = np.linalg.svd(M, compute_uv=False)
        if s.sum() == 0:
            continue
        e = np.cumsum(s ** 2) / np.sum(s ** 2)
        ranks.append(int(np.searchsorted(e, thresh) + 1))
    return sum(ranks) / len(ranks) if ranks else 0.0


def subspace_overlap(A, B, k=8):
    """Mean cosine of principal angles between the top-k left singular
    subspaces. 1 means the two updates act on the same subspace even if
    their specific directions differ."""
    import numpy as np
    ua, _, _ = np.linalg.svd(A, full_matrices=False)
    ub, _, _ = np.linalg.svd(B, full_matrices=False)
    k = min(k, ua.shape[1], ub.shape[1])
    s = np.linalg.svd(ua[:, :k].T @ ub[:, :k], compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", nargs=2, action="append",
                    metavar=("NAME", "PATH"), required=True)
    ap.add_argument("--rank-threshold", type=float, default=0.99)
    args = ap.parse_args()

    import numpy as np

    print("=" * 78)
    print("EFFECTIVE UPDATES  (dW = B @ A, what the model actually sees)")
    print("=" * 78)

    base_raw = load_raw(args.base)
    pairs = pair_factors(base_raw)
    print(f"\nreference: {args.base}")
    print(f"{len(pairs)} LoRA modules identified")

    deltas, flats = {}, {}
    for name, path in args.adapter:
        d = delta_updates(base_raw, load_raw(path))
        if not d:
            print(f"  {name}: no comparable modules -- skipped")
            continue
        deltas[name] = d
        flats[name] = flatten(d)
        print(f"  {name}: {len(d)} modules, "
              f"||d(dW)|| = {np.linalg.norm(flats[name]):.4f}")

    if len(flats) < 2:
        raise SystemExit("need at least two adapters")

    names = list(flats)

    print(f"\n{'=' * 78}")
    print("COSINE BETWEEN EFFECTIVE UPDATES")
    print(f"{'=' * 78}")
    print(f"\n{'':<10}" + "".join(f"{n:>10}" for n in names))
    print("-" * 78)
    cos = {}
    for a in names:
        row = f"{a:<10}"
        for b in names:
            c = cosine(flats[a], flats[b])
            cos[(a, b)] = c
            row += f"{c:>10.4f}"
        print(row)

    print(f"\n{'=' * 78}")
    print("RANK AND SUBSPACE")
    print(f"{'=' * 78}")
    print(f"\n{'adapter':<10} {'effective rank':>16}   "
          f"(singular values for {args.rank_threshold:.0%} of energy)")
    print("-" * 78)
    for n in names:
        r = effective_rank(list(deltas[n].values()), args.rank_threshold)
        print(f"{n:<10} {r:>16.1f}")

    print(f"\nsubspace overlap (mean cosine of top-8 principal angles):")
    print(f"\n{'':<10}" + "".join(f"{n:>10}" for n in names))
    print("-" * 78)
    mods = sorted(set.intersection(*(set(deltas[n]) for n in names)))
    for a in names:
        row = f"{a:<10}"
        for b in names:
            if a == b:
                row += f"{1.0:>10.4f}"
                continue
            ov = [subspace_overlap(deltas[a][m], deltas[b][m])
                  for m in mods[:12]]
            row += f"{sum(ov)/len(ov):>10.4f}"
        print(row)

    print(f"\n{'=' * 78}")
    print("READING THIS")
    print(f"{'=' * 78}")

    off = [cos[(a, b)] for i, a in enumerate(names) for b in names[i + 1:]]
    print(f"""
off-diagonal cosine on effective updates: {min(off):.3f} to {max(off):.3f}

The factor-level comparison gave 0.61 to 0.64 across all pairs. If
these are HIGHER, the factorisation was hiding similarity and the
shared learning direction is stronger than it appeared. If LOWER, the
factor-level alignment was partly inherited from the shared
initialisation rather than from the learning.""")

    same = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
            if a.rstrip("ABC0123456789_") == b.rstrip("ABC0123456789_")]
    diff = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
            if a.rstrip("ABC0123456789_") != b.rstrip("ABC0123456789_")]
    if same and diff:
        ws = sum(cos[p] for p in same) / len(same)
        ds = sum(cos[p] for p in diff) / len(diff)
        print(f"""
within-condition {ws:.3f}, across-condition {ds:.3f}, gap {ws - ds:+.3f}

A gap near zero on the effective updates is the stronger version of
the earlier finding: the conditions are not merely aligned in their
raw parameters, they induce the same weight change in the model.""")

    print("""
CAVEATS. The orientation of A and B is inferred from tensor shapes; if
the convention differs the products would be wrong, and the module
count printed above is the check. Effective rank uses a 99% energy
threshold, which is arbitrary -- the ordering across adapters matters
more than the absolute number. Subspace overlap is computed on the
first twelve modules for speed.""")


if __name__ == "__main__":
    main()