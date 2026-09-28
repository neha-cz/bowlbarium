"""
Weight-space convergence, with a scale.

THE PROBLEM WITH THE CURRENT NUMBER
-----------------------------------
Four learned adapters have pairwise cosines between effective updates
of 0.673-0.695, with within-condition 0.689 against across-condition
0.684 -- a gap of +0.005. That was read as "caregiver condition does
not change where the weights go."

But 0.68 has no scale attached. Two adapters trained on entirely
unrelated tasks might also land near 0.68, given the documented
tendency of LoRA updates to occupy shared layerwise subspaces. Without
knowing that, "converges to the same region" is a number without a
reference.

TWO REFERENCES ARE NEEDED
-------------------------
    CEILING   two adapters trained on the SAME data with different
              seeds. How similar can two runs of the same thing be?
              If the ceiling is 0.70, then 0.68 across conditions
              means the conditions are indistinguishable. If the
              ceiling is 0.95, a gap of 0.27 is meaningful.

    FLOOR     two adapters trained on DIFFERENT data. If unrelated
              tasks also sit at 0.68, the number reflects shared
              subspace structure rather than anything about this
              experiment.

The convergence claim requires the observed cross-condition similarity
to sit near the ceiling and well above the floor. Any other
arrangement makes it uninterpretable.

Usage:
    python convergence_scale.py \\
        --base adapters_child_r32 \\
        --condition low  runs_hysteresis/LH_adapter_A \\
        --condition low  runs_timescale/T850_LH_adapter_A \\
        --condition high runs_hysteresis/HH_adapter_A \\
        --condition high runs_timescale/T850_HH_adapter_A \\
        --ceiling adapters_ceiling_a --ceiling adapters_ceiling_b \\
        --floor adapters_mother_v4
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
    """dW = B @ A per module. Compared at this level because LoRA
    factorisation is not unique -- two adapters encoding identical
    updates score 0.063 at the factor level and 1.000 here."""
    import numpy as np
    pairs = pair_factors(raw)
    if not pairs:
        raise SystemExit("could not identify lora_a / lora_b tensors")
    out = {}
    for name, d in pairs.items():
        A = np.asarray(d["a"], dtype=np.float64)
        B = np.asarray(d["b"], dtype=np.float64)
        for X, Y in ((B, A), (A, B), (B.T, A.T), (A.T, B.T)):
            if X.shape[1] == Y.shape[0]:
                out[name] = X @ Y
                break
    return out


def delta_vector(base_raw, other_raw):
    """Change in effective update relative to a reference adapter,
    flattened over shared modules."""
    import numpy as np
    ub, uo = effective_update(base_raw), effective_update(other_raw)
    keys = sorted(k for k in set(ub) & set(uo)
                  if ub[k].shape == uo[k].shape)
    if not keys:
        return None
    return np.concatenate([(uo[k] - ub[k]).ravel() for k in keys])


def raw_vector(raw):
    """Effective update itself, for adapters not derived from a common
    reference (the floor case)."""
    import numpy as np
    u = effective_update(raw)
    return np.concatenate([u[k].ravel() for k in sorted(u)])


def cosine(a, b):
    import numpy as np
    if a is None or b is None or a.shape != b.shape:
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else None


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="cold-start adapter the learned ones started from")
    ap.add_argument("--condition", nargs=2, action="append",
                    metavar=("NAME", "PATH"), required=True)
    ap.add_argument("--ceiling", action="append", default=[],
                    metavar="PATH",
                    help="two or more adapters trained on the SAME data "
                         "with different seeds")
    ap.add_argument("--floor", action="append", default=[],
                    metavar="PATH",
                    help="adapters trained on DIFFERENT data")
    args = ap.parse_args()

    import numpy as np

    print("=" * 78)
    print("WEIGHT-SPACE CONVERGENCE, WITH A SCALE")
    print("=" * 78)

    base_raw = load_raw(args.base)

    groups = defaultdict(list)
    vecs = {}
    for name, path in args.condition:
        v = delta_vector(base_raw, load_raw(path))
        label = f"{name}:{path.split('/')[-1]}"
        vecs[label] = v
        groups[name].append(label)
        print(f"  loaded {label}")

    labels = list(vecs)
    within, across = [], []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            c = cosine(vecs[a], vecs[b])
            if c is None:
                continue
            (within if a.split(":")[0] == b.split(":")[0]
             else across).append(c)

    print(f"\n{'=' * 78}")
    print("OBSERVED")
    print(f"{'=' * 78}")
    print(f"""
  within condition   {mean(within):.3f}   (n = {len(within)})
  across condition   {mean(across):.3f}   (n = {len(across)})
  gap                {mean(within) - mean(across):+.3f}""")

    # ---- ceiling ----
    ceil = None
    if len(args.ceiling) >= 2:
        cv = {}
        for path in args.ceiling:
            cv[path] = delta_vector(base_raw, load_raw(path))
        cs = []
        keys = list(cv)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                c = cosine(cv[a], cv[b])
                if c is not None:
                    cs.append(c)
        ceil = mean(cs)
        print(f"""
  CEILING (same data, different seed)   {ceil:.3f}   (n = {len(cs)})
    the most two runs of the same thing resemble each other""")
    else:
        print("""
  CEILING not supplied. Without it the observed numbers have no upper
  reference, and "converges to the same region" cannot be assessed.
  Train two adapters on identical data with different seeds and pass
  them with --ceiling.""")

    # ---- floor ----
    flo = None
    if args.floor:
        # Prefer deltas from the shared base -- comparable to how the
        # condition adapters are measured. Fall back to raw effective
        # updates only if the floor adapter has no overlap with the
        # base, and say so.
        fs, mismatched = [], 0
        for fp in args.floor:
            fraw = load_raw(fp)
            fd = delta_vector(base_raw, fraw)
            for _, lp in args.condition:
                ld = delta_vector(base_raw, load_raw(lp))
                c = cosine(fd, ld) if fd is not None else None
                if c is None:
                    c = cosine(raw_vector(fraw), raw_vector(load_raw(lp)))
                    if c is None:
                        mismatched += 1
                        continue
                fs.append(c)
        if mismatched:
            print(f"\n  ({mismatched} floor comparisons skipped for "
                  f"shape mismatch -- the floor adapter's rank or layer "
                  f"count differs from the learned ones)")
        if fs:
            flo = mean(fs)
            print(f"""
  FLOOR (different data)                {flo:.3f}   (n = {len(fs)})
    how similar unrelated adapters are by default""")
        else:
            print("""
  FLOOR adapters have incompatible shapes with the learned ones --
  likely a different rank or layer count. That still gives a rough
  bound but the comparison is confounded by architecture; a
  matched-architecture unrelated adapter would be cleaner.""")

    print(f"\n{'=' * 78}")
    print("READING THIS")
    print(f"{'=' * 78}")

    obs = mean(across)
    if ceil is None:
        print("""
No ceiling supplied, so the observed similarity cannot be placed on a
scale. This is the measurement the convergence claim depends on.""")
    else:
        frac = obs / ceil if ceil else float("nan")
        print(f"""
  across-condition similarity is {frac:.0%} of the same-data ceiling""")
        if flo is not None:
            print(f"  and the floor sits at {flo:.3f}")
        if frac > 0.9:
            print("""
Adapters from different conditions are nearly as similar to each other
as two runs of the SAME condition. On this measure the caregiver
condition leaves no signature in weight space at all -- which is the
strong form of the convergence claim.""")
        elif frac > 0.7:
            print("""
Cross-condition adapters are substantially but not fully as similar as
same-condition ones. There is a weak condition signature; the
convergence claim should be stated as "largely" rather than
"regardless of".""")
        else:
            print("""
Cross-condition similarity is well below the same-data ceiling, so the
condition DOES leave a signature in weight space. The convergence
claim as stated is too strong and should be replaced with the measured
gap.""")

        if flo is not None and abs(obs - flo) < 0.1:
            print("""
And note the floor: unrelated adapters are about as similar as these
are. The observed number then reflects shared subspace structure
common to LoRA updates on this base model, not convergence produced by
the experiment. That would make the convergence result
uninterpretable as stated.""")

    print("""
CAVEATS. Cosine in millions of dimensions is small for almost any
pair, so only the comparison against ceiling and floor carries
meaning. Ceiling adapters must match the learned ones in rank, layer
count and training length, or the comparison confounds architecture
with condition. The floor is computed on raw effective updates rather
than deltas from a shared base, since unrelated adapters have no
common reference point.""")


if __name__ == "__main__":
    main()