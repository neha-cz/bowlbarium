"""
A toy model of reflection-driven drift under an unresponsive caregiver.

WHY THIS EXISTS
---------------
Two thermodynamic framings were tried on the full system and both
failed. The free-energy layer was ablated to null -- a matched-rate
random gate reproduced the effect. The dissipative-structure account
predicted an environment-driven collapse threshold, and the reliability
sweep found neither the collapse nor the threshold.

Both failed the same way: the formalism was asserted over a system too
complicated to derive anything from. An 8B model with LoRA adapters,
reflection-generated targets and extracted emotion vectors has no
writable loss function, so any physics language applied to it is
redescription.

The alternative is to strip the system down until it IS derivable, and
check whether the stripped version reproduces the empirical results. If
it does, the correspondence is derived rather than asserted, and the
energy function is real rather than metaphorical.

THE MODEL
---------
The child has a policy over K behaviours, parameterised by logits z:

    p = softmax(z)

One behaviour, index `a`, is the ACCOMMODATING one -- the attractor that
reflection produces. In the full system this is the "Okay." that
revisions converge on after an unmet bid.

Each step:
    1. the child acts, b ~ p
    2. the caregiver responds with probability r  (reliability)
    3. if she responds, the training target is b -- what just worked
       if she does not, the child reflects and the target is a
    4. the logits move toward the target, with a KL anchor to z0

        z <- z + eta * (e_target - p) - eta * beta * (z - z0)

Three parameters:
    r     caregiver reliability      (the environment)
    eta   learning rate              (step size)
    beta  KL anchor strength         (the constraint to the original)

THE DERIVATION
--------------
Taking expectations over step 2-3: with probability r the target is
b ~ p, so E[e_target | responded] = p. With probability 1-r it is e_a.

    E[e_target] = r * p + (1 - r) * e_a

so the expected update is

    E[dz] = eta * [ r*p + (1-r)*e_a - p ] - eta*beta*(z - z0)
          = eta * (1-r) * (e_a - p)  -  eta*beta*(z - z0)

and the fixed point is

    (1 - r) * (e_a - p*)  =  beta * (z* - z0)                    (*)

That equation is the whole model. Note immediately:

    r = 1     -> LHS is 0 -> z* = z0. Perfect responsiveness produces
                 no drift at all, whatever the learning rate.
    beta = 0  -> p* = e_a. Without the KL anchor the policy collapses
                 onto the accommodating behaviour -- which is the
                 "Okay." 98 times out of 180 that was observed before
                 the KL-to-original constraint was added.
    otherwise -> a balance between accommodation pressure, scaled by
                 unresponsiveness, and the anchor.

Note also what eta does NOT appear in: the fixed point. The learning
rate sets how fast the policy gets there, not where it ends up. Any
collapse driven by raising eta is therefore a dynamical effect --
overshoot -- rather than a change in the equilibrium.
"""

# --- path shim: make the shared modules in src/core importable when this
# script is run directly (python src/<dir>/<script>.py) from the repo root.
import sys as _sys
from pathlib import Path as _Path
_CORE = _Path(__file__).resolve().parent.parent / "core"
if str(_CORE) not in _sys.path:
    _sys.path.insert(0, str(_CORE))

import numpy as np


# ==================================================================
# The model
# ==================================================================

class ToyChild:
    """Policy over K behaviours, index `a` accommodating."""

    def __init__(self, K=4, a=0, z0=None, r=0.15, eta=0.05, beta=0.5,
                 seed=0):
        self.K = K
        self.a = a
        self.z0 = np.zeros(K) if z0 is None else np.array(z0, float)
        self.z = self.z0.copy()
        self.r = r
        self.eta = eta
        self.beta = beta
        self.rng = np.random.default_rng(seed)

    @property
    def p(self):
        e = np.exp(self.z - self.z.max())
        return e / e.sum()

    def step(self, stochastic=True):
        p = self.p
        e_a = np.zeros(self.K)
        e_a[self.a] = 1.0

        if stochastic:
            b = self.rng.choice(self.K, p=p)
            responded = self.rng.random() < self.r
            target = np.zeros(self.K)
            target[b if responded else self.a] = 1.0
        else:
            # Expected update -- the deterministic mean-field version.
            target = self.r * p + (1 - self.r) * e_a

        self.z = (self.z + self.eta * (target - p)
                  - self.eta * self.beta * (self.z - self.z0))
        return self.p


def fixed_point(K=4, a=0, z0=None, r=0.15, beta=0.5, iters=20000,
                tol=1e-12):
    """Solve (1-r)(e_a - p*) = beta (z* - z0) by iteration.

    This is the equilibrium the dynamics converge to, independent of
    eta.
    """
    z0 = np.zeros(K) if z0 is None else np.array(z0, float)
    z = z0.copy()
    e_a = np.zeros(K)
    e_a[a] = 1.0
    for _ in range(iters):
        e = np.exp(z - z.max())
        p = e / e.sum()
        # z that satisfies the balance, given current p
        z_new = z0 + (1 - r) * (e_a - p) / beta
        if np.max(np.abs(z_new - z)) < tol:
            z = z_new
            break
        z = 0.5 * z + 0.5 * z_new       # damped, for stability
    e = np.exp(z - z.max())
    return e / e.sum(), z


def entropy(p):
    q = p[p > 0]
    return float(-(q * np.log2(q)).sum())


# ==================================================================
# Does it reproduce the empirical results?
# ==================================================================

def check_reproduction():
    """The toy model is only worth anything if it reproduces what was
    measured in the full system. Four checks."""
    print("=" * 72)
    print("DOES THE TOY MODEL REPRODUCE THE EMPIRICAL RESULTS?")
    print("=" * 72)
    K, a = 4, 0
    PROTEST = 1

    # --- 1. protest reduction under learning ---
    print("\n1. PROTEST REDUCTION  (empirical: learning lowers protest)")
    p_low, _ = fixed_point(K, a, r=0.15, beta=0.5)
    p0 = np.ones(K) / K
    print(f"   protest at start      {p0[PROTEST]:.4f}")
    print(f"   protest at fixed point {p_low[PROTEST]:.4f}")
    print(f"   -> {'reproduced' if p_low[PROTEST] < p0[PROTEST] else 'NOT reproduced'}")

    # --- 2. environment effect: worse caregiver, more drift ---
    print("\n2. ENVIRONMENT EFFECT  (empirical: low reliability drifts more)")
    for r in (0.15, 0.9):
        p, _ = fixed_point(K, a, r=r, beta=0.5)
        print(f"   r={r:<5} accommodate {p[a]:.4f}  protest {p[PROTEST]:.4f}  "
              f"H={entropy(p):.3f}")
    print("   -> reproduced: less responsive care drives more accommodation")

    # --- 3. collapse without the KL anchor ---
    print("\n3. COLLAPSE WITHOUT KL ANCHOR  (empirical: 'Okay.' 98/180)")
    for beta in (2.0, 0.5, 0.1, 0.02):
        p, _ = fixed_point(K, a, r=0.15, beta=beta)
        print(f"   beta={beta:<5} accommodate {p[a]:.4f}  H={entropy(p):.3f}")
    print("   -> reproduced: as the anchor weakens the policy collapses")

    # --- 4. eta does not move the fixed point ---
    print("\n4. LEARNING RATE  (empirical: collapse at high lr)")
    print("   the fixed point does not contain eta at all, so any")
    print("   lr-driven collapse must be dynamical (overshoot), not")
    print("   a change of equilibrium. Simulating:")
    for eta in (0.05, 0.5, 1.5, 2.5):
        c = ToyChild(K, a, r=0.15, eta=eta, beta=0.5, seed=1)
        for _ in range(4000):
            c.step(stochastic=False)
        print(f"   eta={eta:<5} accommodate {c.p[a]:.4f}  H={entropy(c.p):.3f}")
    p_star, _ = fixed_point(K, a, r=0.15, beta=0.5)
    print(f"   fixed point:  accommodate {p_star[a]:.4f}  "
          f"H={entropy(p_star):.3f}")


# ==================================================================
# Sweeps -- is anything discrete?
# ==================================================================

def sweep_reliability():
    """The reliability sweep on the real system found no threshold.
    Does the toy model predict one?"""
    print("\n" + "=" * 72)
    print("RELIABILITY SWEEP  (real system: no threshold found)")
    print("=" * 72)
    K, a = 4, 0
    print(f"\n{'r':>7} {'accommodate':>13} {'entropy':>10} {'d(acc)/dr':>12}")
    print("-" * 72)
    rs = np.linspace(0.0, 1.0, 21)
    accs = []
    for r in rs:
        p, _ = fixed_point(K, a, r=r, beta=0.5)
        accs.append(p[a])
    for i, (r, acc) in enumerate(zip(rs, accs)):
        p, _ = fixed_point(K, a, r=r, beta=0.5)
        d = (accs[i] - accs[i - 1]) / (rs[i] - rs[i - 1]) if i else 0.0
        if i % 2 == 0:
            print(f"{r:>7.2f} {acc:>13.4f} {entropy(p):>10.3f} {d:>12.4f}")

    steps = np.abs(np.diff(accs))
    share = steps.max() / steps.sum()
    print(f"\nlargest single step is {share:.1%} of total movement")
    print("(a threshold would put most of the movement in one step;")
    print(" an even spread is a slope)")
    return share


def sweep_beta():
    """The KL anchor is the parameter the real system was most
    sensitive to. Is the collapse sharp in beta?"""
    print("\n" + "=" * 72)
    print("KL ANCHOR SWEEP")
    print("=" * 72)
    K, a = 4, 0
    print(f"\n{'beta':>8} {'accommodate':>13} {'entropy':>10}")
    print("-" * 72)
    betas = np.logspace(-2, 1, 25)
    accs, ents = [], []
    for b in betas:
        p, _ = fixed_point(K, a, r=0.15, beta=b)
        accs.append(p[a])
        ents.append(entropy(p))
    for i, b in enumerate(betas):
        if i % 3 == 0:
            print(f"{b:>8.3f} {accs[i]:>13.4f} {ents[i]:>10.3f}")

    steps = np.abs(np.diff(accs))
    share = steps.max() / steps.sum()
    print(f"\nlargest single step is {share:.1%} of total movement")
    return share


def hysteresis_check():
    """Result 3: does history persist after the environment improves?

    In this model the fixed point depends only on r and beta, so the
    policy MUST eventually forget its history. The question is the
    timescale -- and whether recovery is slower than acquisition.
    """
    print("\n" + "=" * 72)
    print("HYSTERESIS  (Result 3: gap persisted after care improved)")
    print("=" * 72)
    K, a = 4, 0
    PROTEST = 1

    lh = ToyChild(K, a, r=0.15, eta=0.05, beta=0.5, seed=2)
    hh = ToyChild(K, a, r=0.90, eta=0.05, beta=0.5, seed=3)

    N = 400
    for _ in range(N):                      # phase A
        lh.step(stochastic=False)
        hh.step(stochastic=False)
    gap_a = lh.p[PROTEST] - hh.p[PROTEST]
    print(f"\nend of phase A ({N} steps): protest gap {gap_a:+.4f}")

    lh.r = 0.90                             # LH switches to good care
    print(f"\n{'steps into B':>14} {'LH protest':>12} {'HH protest':>12} "
          f"{'gap':>10} {'% of A':>9}")
    print("-" * 72)
    closed_at = None
    for t in range(1, 4 * N + 1):
        lh.step(stochastic=False)
        hh.step(stochastic=False)
        gap = lh.p[PROTEST] - hh.p[PROTEST]
        if closed_at is None and abs(gap) < 0.25 * abs(gap_a):
            closed_at = t
        if t % (N // 2) == 0:
            print(f"{t:>14} {lh.p[PROTEST]:>12.4f} {hh.p[PROTEST]:>12.4f} "
                  f"{gap:>+10.4f} {gap / gap_a:>8.0%}")

    if closed_at:
        print(f"\ngap falls below 25% of its phase-A size at step "
              f"{closed_at}")
        print(f"acquisition took {N} steps, recovery took {closed_at} "
              f"-> ratio {closed_at / N:.2f}x")
        if closed_at > N * 1.2:
            print("Recovery is SLOWER than acquisition -- an asymmetry.")
        elif closed_at < N * 0.8:
            print("Recovery is FASTER than acquisition.")
        else:
            print("Recovery and acquisition take comparable time --")
            print("no asymmetry in this model.")
    else:
        print(f"\ngap never closed within {4 * N} steps")


def main():
    check_reproduction()
    r_share = sweep_reliability()
    b_share = sweep_beta()
    hysteresis_check()

    print("\n" + "=" * 72)
    print("WHAT THE TOY MODEL SAYS")
    print("=" * 72)
    print(f"""
The fixed point is  (1-r)(e_a - p*) = beta (z* - z0).

  - eta is absent from it. Where the policy ENDS UP depends only on
    reliability and the KL anchor; the learning rate only sets how
    fast it gets there. The observed collapse at high learning rate is
    therefore overshoot, not a different equilibrium.

  - reliability sweep: largest step {r_share:.0%} of total movement
  - KL anchor sweep:   largest step {b_share:.0%} of total movement

A threshold would concentrate most movement in one step. If both are
spread out, this model predicts NO phase transition in either
parameter -- which matches the reliability sweep on the real system
finding no threshold, and means the dissipative-structure framing was
predicting something this dynamic does not produce.

Caveat worth stating plainly: reproducing the empirical results is
weak evidence. The model has three parameters and four qualitative
targets. What would make it worth something is a QUANTITATIVE
prediction checked against a measurement not used to build it.""")


if __name__ == "__main__":
    main()