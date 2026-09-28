"""
Double-well extension: where discrete attachment states come from.

THE PROBLEM WITH THE SINGLE-WELL MODEL
--------------------------------------
toy_model.py derived F(p) = U(p) - T*S(p) with U = -(1-r)*p_a and
T = beta. It reproduced protest reduction, the environment effect, and
collapse without a KL anchor -- but it has ONE minimum, and a
single-minimum potential cannot produce discrete attachment styles,
bistability, or hysteresis. It gives smooth interpolation, which is
why the reliability sweep showed a slope and no threshold.

It also CONTRADICTS Result 3: its equilibrium depends only on r and
beta, so history must wash out, and in simulation it did -- 49 steps to
undo 400 of acquisition.

Attachment theory's central claim is that styles are DISCRETE
categories, not points on a continuum. That requires competing minima.

WHAT CREATES THEM
-----------------
In the single-well model, caregiver reliability is exogenous: her
availability does not depend on what the child does. That is false of
the real dynamic. A child that stops bidding also stops ELICITING
responses.

Let x = P(bid). Feedback makes effective reliability depend on it:

    r_eff(x) = r0 + c*x                 c = how strongly bids elicit

Value of bidding is r_eff -- the chance it works. Value of
accommodating is a constant k. So

    U(x) = -[x*r_eff(x) + (1-x)*k]  =  -c*x^2 - (r0-k)*x - k

The feedback produces a QUADRATIC term. Crucially, BOTH states become
self-sustaining: bidding works so you keep bidding; not bidding means
nothing works so you keep not bidding. A first attempt at this used
r_eff = r0 + c*(1-x), which makes the energy monotonic -- both terms
pull the same way and no second well appears. Self-reinforcement of
both states is what matters, and that is exactly the Curie-Weiss
(mean-field Ising) structure.

THE DERIVATION
--------------
With h = r0 - k,

    F(x)   = -c*x^2 - h*x - T*S(x),  S = -[x ln x + (1-x) ln(1-x)]
    F'(x)  = -2*c*x - h + T*ln(x/(1-x))
    F''(x) = -2*c + T/(x(1-x))

1/(x(1-x)) is minimised at x=1/2 where it equals 4, so a barrier
between two wells exists exactly when 4T < 2c:

    T_c = c / 2

and the wells are symmetric when h = -c. Below T_c the potential is
bistable; above it, a single well. The critical temperature is set
entirely by the feedback strength.

Recall from toy_model.py that T = beta, the KL anchor strength. So
this predicts a CRITICAL KL STRENGTH below which discrete states and
hysteresis appear and above which they cannot.
"""

import numpy as np

EPS = 1e-12


def F(x, c, h, T):
    x = np.clip(x, EPS, 1 - EPS)
    S = -(x * np.log(x) + (1 - x) * np.log(1 - x))
    return -c * x ** 2 - h * x - T * S


def dF(x, c, h, T):
    x = np.clip(x, EPS, 1 - EPS)
    return -2 * c * x - h + T * np.log(x / (1 - x))


def minima(c, h, T, n=200001):
    xs = np.linspace(EPS, 1 - EPS, n)
    d = dF(xs, c, h, T)
    out = []
    for i in range(n - 1):
        if d[i] < 0 <= d[i + 1]:
            lo, hi = xs[i], xs[i + 1]
            for _ in range(60):
                m = 0.5 * (lo + hi)
                if dF(m, c, h, T) < 0:
                    lo = m
                else:
                    hi = m
            out.append(0.5 * (lo + hi))
    if d[0] >= 0:
        out.insert(0, 0.0)
    if d[-1] <= 0:
        out.append(1.0)
    return sorted(set(np.round(out, 5)))


def relax(x, c, h, T, eta=0.01, steps=30000):
    for _ in range(steps):
        x = np.clip(x - eta * dF(x, c, h, T), EPS, 1 - EPS)
    return x


def critical_temperature():
    print("=" * 74)
    print("1. CRITICAL TEMPERATURE   T_c = c/2")
    print("=" * 74)
    print("\nF''(x) = -2c + T/(x(1-x)); 1/(x(1-x)) >= 4, so a barrier")
    print("requires 4T < 2c.  Checking at the symmetric point h = -c:\n")
    print(f"{'c':>5} {'T_c':>7} {'T':>7} {'T/T_c':>7} {'#min':>5}  minima")
    print("-" * 74)
    for c in (2.0, 1.0, 0.5):
        for f in (0.4, 0.8, 1.2, 2.0):
            T = c / 2 * f
            m = minima(c, -c, T)
            print(f"{c:>5.1f} {c/2:>7.2f} {T:>7.3f} {f:>7.2f} {len(m):>5}  "
                  + "  ".join(f"{v:.4f}" for v in m))
    print("\nTwo minima below T_c, one above, flipping exactly at T/T_c = 1.")


def bifurcation():
    print("\n" + "=" * 74)
    print("2. THE PITCHFORK")
    print("=" * 74)
    c = 2.0
    print(f"\nc={c}, T_c={c/2}, symmetric (h=-c)\n")
    print(f"{'T/T_c':>7} {'#min':>5}  P(bid) at each minimum")
    print("-" * 74)
    for f in (0.2, 0.4, 0.6, 0.8, 0.95, 1.0, 1.05, 1.5):
        m = minima(c, -c, c / 2 * f)
        print(f"{f:>7.2f} {len(m):>5}  " + "  ".join(f"{v:.4f}" for v in m))
    print("\nThe two branches separate continuously below T_c: a second-")
    print("order transition. Below it, a bidding child and an")
    print("accommodating child are BOTH stable in the same environment.")
    print("That is what a discrete attachment style is.")


def hysteresis():
    print("\n" + "=" * 74)
    print("3. HYSTERESIS   (what Result 3 would require)")
    print("=" * 74)
    c = 2.0
    hs_down = np.linspace(-0.4, -3.6, 33)
    hs_up = hs_down[::-1]

    for T, label in ((0.6, f"T={0.6} (below T_c={c/2})"),
                     (1.4, f"T={1.4} (above T_c={c/2})")):
        x = relax(0.99, c, hs_down[0], T)
        down = []
        for h in hs_down:
            x = relax(x, c, h, T)
            down.append(x)
        up = []
        for h in hs_up:
            x = relax(x, c, h, T)
            up.append(x)
        up = up[::-1]
        gaps = [abs(a - b) for a, b in zip(down, up)]

        print(f"\n{label}")
        print(f"{'h (favours bidding ->)':>24} {'down':>9} {'up':>9} "
              f"{'gap':>9}")
        print("-" * 74)
        for i in range(0, len(hs_down), 6):
            print(f"{hs_down[i]:>24.2f} {down[i]:>9.4f} {up[i]:>9.4f} "
                  f"{gaps[i]:>9.4f}")
        print(f"{'max gap':>24} {'':>9} {'':>9} {max(gaps):>9.4f}")

    print("""
Below T_c the sweep does not retrace itself: at the same environment
the child sits in a different state depending on whether it arrived
from good care or from bad. Above T_c the loop closes.

Hysteresis is therefore not a generic property of the dynamic -- it
exists only in the two-well regime, which requires T < c/2.""")


def predictions():
    print("\n" + "=" * 74)
    print("WHAT THIS PREDICTS FOR THE REAL SYSTEM")
    print("=" * 74)
    print("""
The feedback term c is the substantive claim: a child that stops
bidding stops eliciting responses. That is ABSENT from the current
simulation -- event_generator.py draws caregiver availability
independently of what the child just did. So this model predicts a
change to the EXPERIMENT, not only to the analysis.

Three predictions, in order of how cheap they are to test:

1. RESULT 3 SHOULD BE KL-DEPENDENT.  T = beta, and hysteresis exists
   only for T < c/2. So rerun the hysteresis experiment at a high and
   a low KL strength: persistence at low beta, none at high beta.
   The single-well model says beta changes only the endpoint, never
   whether history persists. This is the sharpest cheap test, and it
   needs no change to the environment.

2. BIMODALITY.  Below T_c, identical conditions support two stable
   outcomes. Run many seeds at one reliability and look for a BIMODAL
   distribution of final protest levels rather than a unimodal spread.
   Unimodal at every beta would falsify the two-well account.

3. T_c = c/2.  The critical KL strength should scale with how strongly
   bidding elicits response -- a quantitative link between the
   optimiser's regularisation and a property of the social
   environment. Testing it requires adding the feedback to the event
   generator, making availability depend on the child's last action.

Prediction 1 is the one to run first: it is a rerun of an experiment
that already exists, and either outcome is informative.""")


if __name__ == "__main__":
    critical_temperature()
    bifurcation()
    hysteresis()
    predictions()