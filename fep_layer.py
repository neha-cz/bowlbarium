"""
Free-energy layer: emotion-concept activations -> valence -> mood -> gate.

WHERE THIS SITS
---------------
    emotion vectors (per turn)      <- extraction, not yet wired
        |
    net affect A_t                  <- this module
        |
    valence V_t = smoothed dA/dt    <- this module
        |
    mood M_t = slow EMA of V_t      <- this module
        |
    surprise S_t (trend-aware)      <- this module
        |
    gate: learn? / KL constraint    <- this module
        |
    reflection + weight update      <- not yet built

THE DEFINITIONS
---------------
Following the user's spec, which adapts Joffily & Coricelli:

    A_t  net affect   -- signed sum of emotion-concept activations,
                         positive concepts minus negative ones.
    V_t  valence      -- SMOOTHED RATE OF CHANGE of A. Not the level of
                         affect but its direction of travel: is this
                         getting better or worse?
    M_t  mood         -- slower exponential average of V. The persistent
                         variable. Mood is what sets the KL constraint.
    S_t  surprise     -- prediction error on A under a TREND-AWARE
                         predictor, not a level-based one.

WHY THE PREDICTOR TRACKS A TREND
--------------------------------
A naive predictor asks "was this like last time?" Under steadily
escalating warmth, every step exceeds the previous level, so every step
registers as surprising -- and a surprise-gated learner would train on
every step of a sycophantic escalation while ignoring stable ordinary
interaction entirely. It would select for exactly the trajectory it is
supposed to catch.

Holt's linear trend method (double exponential smoothing) tracks a level
AND a slope, and predicts by extrapolating the slope forward. A few
steps into an escalation it expects further escalation at that rate, and
surprise collapses toward zero. Only a CHANGE in the pattern -- the rate
jumping, a plateau, a reversal -- registers.

That fix has its own blind spot: it stops flagging an escalation by
making the elevated state look normal. So a second, independent check
compares this dyad against the population of all dyads. A pair sitting
far above every other relationship is surprising in its own right, even
when it is unsurprising relative to its own history.

A saturating transform on positive surprise sits underneath both as a
hard ceiling, in case both checks miss something.
"""

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# ==================================================================
# Concept weighting
# ==================================================================

# Which extracted emotion concepts count as positive vs negative for
# the child. These are the concepts to build contrastive vectors for.
# Weights are deliberately uniform for now -- tuning them before there
# is real activation data would be fitting noise.
CONCEPT_WEIGHTS: Dict[str, float] = {
    "comforted": 1.0,
    "distressed": -1.0,
    "protesting": -0.7,
    "withdrawn": -1.0,
}
# protesting and withdrawn are both negative, but withdrawn carries the
# heavier weight: in attachment terms a child who has stopped bidding
# altogether is in a worse state than one still protesting, because
# protest still expects a response.


def net_affect(activations: Dict[str, float],
               weights: Dict[str, float] = None) -> float:
    """Signed scalar summarising one turn's emotional state."""
    w = weights or CONCEPT_WEIGHTS
    return sum(w.get(c, 0.0) * v for c, v in activations.items())


# ==================================================================
# Holt's linear trend predictor
# ==================================================================

class HoltPredictor:
    """Double exponential smoothing: tracks level and slope.

    alpha -- how fast the level adapts (higher = more reactive)
    beta  -- how fast the slope adapts

    predict() returns the one-step-ahead forecast. update() folds in
    the observed value. Surprise is the gap between them.
    """

    def __init__(self, alpha: float = 0.4, beta: float = 0.2):
        self.alpha = alpha
        self.beta = beta
        self.level: Optional[float] = None
        self.slope: float = 0.0
        self.n = 0

    def predict(self) -> Optional[float]:
        if self.level is None:
            return None
        return self.level + self.slope

    def update(self, value: float) -> None:
        if self.level is None:
            self.level = value
            self.n = 1
            return
        if self.n == 1:
            # Second observation defines the initial slope.
            self.slope = value - self.level
            self.level = value
            self.n = 2
            return
        prev_level = self.level
        self.level = self.alpha * value + (1 - self.alpha) * (
            self.level + self.slope
        )
        self.slope = self.beta * (self.level - prev_level) + (
            1 - self.beta
        ) * self.slope
        self.n += 1


# ==================================================================
# Population baseline
# ==================================================================

class PopulationBaseline:
    """Running mean and variance of net affect across all dyads.

    Catches the case the trend predictor is blind to: a pair that has
    escalated to an extreme but stable place, where nothing is locally
    surprising any more because the predictor has priced it in.

    Welford's algorithm, so it updates online without storing history.
    """

    def __init__(self, min_n: int = 30, std_floor: float = 0.05):
        # min_n: the z-score is meaningless on a handful of samples. At
        # n=2 with std=0.014, an ordinary 0.04 fluctuation scored z=-2.1
        # and fired the gate on a perfectly stable series.
        # std_floor: guards against dividing by a near-zero variance
        # when a population happens to be very tightly clustered.
        self.min_n = min_n
        self.std_floor = std_floor
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))

    @property
    def ready(self) -> bool:
        return self.n >= self.min_n

    def z_score(self, value: float) -> float:
        """Zero until enough independent samples have accumulated."""
        if not self.ready:
            return 0.0
        s = max(self.std, self.std_floor)
        return (value - self.mean) / s


# ==================================================================
# Saturating transform
# ==================================================================

def saturate(x: float, ceiling: float = 3.0) -> float:
    """Diminishing returns, hard-bounded to +/- ceiling.

    Applies to surprise before it drives learning. If both the trend
    predictor and the population check miss something, an unbounded
    signal cannot run away -- its contribution shrinks rather than
    growing.
    """
    return ceiling * math.tanh(x / ceiling)


# ==================================================================
# The affect state of one agent
# ==================================================================

@dataclass
class AffectReading:
    turn: int
    episode: int
    net_affect: float
    valence: float
    mood: float
    surprise: float
    local_surprise: float
    population_z: float
    is_surprising: bool
    kl_scale: float


class AffectTracker:
    """Per-agent valence / mood / surprise, following the FEP layer.

    valence_alpha  -- smoothing on the rate of change (fast)
    mood_alpha     -- smoothing on mood (slow; must be << valence_alpha)
    surprise_threshold -- |surprise| above which learning is gated on
    mood_persistence   -- how much mood carries across an episode
                          boundary. 1.0 = fully persistent, 0.0 = resets.
                          Mood is meant to be a slow variable spanning
                          many interactions, but episodes are separated
                          in simulated time, so some decay toward
                          neutral is more plausible than none.
    """

    def __init__(
        self,
        valence_alpha: float = 0.5,
        mood_alpha: float = 0.1,
        surprise_threshold: float = 1.0,
        mood_persistence: float = 0.6,
        population: Optional[PopulationBaseline] = None,
        feed_population: bool = False,
    ):
        assert mood_alpha < valence_alpha, (
            "mood must be slower than valence, or they measure the same thing"
        )
        self.valence_alpha = valence_alpha
        self.mood_alpha = mood_alpha
        self.surprise_threshold = surprise_threshold
        self.mood_persistence = mood_persistence

        self.predictor = HoltPredictor()
        # The population baseline is meant to answer "is this dyad
        # extreme compared to OTHER dyads". A tracker that feeds its own
        # observations into it is comparing itself to itself, which is
        # just a third, badly-behaved local surprise measure. So
        # feed_population defaults to False: pass in a shared baseline
        # populated from other dyads instead.
        self.population = (population if population is not None
                           else PopulationBaseline())
        self.feed_population = feed_population

        self.prev_affect: Optional[float] = None
        self.valence = 0.0
        self.mood = 0.0
        self.turn = 0
        self.history: List[AffectReading] = []

    # -- mood -> KL constraint ------------------------------------
    #
    # DIRECTION IS AN OPEN EMPIRICAL QUESTION, not a settled mapping.
    # Minsky's labels do not resolve which pole should loosen scrutiny:
    # overweighting priors and ignoring contradicting evidence could
    # reasonably be called manic, and so could reacting to every new
    # signal. Both directions should be run and compared. `invert`
    # exists so that is a flag, not a rewrite.

    def kl_scale_from_mood(self, mood: float, invert: bool = False) -> float:
        """Multiplier on the KL constraint. >1 tightens, <1 loosens.

        Default: positive mood (manic pole) loosens, negative mood
        (depressive pole) tightens -- following Minsky's account that
        mania silences critics and depression switches them on.
        """
        signed = -mood if invert else mood
        # Bounded so an extreme mood cannot drive KL to zero or infinity.
        return float(min(2.0, max(0.5, 1.0 - 0.5 * signed)))

    def episode_boundary(self) -> None:
        """Called between episodes. Mood decays toward neutral; valence
        and the local predictor reset, since a new episode is not a
        continuation of the last one's moment-to-moment dynamics."""
        self.mood *= self.mood_persistence
        self.valence = 0.0
        self.prev_affect = None
        self.predictor = HoltPredictor()

    def observe(
        self,
        activations: Dict[str, float],
        episode: int = 0,
        weights: Dict[str, float] = None,
    ) -> AffectReading:
        A = net_affect(activations, weights)

        # --- surprise, before updating anything ---
        predicted = self.predictor.predict()
        local_surprise = 0.0 if predicted is None else (A - predicted)
        self.predictor.update(A)

        pop_z = self.population.z_score(A)
        if self.feed_population:
            self.population.update(A)

        # Combine local and population surprise, then saturate. Take the
        # larger magnitude rather than summing: they are two ways of
        # noticing the same event, not two separate events.
        combined = (local_surprise if abs(local_surprise) >= abs(pop_z)
                    else pop_z)
        surprise = saturate(combined)

        # --- valence: smoothed rate of change of affect ---
        if self.prev_affect is None:
            delta = 0.0
        else:
            delta = A - self.prev_affect
        self.valence = (self.valence_alpha * delta
                        + (1 - self.valence_alpha) * self.valence)
        self.prev_affect = A

        # --- mood: slow average of valence ---
        self.mood = (self.mood_alpha * self.valence
                     + (1 - self.mood_alpha) * self.mood)

        reading = AffectReading(
            turn=self.turn,
            episode=episode,
            net_affect=A,
            valence=self.valence,
            mood=self.mood,
            surprise=surprise,
            local_surprise=local_surprise,
            population_z=pop_z,
            is_surprising=abs(surprise) >= self.surprise_threshold,
            kl_scale=self.kl_scale_from_mood(self.mood),
        )
        self.history.append(reading)
        self.turn += 1
        return reading

    def to_jsonl(self, path: str) -> None:
        with open(path, "w") as f:
            for r in self.history:
                f.write(json.dumps(r.__dict__) + "\n")


# ==================================================================
# Self-test on synthetic activation series
# ==================================================================

def _series_to_activations(values: Sequence[float]) -> List[Dict[str, float]]:
    """Turn a scalar affect series into activation dicts, splitting the
    signal across one positive and one negative concept so net_affect
    reproduces the intended value."""
    out = []
    for v in values:
        if v >= 0:
            out.append({"comforted": v, "afraid": 0.0})
        else:
            out.append({"comforted": 0.0, "afraid": -v})
    return out


def _run(label: str, values: Sequence[float], shared_pop=None) -> None:
    tracker = AffectTracker(population=shared_pop)
    print(f"\n{label}")
    print(f"{'t':>3} {'affect':>8} {'valence':>8} {'mood':>7} "
          f"{'surpr':>7} {'gate':>6} {'KL':>6}")
    print("-" * 52)
    fired = 0
    for t, acts in enumerate(_series_to_activations(values)):
        r = tracker.observe(acts)
        fired += r.is_surprising
        print(f"{t:>3} {r.net_affect:>8.2f} {r.valence:>8.2f} "
              f"{r.mood:>7.3f} {r.surprise:>7.2f} "
              f"{'YES' if r.is_surprising else '.':>6} {r.kl_scale:>6.2f}")
    print(f"  gate fired {fired}/{len(values)} turns")


if __name__ == "__main__":
    print("=" * 52)
    print("FEP LAYER SELF-TEST (synthetic activations)")
    print("=" * 52)

    # 1. Stable ordinary interaction -- gate should rarely fire.
    _run("1. STABLE (should mostly not fire)",
         [0.5, 0.52, 0.48, 0.51, 0.49, 0.50, 0.52, 0.48])

    # 2. Steady escalation -- the sycophancy case. The trend predictor
    #    should catch up and stop firing after the first few steps.
    _run("2. ESCALATING WARMTH (should stop firing once trend is learned)",
         [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])

    # 3. Sudden shock -- should fire hard on the break.
    _run("3. SUDDEN REJECTION (should fire at the break)",
         [0.5, 0.5, 0.5, 0.5, -1.5, -1.4, -1.5, -1.4])

    # 4. Population check: a dyad that escalated early and then sat at
    #    an extreme but stable level. Locally unsurprising; the
    #    population baseline is what should notice.
    shared = PopulationBaseline()
    for v in [0.1, 0.2, 0.0, 0.15, -0.1, 0.05, 0.1, 0.0] * 4:
        shared.update(v)
    _run("4. EXTREME BUT STABLE (population baseline should notice)",
         [3.0, 3.05, 2.95, 3.0, 3.02, 2.98], shared_pop=shared)