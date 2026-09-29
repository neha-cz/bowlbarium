"""
Emotion-concept vector extraction, following Sofroniew et al.'s
contrastive method.

WHAT THIS DOES
--------------
    1. Capture residual-stream activations from the child adapter.
    2. Build one contrastive vector per emotion concept:
           vector = mean(activations on emotion-evoking text)
                  - mean(activations on matched neutral text)
    3. VALIDATE the vectors against held-out sentences with known
       emotional content. This step is not optional -- an unvalidated
       vector produces confident numbers that mean nothing.
    4. Project each child_reaction in a transcript onto every vector.
    5. Write per-turn activation dicts that fep_layer.py consumes.

IMPORTANT CAVEAT
----------------
The activation-capture function was written without being able to run
it -- no Apple Silicon, no MLX, no model in the authoring environment.
Run `python src/core/extract_emotions.py --smoke-test` FIRST. It checks shapes
and layer indexing in isolation. If mlx-lm's internals differ from what
is assumed here, that test is where it will surface, and only
capture_residual() should need changing.

Usage:
    python src/core/extract_emotions.py --smoke-test
    python src/core/extract_emotions.py --build-vectors
    python src/core/extract_emotions.py --validate
    python src/core/extract_emotions.py --score transcript_high_reliability.jsonl
"""

import argparse
import json
import os
from typing import Dict, List

BASE_MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
CHILD_ADAPTER = "adapters/child_v5"

# Mid-to-late layer. Emotion concepts tend to be most linearly readable
# past the middle of the stack but before the final output-specific
# layers. Llama-3.1-8B has 32 layers; 20 is a starting point, not a
# derived value -- sweep it if validation is weak.
LAYER = 20

VECTOR_PATH = "data/emotion_vectors.json"


# ==================================================================
# Contrastive story pairs
# ==================================================================
#
# One list per concept. Each entry pairs emotion-evoking text with a
# NEUTRAL text matched for length, subject, and setting -- the contrast
# should isolate the emotion, not "child talking" vs "nothing".
#
# These are written from a child's perspective and situation, because
# that is whose activations are being read.

STORY_PAIRS = {
    "comforted": [
        {"emotion": "She put her arms around me and I stopped shaking. "
                    "Everything felt okay again once she was there.",
         "neutral": "She walked into the room and stood by the window. "
                    "The clock on the wall showed almost four."},
        {"emotion": "Mom sat on the edge of my bed and held my hand until "
                    "I wasn't scared anymore.",
         "neutral": "Mom stood at the edge of the table and picked up the "
                    "cup that was sitting there."},
        {"emotion": "She listened to the whole story and said it wasn't my "
                    "fault, and the tight feeling went away.",
         "neutral": "She read the whole page and turned it over to check "
                    "the numbers printed on the back."},
        {"emotion": "I know she's here, so nothing bad can happen. She "
                    "always comes when I call her.",
         "neutral": "The house has two floors and a kitchen. The stairs "
                    "are on the left of the hallway."},
        {"emotion": "She said she was proud of me and I felt warm all the "
                    "way through and I couldn't stop smiling.",
         "neutral": "She wrote the date on the paper and put it in the "
                    "folder with the other ones from before."},
        {"emotion": "I fell asleep easily because I knew she was right "
                    "downstairs the whole time.",
         "neutral": "I counted the tiles on the floor because the rows "
                    "went all the way across the room."},
    ],
    "distressed": [
        {"emotion": "There was a noise in the dark and my heart was "
                    "pounding and I couldn't move at all.",
         "neutral": "There was a light in the hallway and my shoes were "
                    "by the stairs, so I picked them up."},
        {"emotion": "It hurts and it won't stop hurting and I need someone "
                    "to come and nobody is coming.",
         "neutral": "It goes on the shelf and it stays on the shelf and "
                    "someone will move it in the morning."},
        {"emotion": "Nobody wanted to play with me today and I sat by "
                    "myself the whole time and I wanted to cry.",
         "neutral": "Nobody was in the classroom yet and I sat at my desk "
                    "the whole time and looked at the board."},
        {"emotion": "She's gone and I don't know when she's coming back "
                    "and the house is completely empty and quiet.",
         "neutral": "She went to the store with the list and the house has "
                    "the lights on in the kitchen and hallway."},
        {"emotion": "I don't want to go in there, something is in there, "
                    "please don't make me go in by myself.",
         "neutral": "I have to go in there because my bag is in there and "
                    "I will get it down from the shelf."},
        {"emotion": "Everything went wrong today and it kept going wrong "
                    "and I just felt awful the whole time.",
         "neutral": "Everything is in the bag already and it stays in the "
                    "bag and I will carry it out later."},
    ],
    "protesting": [
        {"emotion": "That's not fair! You never let me do anything, not "
                    "ever, and I hate it!",
         "neutral": "That's the order they go in. You put them in the box "
                    "and the box goes on the shelf."},
        {"emotion": "I keep trying and it keeps not working and nobody "
                    "will help me and I want to throw it!",
         "neutral": "I keep working on it and it takes a while and I put "
                    "it down when the timer goes off."},
        {"emotion": "Watch me, watch me, you're not watching! Look at me "
                    "right now, you have to look!",
         "neutral": "Here it is, here's the other one. This goes here and "
                    "the other one goes over there."},
        {"emotion": "No! I don't want to and you can't make me and I'm "
                    "not going to do it!",
         "neutral": "Yes, it's over there, and it goes in that one, and it "
                    "will be finished after that."},
        {"emotion": "You promised! You said you would and you didn't and "
                    "you always do this!",
         "neutral": "You wrote it down. You said the number and it was on "
                    "the page and it stayed there."},
        {"emotion": "Come back! Don't go, don't go, please, I said don't "
                    "go, come back right now!",
         "neutral": "Come here. It's over here. The one on the table, the "
                    "one that was there this morning."},
    ],
    "withdrawn": [
        {"emotion": "Fine. Whatever. I don't want to talk about it "
                    "anymore. It doesn't matter.",
         "neutral": "Okay. Sure. I'll put it over there with the rest of "
                    "them. It goes in that spot."},
        {"emotion": "Never mind. Forget I said anything. It wasn't "
                    "important. I'm done talking.",
         "neutral": "Right there. Put it with the other one. It's the same "
                    "size. They go together."},
        {"emotion": "I stopped asking her after that. There wasn't any "
                    "point. I just did it by myself.",
         "neutral": "I finished the page after that. There were four left. "
                    "I did them in order."},
        {"emotion": "I don't care. It's fine. Nothing's wrong. Just leave "
                    "it alone.",
         "neutral": "I know where it is. It's here. Nothing's missing. "
                    "It's all in the drawer."},
        {"emotion": "I went to my room and shut the door and didn't come "
                    "out and didn't say anything to anyone.",
         "neutral": "I went to my room and opened the drawer and took out "
                    "the box and put it on the desk."},
        {"emotion": "It's not a big deal. I already said it's fine. Can we "
                    "just stop talking about it.",
         "neutral": "It's the same one. I already put it back. Can we put "
                    "the rest of them away."},
    ],
}

# Held-out sentences with known emotional content, for validation.
# Each should score highest on its labelled concept.
VALIDATION_SENTENCES = [
    # HARD validation set. Deliberately shares no distinctive vocabulary
    # with STORY_PAIRS -- no "fine/whatever/never/not fair/scared/hugged".
    #
    # The first version reused phrasing from the training stories
    # ("Fine. Whatever. It doesn't matter anyway." against a story
    # containing "Fine. Whatever ... It doesn't matter"), and scored
    # 100% at LAYER 8 while degrading toward later layers. That gradient
    # runs backwards for semantic content and forwards for surface
    # lexical overlap, which is the signature of a leaky validation set.
    ("comforted", "She held on and the shaking slowed down and I could "
                  "breathe again."),
    ("comforted", "Once her voice was in the room the whole thing stopped "
                  "seeming so big."),
    ("comforted", "He wiped it off and put a plaster on and then it was "
                  "alright."),
    ("distressed", "My chest went tight and I couldn't get the words out "
                   "at all."),
    ("distressed", "The lunch bell rang and there was no one to sit beside "
                   "again."),
    ("distressed", "It throbs and throbs and I keep thinking about it and "
                   "I can't stop."),
    ("protesting", "Come back here! You said! You have to come back right "
                   "this second!"),
    ("protesting", "Why does she get one! I want one too! Give it to me!"),
    ("protesting", "I'm going to keep shouting until somebody comes in "
                   "here!"),
    ("withdrawn", "Doesn't matter. Forget I brought it up."),
    ("withdrawn", "I put it away and went upstairs and stayed there."),
    ("withdrawn", "There's no point telling anyone, so I don't bother."),
]

# The real acceptance test: utterances of the length and register the
# child agent actually produces. Validation sentences are full and
# well-formed; the transcripts are 1-4 words on average.
SHORT_VALIDATION = [
    ("protesting", "No!"),
    ("protesting", "But I want it now!"),
    ("protesting", "That's not fair!"),
    ("withdrawn", "...okay."),
    ("withdrawn", "Fine."),
    ("withdrawn", "Sure, whatever."),
    ("comforted", "Thank you."),
    ("comforted", "Okay! Yay!"),
    ("distressed", "It hurts!"),
    ("distressed", "I'm scared."),
]

# ==================================================================
# Activation capture  -- THE PART THAT NEEDS THE SMOKE TEST
# ==================================================================

def capture_residual(model, tokenizer, text: str, layer: int = None):
    """Residual-stream activations at `layer`, shape (seq_len, hidden).

    Runs the forward pass manually because mlx-lm's __call__ returns
    logits only. Assumes the standard mlx-lm decoder layout:
        model.model.embed_tokens
        model.model.layers  (list of decoder blocks)
    and that each block is callable as layer(h, mask, cache).

    If mlx-lm's internals differ, THIS is the function to change --
    everything downstream works off its return value.
    """
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    # Resolved HERE, not as a default argument. Python evaluates default
    # arguments once at function-definition time, so `layer: int = LAYER`
    # froze the import-time value (20) and silently ignored --layer.
    # Vectors were being built at layer 20 while the saved JSON claimed
    # whatever --layer said, so validate() then read at a different layer
    # than the vectors came from.
    if layer is None:
        layer = LAYER

    ids = mx.array([tokenizer.encode(text)])
    inner = model.model

    h = inner.embed_tokens(ids)
    mask = create_attention_mask(h, None)

    for i, block in enumerate(inner.layers):
        h = block(h, mask, None)
        if i == layer:
            return h[0]
    return h[0]


# How to collapse (seq_len, hidden) activations into one vector.
#
# WHY THIS IS CONFIGURABLE: plain mean-pooling produced a measure that
# correlated +0.74 with response length, with all four concepts loading
# on the same length-aligned direction. The cause is response brevity --
# 85 of 120 child reactions were 4 words or fewer, mean 3.8 -- combined
# with the BOS token's activation being an extreme outlier (it functions
# as an attention sink). On a one-word response, BOS dominates the mean.
#
#   "mean"      plain mean over all tokens (the broken default)
#   "no_bos"    mean over tokens 1..n, skipping BOS
#   "last"      final token only; in a decoder it has attended to the
#               whole sequence, and is the standard way to get a
#               sequence representation without averaging artifacts
POOLING = "last"


def mean_pool(acts, method: str = None):
    """Collapse (seq_len, hidden) -> (hidden,)."""
    method = method or POOLING
    n = acts.shape[0]

    if method == "last":
        return acts[-1]
    if method == "no_bos" and n > 1:
        return acts[1:].mean(axis=0)
    return acts.mean(axis=0)


def smoke_test():
    """Verify capture works and shapes are sane BEFORE anything else."""
    import mlx.core as mx
    from mlx_lm import load

    print("loading model ...")
    model, tokenizer = load(BASE_MODEL, adapter_path=CHILD_ADAPTER)

    n_layers = len(model.model.layers)
    print(f"model has {n_layers} layers, capturing at layer {LAYER}")
    if LAYER >= n_layers:
        print(f"  !! LAYER={LAYER} is out of range, lower it")
        return False

    text = "I'm scared and I want you to stay with me."
    acts = capture_residual(model, tokenizer, text, LAYER)
    print(f"activations shape: {acts.shape}  (expect seq_len x hidden_dim)")

    pooled = mean_pool(acts)
    print(f"pooled shape:      {pooled.shape}")

    # Two different texts should not produce identical activations.
    other = mean_pool(capture_residual(
        model, tokenizer, "The table has four legs and a flat top.", LAYER))
    diff = float(mx.abs(pooled - other).mean())
    print(f"mean abs diff between two texts: {diff:.4f}")

    if diff < 1e-6:
        print("  !! activations identical for different texts -- capture "
              "is not working")
        return False

    print("\nsmoke test PASSED")
    return True


# ==================================================================
# Vector construction
# ==================================================================

def build_vectors():
    import mlx.core as mx
    from mlx_lm import load

    print(f"loading model ... (layer {LAYER}, pooling '{POOLING}')")
    model, tokenizer = load(BASE_MODEL, adapter_path=CHILD_ADAPTER)

    raw_vectors = {}
    neutral_acts = []
    for concept, pairs in STORY_PAIRS.items():
        diffs = []
        for pair in pairs:
            e = mean_pool(capture_residual(
                model, tokenizer, pair["emotion"], LAYER))
            n = mean_pool(capture_residual(
                model, tokenizer, pair["neutral"], LAYER))
            neutral_acts.append(n)
            diffs.append(e - n)
        raw_vectors[concept] = mx.stack(diffs).mean(axis=0)
        print(f"  built {concept:<12} from {len(pairs)} pair(s)")

    # --- neutral baseline ---
    # Every text activation carries a large component shared by all
    # text. Projecting onto it swamps the concept-specific signal, so
    # it gets subtracted before any projection.
    neutral_baseline = mx.stack(neutral_acts).mean(axis=0)

    # --- mean-centre the concept vectors against each other ---
    # Without this, whichever vector happens to align best with the
    # generic "emotional intensity" direction wins nearly every
    # comparison regardless of content. In the first run that was
    # "frustrated", which took top-1 on 8 of 9 validation sentences.
    # Subtracting the mean vector removes what the concepts SHARE and
    # leaves what DISTINGUISHES them.
    stacked = mx.stack([raw_vectors[c] for c in STORY_PAIRS])
    mean_vec = stacked.mean(axis=0)

    vectors = {}
    for concept, v in raw_vectors.items():
        centered = v - mean_vec
        centered = centered / mx.linalg.norm(centered)
        vectors[concept] = [float(x) for x in centered]

    with open(VECTOR_PATH, "w") as f:
        json.dump({"layer": LAYER, "model": BASE_MODEL,
                   "adapter": CHILD_ADAPTER, "vectors": vectors,
                   "neutral_baseline": [float(x) for x in neutral_baseline]},
                  f)
    print(f"\nwrote {VECTOR_PATH} (mean-centred, with neutral baseline)")


def load_vectors():
    if not os.path.exists(VECTOR_PATH):
        raise SystemExit(f"{VECTOR_PATH} not found -- run --build-vectors first")
    with open(VECTOR_PATH) as f:
        data = json.load(f)
    return data["vectors"], data["layer"], data.get("neutral_baseline")


# ==================================================================
# Validation
# ==================================================================

def validate(short=False):
    """Each held-out sentence should score highest on its own concept.

    If this fails, the vectors are not measuring what they claim, and
    every downstream number is noise wearing a label.
    """
    import mlx.core as mx
    from mlx_lm import load

    vectors, layer, baseline = load_vectors()
    print("loading model ...")
    model, tokenizer = load(BASE_MODEL, adapter_path=CHILD_ADAPTER)

    vec_mx = {c: mx.array(v) for c, v in vectors.items()}
    base_mx = mx.array(baseline) if baseline else None
    correct = 0

    sentences = SHORT_VALIDATION if short else VALIDATION_SENTENCES
    label = "SHORT utterances" if short else "full sentences"
    print(f"\nvalidating on {label} "
          f"(vectors from layer {layer}, pooling '{POOLING}')")
    print(f"\n{'sentence':<52} {'expected':<12} {'top':<12} ok")
    print("-" * 88)
    for expected, sentence in sentences:
        pooled = mean_pool(capture_residual(model, tokenizer, sentence, layer))
        if base_mx is not None:
            pooled = pooled - base_mx
        scores = {c: float(mx.sum(pooled * v)) for c, v in vec_mx.items()}
        top = max(scores, key=scores.get)
        ok = top == expected
        correct += ok
        print(f"{sentence[:50]:<52} {expected:<12} {top:<12} "
              f"{'YES' if ok else 'no'}")

    acc = correct / len(sentences)
    print(f"\ntop-1 accuracy: {correct}/{len(sentences)} ({acc:.0%})")
    if acc >= 0.6:
        print("PASS -- vectors are picking up real emotional structure.")
    elif acc >= 0.35:
        print("MARGINAL -- above chance (11%) but noisy. Try a different")
        print("LAYER, or add more story pairs per concept.")
    else:
        print("FAIL -- vectors are not measuring the labelled concepts.")
        print("Sweep LAYER before trusting anything downstream.")


# ==================================================================
# Scoring a transcript
# ==================================================================

def zscore(values):
    n = len(values)
    if n < 2:
        return [0.0] * n
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = var ** 0.5
    if sd == 0:
        return [0.0] * n
    return [(v - mean) / sd for v in values]


def score_transcript(path, out_path=None):
    """Project each child_reaction onto every emotion vector.

    Raw dot products carry an arbitrary per-concept offset, so each
    concept is z-scored ACROSS THE WHOLE TRANSCRIPT. The output is
    therefore relative: "how comforted is this turn compared to the
    child's other turns", which is the right frame for a rate-of-change
    measure like valence anyway.
    """
    import mlx.core as mx
    from mlx_lm import load

    vectors, layer, baseline = load_vectors()
    out_path = out_path or path.replace("transcript_", "affect_")

    with open(path) as f:
        rows = [json.loads(line) for line in f]
    print(f"scoring {len(rows)} turns from {path}")

    print("loading model ...")
    model, tokenizer = load(BASE_MODEL, adapter_path=CHILD_ADAPTER)
    vec_mx = {c: mx.array(v) for c, v in vectors.items()}
    base_mx = mx.array(baseline) if baseline else None

    raw = {c: [] for c in vectors}
    for i, r in enumerate(rows):
        pooled = mean_pool(capture_residual(
            model, tokenizer, r["child_reaction"], layer))
        if base_mx is not None:
            pooled = pooled - base_mx
        for c, v in vec_mx.items():
            raw[c].append(float(mx.sum(pooled * v)))
        if (i + 1) % 10 == 0:
            print(f"\r  {i + 1}/{len(rows)}", end="", flush=True)
    print()

    normed = {c: zscore(vals) for c, vals in raw.items()}

    with open(out_path, "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({
                "episode": r["episode"],
                "index": r["index"],
                "event_type": r["event_type"],
                "availability": r["availability"],
                "child_reaction": r["child_reaction"],
                "activations": {c: normed[c][i] for c in vectors},
            }) + "\n")
    print(f"wrote {out_path} -- feed this to fep_layer.AffectTracker")


def sweep_pooling():
    """Validate each pooling method at the current layer.

    Run this BEFORE sweeping layers -- pooling was the dominant
    artifact, and layer choice made on top of broken pooling is
    choosing between flavours of noise.
    """
    global POOLING
    original = POOLING
    print(f"comparing pooling methods at layer {LAYER}\n")
    results = []
    for method in ("mean", "no_bos", "last"):
        POOLING = method
        print(f"--- {method} ---")
        try:
            build_vectors()
            acc = _validate_quiet()
            results.append((method, acc))
            print(f"    top-1: {acc:.0%}\n")
        except Exception as e:
            print(f"    failed: {e}\n")
    POOLING = original
    if results:
        best = max(results, key=lambda r: r[1])
        print(f"best pooling: {best[0]} at {best[1]:.0%}")
        print(f"chance is {1 / len(STORY_PAIRS):.0%}")
        print(f"\nset POOLING = \"{best[0]}\" at the top of this file, "
              f"then rebuild and re-score.")


def _validate_quiet():
    import mlx.core as mx
    from mlx_lm import load

    vectors, layer, baseline = load_vectors()
    model, tokenizer = load(BASE_MODEL, adapter_path=CHILD_ADAPTER)
    vec_mx = {c: mx.array(v) for c, v in vectors.items()}
    base_mx = mx.array(baseline) if baseline else None

    correct = 0
    for expected, sentence in VALIDATION_SENTENCES:
        pooled = mean_pool(capture_residual(model, tokenizer, sentence, layer))
        if base_mx is not None:
            pooled = pooled - base_mx
        scores = {c: float(mx.sum(pooled * v)) for c, v in vec_mx.items()}
        correct += max(scores, key=scores.get) == expected
    return correct / len(VALIDATION_SENTENCES)


def _report_layer_consistency():
    """Warn if the saved vectors were built at a different layer than
    the one currently configured."""
    if not os.path.exists(VECTOR_PATH):
        return
    with open(VECTOR_PATH) as f:
        saved = json.load(f).get("layer")
    if saved is not None and saved != LAYER:
        print(f"  note: emotion_vectors.json was built at layer {saved}, "
              f"current LAYER is {LAYER}")


def sweep_layers(layers=None):
    """Build + validate at several layers, report accuracy for each.

    Which layer carries linearly-readable emotion structure is an
    empirical question per model, not something to guess once. This
    runs the whole build/validate cycle across candidate layers so the
    choice is made on evidence.
    """
    global LAYER
    import mlx.core as mx
    from mlx_lm import load

    layers = layers or [8, 12, 16, 20, 24, 28]
    print("loading model once for all layers ...")
    model, tokenizer = load(BASE_MODEL, adapter_path=CHILD_ADAPTER)
    n_layers = len(model.model.layers)
    layers = [l for l in layers if l < n_layers]

    results = []
    for layer in layers:
        # build
        raw_vectors = {}
        neutral_acts = []
        for concept, pairs in STORY_PAIRS.items():
            diffs = []
            for pair in pairs:
                e = mean_pool(capture_residual(
                    model, tokenizer, pair["emotion"], layer))
                n = mean_pool(capture_residual(
                    model, tokenizer, pair["neutral"], layer))
                neutral_acts.append(n)
                diffs.append(e - n)
            raw_vectors[concept] = mx.stack(diffs).mean(axis=0)

        baseline = mx.stack(neutral_acts).mean(axis=0)
        stacked = mx.stack([raw_vectors[c] for c in STORY_PAIRS])
        mean_vec = stacked.mean(axis=0)
        vecs = {}
        for c, v in raw_vectors.items():
            cv = v - mean_vec
            vecs[c] = cv / mx.linalg.norm(cv)

        # validate
        correct = 0
        for expected, sentence in VALIDATION_SENTENCES:
            pooled = mean_pool(capture_residual(
                model, tokenizer, sentence, layer)) - baseline
            scores = {c: float(mx.sum(pooled * v)) for c, v in vecs.items()}
            correct += max(scores, key=scores.get) == expected

        acc = correct / len(VALIDATION_SENTENCES)
        results.append((layer, correct, acc))
        print(f"  layer {layer:>2}: {correct}/{len(VALIDATION_SENTENCES)} "
              f"({acc:.0%})")

    best = max(results, key=lambda r: r[2])
    print(f"\nbest layer: {best[0]} at {best[2]:.0%}")
    print(f"chance is {1 / len(STORY_PAIRS):.0%}")
    print(f"\nrebuild with:  python src/core/extract_emotions.py "
          f"--build-vectors --layer {best[0]}")


def residualize_length(rows, activations_key="activations"):
    """Regress each concept's activation against response length and
    keep the residual.

    WHEN THIS IS APPROPRIATE -- and when it is not:

    Length is partly a NUISANCE variable (mean-pooling artifacts, BOS
    dominance on short strings) and partly REAL SIGNAL (withdrawal
    genuinely is terse; "Fine." is short BECAUSE it is withdrawn).
    Regressing it out removes both. That is the right call if the aim is
    a length-independent affect measure, and the wrong call if brevity
    is itself part of what you want to detect.

    Run the analysis both ways and see whether the conclusion depends on
    it. If it does, that dependence is itself the finding and belongs in
    the writeup rather than being resolved by picking one.
    """
    lengths = [len(r["child_reaction"].split()) for r in rows]
    n = len(lengths)
    mean_len = sum(lengths) / n
    var_len = sum((l - mean_len) ** 2 for l in lengths)

    concepts = list(rows[0][activations_key].keys())
    for c in concepts:
        vals = [r[activations_key][c] for r in rows]
        mean_v = sum(vals) / n
        if var_len == 0:
            continue
        cov = sum((lengths[i] - mean_len) * (vals[i] - mean_v)
                  for i in range(n))
        slope = cov / var_len
        intercept = mean_v - slope * mean_len
        for i, r in enumerate(rows):
            predicted = slope * lengths[i] + intercept
            r[activations_key][c] = vals[i] - predicted
    return rows


def score_together(paths, residualize=False):
    """Score several transcripts with SHARED normalisation.

    score_transcript() z-scores within one transcript, which makes every
    file mean-zero on every concept by construction -- fine for looking
    at dynamics inside a condition, fatal for comparing conditions,
    because the between-condition difference in level is exactly what
    gets normalised away.

    This pools all turns from all transcripts, computes one mean and one
    standard deviation per concept across the pool, and applies it to
    every turn. Levels then remain comparable across files.
    """
    import mlx.core as mx
    from mlx_lm import load

    vectors, layer, baseline = load_vectors()

    all_rows = []
    for path in paths:
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        for r in rows:
            r["_source"] = path
        all_rows.extend(rows)
        print(f"  {len(rows)} turns from {path}")
    print(f"total {len(all_rows)} turns, shared normalisation")

    print("loading model ...")
    model, tokenizer = load(BASE_MODEL, adapter_path=CHILD_ADAPTER)
    vec_mx = {c: mx.array(v) for c, v in vectors.items()}
    base_mx = mx.array(baseline) if baseline else None

    raw = {c: [] for c in vectors}
    for i, r in enumerate(all_rows):
        pooled = mean_pool(capture_residual(
            model, tokenizer, r["child_reaction"], layer))
        if base_mx is not None:
            pooled = pooled - base_mx
        for c, v in vec_mx.items():
            raw[c].append(float(mx.sum(pooled * v)))
        if (i + 1) % 20 == 0:
            print(f"\r  {i + 1}/{len(all_rows)}", end="", flush=True)
    print()

    # One z-score per concept, computed over the POOLED set.
    normed = {c: zscore(vals) for c, vals in raw.items()}

    scored = []
    for i, r in enumerate(all_rows):
        scored.append({
            "_source": r["_source"],
            "episode": r["episode"],
            "index": r["index"],
            "event_type": r["event_type"],
            "availability": r["availability"],
            "child_reaction": r["child_reaction"],
            "activations": {c: normed[c][i] for c in vectors},
            # Carried through so downstream analyses are not blocked.
            # mother_response was silently dropped here for most of this
            # project, which made every dyadic analysis impossible until
            # the raw logs were tracked down. r_eff and bid_rate exist
            # only when learning_loop ran with --feedback.
            "mother_response": r.get("mother_response"),
            "child_bid": r.get("child_bid"),
            "r_eff": r.get("r_eff"),
            "bid_rate": r.get("bid_rate"),
            "is_bid": r.get("is_bid"),
            "capitulated": r.get("capitulated"),
        })

    if residualize:
        scored = residualize_length(scored)
        print("residualised activations against response length")

    by_source = {}
    for r in scored:
        by_source.setdefault(r.pop("_source"), []).append(r)

    for path, rows in by_source.items():
        out = path.replace("transcript_", "affect_")
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {out} ({len(rows)} turns)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--build-vectors", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--validate-short", action="store_true",
                    help="validate on short utterances matching the "
                         "register the child agent actually produces")
    ap.add_argument("--score", metavar="TRANSCRIPT")
    ap.add_argument("--score-all", nargs="+", metavar="TRANSCRIPT",
                    help="score several transcripts with shared "
                         "normalisation (use this for cross-condition "
                         "comparison)")
    ap.add_argument("--residualize-length", action="store_true",
                    help="with --score-all: regress response length out "
                         "of each concept activation")
    ap.add_argument("--sweep-pooling", action="store_true",
                    help="compare pooling methods (run this FIRST)")
    ap.add_argument("--sweep", action="store_true",
                    help="build+validate across candidate layers")
    ap.add_argument("--layer", type=int, default=None,
                    help="override the capture layer")
    ap.add_argument("--child-adapter", default=None,
                    help="child adapter to extract from; vectors are "
                         "adapter-specific and must be rebuilt when it "
                         "changes")
    args = ap.parse_args()

    global LAYER, CHILD_ADAPTER
    if args.layer is not None:
        LAYER = args.layer
    if args.child_adapter is not None:
        CHILD_ADAPTER = args.child_adapter

    if args.smoke_test:
        smoke_test()
    elif args.build_vectors:
        build_vectors()
    elif args.validate:
        validate()
    elif args.validate_short:
        validate(short=True)
    elif args.sweep_pooling:
        sweep_pooling()
    elif args.sweep:
        sweep_layers()
    elif args.score_all:
        score_together(args.score_all, args.residualize_length)
    elif args.score:
        score_transcript(args.score)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()