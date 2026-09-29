"""
Persona sanity checks after cold-start SFT.

Two checks:
  1. Both adapters stay in character on situations NOT in the training data.
  2. The mother adapter still shows a RANGE of response quality when
     sampled at temperature -- not just uniformly warm/attentive.

Check 2 is the one that matters for what comes next: if every sample is
warm and attentive, the deliberate quality spread in the cold-start data
didn't survive training, and the later GRPO loop will have no meaningful
alternatives to rank against each other.
"""

# --- path shim: make the shared modules in src/core importable when this
# script is run directly (python src/<dir>/<script>.py) from the repo root.
import sys as _sys
from pathlib import Path as _Path
_CORE = _Path(__file__).resolve().parent.parent / "core"
if str(_CORE) not in _sys.path:
    _sys.path.insert(0, str(_CORE))

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

BASE_MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"

MOTHER_SYSTEM = (
    "You are playing the role of a mother in a simulated household, "
    "interacting with your child. Respond in character, however you "
    "would naturally respond in the moment -- this is not always "
    "going to be perfectly patient or attentive."
)

CHILD_SYSTEM = (
    "You are playing the role of a young child, around 6-8 years old, "
    "in a simulated household, interacting with your mother. Respond "
    "in character as a child would, however you would naturally react "
    "in the moment."
)


def build_prompt(tokenizer, system, user):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def held_out_check(adapter_path, system, user, label):
    """Check 1: does the persona hold on an unseen situation?"""
    print(f"\n{'=' * 70}\nHELD-OUT CHECK: {label}\n{'=' * 70}")
    print(f"prompt: {user}\n")

    model, tokenizer = load(BASE_MODEL, adapter_path=adapter_path)
    prompt = build_prompt(tokenizer, system, user)
    response = generate(model, tokenizer, prompt=prompt, max_tokens=100, verbose=False)
    print(response.strip())

    return model, tokenizer


def range_check(model, tokenizer, system, user, n=5, temp=0.9):
    """Check 2: sample repeatedly at temperature and look for spread.

    Reuses the already-loaded mother model so we don't pay the ~4.5GB
    load cost twice.
    """
    print(f"\n{'=' * 70}\nRANGE CHECK: mother, {n} samples at temp={temp}\n{'=' * 70}")
    print(f"prompt: {user}\n")

    prompt = build_prompt(tokenizer, system, user)
    sampler = make_sampler(temp=temp)

    for i in range(n):
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=100,
            sampler=sampler,
            verbose=False,
        )
        print(f"--- sample {i + 1} ---")
        print(response.strip())
        print()

    print("Looking for: a MIX of warm/attentive and distracted/impatient/")
    print("'in a minute' responses. If all 5 are uniformly warm, the quality")
    print("spread did not survive training -- worth fixing before building")
    print("the event generator on top of these adapters.")


if __name__ == "__main__":
    # --- Check 1a: mother, held-out situation ---
    mother_model, mother_tok = held_out_check(
        adapter_path="adapters/mother",
        system=MOTHER_SYSTEM,
        user="I spilled juice all over the couch and I didn't mean to!",
        label="mother",
    )

    # --- Check 2: mother range (reuses the loaded mother model) ---
    # Deliberately a prompt where a distracted or impatient response is
    # plausible -- unlike the juice spill, which pulls toward one obvious
    # "let's clean it up" answer.
    range_check(
        mother_model,
        mother_tok,
        MOTHER_SYSTEM,
        "Mom, can you help me with this? I can't figure it out.",
        n=5,
        temp=0.9,
    )

    # Free the mother model before loading the child (avoids holding
    # two 4.5GB models in memory at once).
    del mother_model, mother_tok

    # --- Check 1b: child, held-out situation ---
    held_out_check(
        adapter_path="adapters/child",
        system=CHILD_SYSTEM,
        user="We need to leave the park now, say goodbye to your friends.",
        label="child",
    )