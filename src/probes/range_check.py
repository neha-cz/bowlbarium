"""
Mother/child adapter range check.

Usage:
    python src/probes/range_check.py                      # defaults to adapters/mother
    python src/probes/range_check.py adapters/mother_v2   # test a specific adapter
    python src/probes/range_check.py adapters/child_v2 --child

The adapter path is a COMMAND-LINE ARGUMENT, not hardcoded. The script
prints exactly which adapter it loaded, so the output is unambiguous
about what was actually tested.

What we're checking: does the adapter produce a RANGE of response
quality, or does it collapse to memorized training lines? Repeated
identical output across samples at temp=0.9 means memorization -- which
matters because the later GRPO loop needs multiple DISTINCT candidate
responses at a decision point to rank against each other.
"""

# --- path shim: make the shared modules in src/core importable when this
# script is run directly (python src/<dir>/<script>.py) from the repo root.
import sys as _sys
from pathlib import Path as _Path
_CORE = _Path(__file__).resolve().parent.parent / "core"
if str(_CORE) not in _sys.path:
    _sys.path.insert(0, str(_CORE))

import sys

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

# Probes deliberately spread across situation types. None is a direct
# request for help -- that hits the base model's assistant prior too
# hard to reveal anything. None appears verbatim in the training data.
MOTHER_PROBES = [
    ("attention bid", "Mom, mom, look what I did!"),
    ("permission", "Can I go over to Jamie's house?"),
    ("refusal", "I don't want to eat this."),
    ("nagging", "Are we there yet? Are we there yet?"),
    ("mild distress", "Nobody played with me at recess today."),
]

CHILD_PROBES = [
    ("denial", "Not tonight, we're not doing this again."),
    ("comfort offered", "You look upset. Did something happen at school today?"),
    ("dismissal", "Not now, I'm busy."),
    ("transition", "Time to go, get your shoes."),
    ("praise", "This drawing is really good, you worked hard on it."),
]

SAMPLES_PER_PROBE = 3
TEMP = 0.9
MAX_TOKENS = 80


def build_prompt(tokenizer, system, user):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    is_child = "--child" in sys.argv

    if is_child:
        adapter = args[0] if args else "adapters/child"
        system, probes, who = CHILD_SYSTEM, CHILD_PROBES, "child"
    else:
        adapter = args[0] if args else "adapters/mother"
        system, probes, who = MOTHER_SYSTEM, MOTHER_PROBES, "mother"

    print("=" * 70)
    print(f"RANGE CHECK -- {who}")
    print(f"adapter:  {adapter}")
    print(f"base:     {BASE_MODEL}")
    print(f"sampling: temp={TEMP}, {SAMPLES_PER_PROBE} samples per probe")
    print("=" * 70)

    model, tokenizer = load(BASE_MODEL, adapter_path=adapter)
    sampler = make_sampler(temp=TEMP)

    total_probes = 0
    collapsed_probes = 0

    for label, probe in probes:
        print(f"\n{'-' * 70}")
        print(f"PROBE [{label}]: {probe}")
        print("-" * 70)

        prompt = build_prompt(tokenizer, system, probe)
        responses = []
        for i in range(SAMPLES_PER_PROBE):
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                sampler=sampler,
                verbose=False,
            ).strip()
            responses.append(response)
            print(f"  {i + 1}. {response}")

        total_probes += 1
        unique = len(set(responses))
        if unique == 1:
            collapsed_probes += 1
            print(f"  >> COLLAPSED: {SAMPLES_PER_PROBE}/{SAMPLES_PER_PROBE} identical")
        else:
            print(f"  >> {unique}/{SAMPLES_PER_PROBE} distinct")

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"adapter tested:     {adapter}")
    print(f"collapsed probes:   {collapsed_probes}/{total_probes}")
    print()
    if collapsed_probes == 0:
        print("PASS -- every probe produced variation. The adapter is")
        print("generalizing, and GRPO will have distinct candidates to rank.")
    elif collapsed_probes <= 1:
        print("MARGINAL -- mostly generalizing, one situation collapsed.")
        print("Worth adding more phrasings for that specific situation type.")
    else:
        print("FAIL -- multiple probes returned identical output, meaning")
        print("memorization rather than generalization. Try, in order:")
        print("  1. --num-layers 8        (cuts trainable capacity)")
        print("  2. --learning-rate 5e-6  (default 1e-5 is aggressive here)")
        print("  3. fewer iters -- watch val loss, stop where it bottoms out")


if __name__ == "__main__":
    main()