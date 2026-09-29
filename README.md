# Marvarium

The premise of our experiment is a simulation between a caregiver and a child who interact over time. After each interaction, the child reflects on the experience and revises their behavior for the following interaction. Heavily inspired by Marvin Minsky's The Emotion Machine. 

## Methods 

To set up the environment, we use two fine-tuned agents, both Llama-3.1-8B-Instruct at 4-bit precision with separate LoRA adapters cold-started on hand-written exchanges to simulate the parent-child dynamic. The learner's adapter uses rank 32 across 16 layers, 6.82M trainable parameters.

An event generator produces ten situation types — injury, fear, attention bid, social distress, denial, separation, reunion, achievement, conflict, mundane — with five surface variants each. In each of the situations, the child makes bids for attention and protests when they are refused. The sequence is generated once per seed and both arms run on it, so situation base rates are identical by construction.

Protest is measured as a linear projection onto contrastive concept vectors at the last-token position of the residual stream. We tracked the activations of four concepts — comforted, distressed, protesting, withdrawn — with vectors built from six story pairs each, mean-centred against one another, with a neutral baseline subtracted. This technique was adopted from Anthropic’s exploration of functional emotion vectors (Sofroniew et. al, 2026).

The learning loop is structured based on a conversation exchange between the mother and child agents. During each exchange, the child responds and its response is scored by the concept vectors. If the affective change produces “surprise”, as defined by the Free Energy Principle, the learner reflects on what happened, produces a revision informed by that reflection, and its weight updates under a KL constraint. 

Every learning run is paired with a frozen twin experiencing the identical partner, events, gating, reflections and revisions — everything except the gradient step. This isolates weight-updating from the content of the interaction.

## Emergence of non-epistemic sycophancy-like behavior

Protest decay behaviorally involves an agent becoming more deferential, which AI safety research calls sycophancy. Traditionally, this deference is attributed to the training signal: human raters prefer agreeable responses, reward models learn the preference, and policies optimized against them defer. However, in our experiment, we explore how deference-like behavior arises without a preference signal, implying the standard account for sycophancy is incomplete and the standard mitigations do not reach it.

## Repository layout

```
configs/           mlx_lm LoRA training configs (*.yaml)
data/              cold-start SFT data (child/, mother/, child_alt/) and emotion_vectors.json
adapters/          trained LoRA adapters (mother_*, child_*, ceil_*, fixed_*, floor_matched)
runs/              experiment outputs, one folder per experiment (learning/, dose/, hysteresis/, ...)
                   runs/pilot/ holds the first single-seed run; runs/multi_seed/ is untracked scratch
docs/              figures
src/core/          the pipeline: event_generator -> run_episodes / learning_loop -> extract_emotions,
                   plus fep_layer (valence/mood/gate), drift_analysis and build_coldstart_data
src/experiments/   drivers that launch the pipeline across seeds, doses, reliabilities, phases
src/analysis/      post-hoc analyses over the affect files in runs/
src/probes/        model-in-the-loop checks (load an adapter and generate)
src/theory/        toy models with no data dependency
```

Every Python script is run **from the repository root**, e.g.

```
python src/experiments/multi_seed_learning.py --seeds 600 601 602
python src/analysis/decoupling.py --learning "runs/learning/*_on.jsonl" --frozen "runs/learning/*_off.jsonl"
mlx_lm.lora -c configs/child_r32.yaml
```

Paths to data, adapters and runs are relative to the root. Scripts outside `src/core/`
carry a small path shim at the top so they can import the shared modules in `src/core/`
without any install step.

### References:

- Minsky, Marvin. The Emotion Machine: Commonsense Thinking, Artificial Intelligence, and the Future of the Human Mind. (2006)
- Friston, Karl. The free-energy principle: a unified brain theory? (2010)
- Joffily, Matteus & Coricelli, Giorgio. Emotional Valence and the Free-Energy Principle (2013)
- Park, Joon Sung, et al. Generative Agents: Interactive Simulacra of Human Behavior (2023)
- Park, Joon Sung, et al. LLM Agents Grounded in Self-Reports Enable General-Purpose Simulation of Individuals (2024)
- Jaques, Natasha. Social Influence as Intrinsic Motivation for Multi-Agent Deep Reinforcement Learning. (2018)
- Lindsey, Jack. Emergent Introspective Awareness in Large Language Models. (2026)
