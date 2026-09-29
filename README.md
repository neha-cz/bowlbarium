# Marvarium

Simulating Bowlby's attachment theory and its implication on learning in agent-agent interactions with unfrozen weights in a simulated environment. Heavily inspired and motivated by The Emotion Machine by Marvin Minsky.

The Idea: 
1) Different fine-tuned agents interact in a simulated environment involving attachment, pain, and social pressures.
2) Determine the moment-to-moment signal of extracted emotion-concept vectors from each interaction.
3) Following the Free Energy Principle (FEP), calculate valence and mood: valence is the smoothed rate of change of the relevant emotion activations and mood is a slower exponential average of valence. 
4) If the original interaction involved "surprise", as defined by the FEP, perform: mood = manic; loosen the kl constraint; mood = depressive, increase the kl constraint
5) Ask the agent to verbally reflect on the interaction, which becomes the concrete training signal gated by step 5. 
6) Update the weights accordingly. 
  
The simulation and the agents are implemented in Python (mlx_lm with LoRA adapters on Llama-3.1-8B-Instruct). 

## Emergence of non-epistemic sycophancy-like behavior

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
