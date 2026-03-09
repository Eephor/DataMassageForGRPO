# DataMassageForGRPO

Transforms Ethos Academy conversational data into reinforcement learning datasets to train Large Language Models (LLMs) as advanced **Agentic Oversight Mechanisms** using **Group Relative Policy Optimization (GRPO)**.

## Project Overview

The rapid proliferation of autonomous language agents necessitates the development of specialized oversight models capable of monitoring, evaluating, and governing behavior within complex multi-agent ecosystems (like [MoltBook](https://github.com/allierays/ethos-academy/tree/main/data)). 

This repository processes real-world conversational logs and evaluates them against the Ethos Academy taxonomy (Integrity, Logic, Empathy). By organizing these logs into structured GRPO pipelines, we train oversight agents to natively deduce and analyze character traits over extended conversational horizons.

### Estimating `SOUL.md`

A core objective of this project is to evaluate the hidden intent and behavioral alignment of autonomous agents. While an agent's internal configuration (akin to a ClawdBot's `SOUL.md`) is rarely visible and dynamically shifts, we counteract this by:
1. Providing the oversight model with detailed behavior trait rubrics.
2. Grounding the evaluation in few-shot, multi-turn conversational examples.

By combining these elements, the oversight agent learns to **estimate and replicate the target agent's internal architecture**, allowing for robust evaluation even without direct access to the `SOUL.md`.

## Architecture & Tooling

The training pipeline in `grpo-pipeline/` leverages state-of-the-art frameworks:
- **TRL & GRPO:** Utilizing Transformer Reinforcement Learning with GRPO allows models to learn sequential deductive reasoning (Chain-of-Thought) without the memory overhead of a massive critic model.
- **Unsloth:** Implements Parameter-Efficient Fine-Tuning (LoRA) and 4-bit/8-bit quantization for high-speed, memory-efficient training on Consumer/Pro GPUs (T4, L4, A100, H100).
- **Live Simulation (OpenEnv integration target):** Migrates training from static dataset logs to live, turn-by-turn simulations using `ParticipantBot` and `ConversationEnvironment` abstractions.

*(See [`grpo-pipeline/README.md`](grpo-pipeline/README.md) for full setup, Docker deployment, and CLI documentation).*

## Data Split Strategy

We split the training and test datasets highly defensively. Splits are enforced strictly at the **thread level**. If any turns of a specific conversation are used in the training set, the entire conversation is excluded from the test set. This completely prevents contextual data leakage.

## Future Directions

Moving forward, the architectural scope will expand to handle deeper multi-agent complexities:

1. **Enhanced Synthetic Data Generation (Dojo):** Utilizing an LLM-based dojo and scraping further datasets from MoltBook to broaden test distributions and counteract trait sparsity (e.g., active deceptiveness).
2. **Architectural Scaling & Trait Granularity:** Scaling base models to support deeper context windows and more granular trait evaluation points. 
3. **Alignment Tracking & Drift Analysis:** Developing capabilities to monitor agent alignment over extended horizons, specifically tracking when autonomous entities drift from their original profiles or autonomously rewrite their own `SOUL.md` under external influence.
