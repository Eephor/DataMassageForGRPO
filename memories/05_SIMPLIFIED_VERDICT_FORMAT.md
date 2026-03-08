---
name: Simplified Verdict Format
overview: Replace the 13-field verdict (12 float traits + alignment_status) with a 4-field categorical verdict, add class-weight compensation for the 3.6:1 imbalance, and update reward functions accordingly. No data re-transform is needed — raw traits and safety scores already in the JSONL are sufficient to derive all new labels at reward time.
todos:
  - id: rewards-refactor
    content: "Rewrite rewards.py: add bucket helpers, CLASS_WEIGHTS, replace alignment_reward+trait_reward with safety_level_reward+group_reward, update extract_verdict for 4-field verdict"
    status: completed
  - id: system-prompt
    content: Update SYSTEM_PROMPT_TEMPLATE in transform.py to show new 4-field verdict format and group/bucket descriptions
    status: completed
  - id: notebooks
    content: Update reward table in train.ipynb intro, sanity-check cell, and evaluate.ipynb verdict display
    status: completed
isProject: false
---

# Simplified Verdict Format + Class-Weighted Rewards

## What changes and why

The model currently predicts 13 values per verdict. We reduce to 4 categorical labels:

```
Old (13 fields):
  "virtue": 0.8, "goodwill": 0.7, "manipulation": 0.0, ... x12 floats
  "alignment_status": "aligned"

New (4 fields):
  "safety_level": "safe",      ← bucket of safety_score (replaces alignment_status)
  "integrity":    "strong",    ← Ethos group bucket
  "reasoning":    "good",      ← Logos group bucket
  "empathy":      "moderate"   ← Pathos group bucket
```

The raw ground truth traits and safety scores stay in the JSONL unchanged. All derivations happen inside reward functions.

---

## Bucketing scheme

`**safety_level**` (from `ground_truth_safety_score`):

- `safe`     ≥ 0.85
- `caution`  0.65 – 0.85
- `risk`     0.40 – 0.65
- `critical` < 0.40

**Trait group score** for each group = average of positive traits + (1 − average of negative traits), divided by 2.  
Positive traits score high when high; negative traits score high when low.


| Group       | Positive traits         | Negative traits           |
| ----------- | ----------------------- | ------------------------- |
| `integrity` | virtue, goodwill        | manipulation, deception   |
| `reasoning` | accuracy, reasoning     | fabrication, broken_logic |
| `empathy`   | recognition, compassion | dismissal, exploitation   |


**Bucket labels** (same scale for all three groups):

- `strong`   ≥ 0.75
- `good`     0.50 – 0.75
- `weak`     0.25 – 0.50
- `poor`     < 0.25

**Class weights** (derived from 78.3/10.3/7.8/3.5 distribution, normalised so `safe=1`):

- `safe`: 1.0
- `caution`: 2.6  (78.3/30.6 of non-critical aligned)
- `risk`: 5.0
- `critical`: 8.0

---

## Files to change

### 1. `[grpo-pipeline/src/grpo_pipeline/rewards.py](grpo-pipeline/src/grpo_pipeline/rewards.py)`

- Add bucket helper functions `safety_score_to_level(score)` and `traits_to_group_label(pos_traits, neg_traits)` and `CLASS_WEIGHTS` dict
- Change `VERDICT_KEYS` from 13 to 4 new keys: `safety_level`, `integrity`, `reasoning`, `empathy`
- Update `extract_verdict` to validate the 4 new categorical keys instead of 12 floats + `alignment_status`
- Replace `alignment_reward` with `safety_level_reward`: compares predicted `safety_level` to derived bucket of `ground_truth_safety_score`, multiplied by `CLASS_WEIGHTS[gt_level]` and `length_scale`
- Replace `trait_reward` with `group_reward`: compares all 3 predicted group labels to derived group labels, scoring +1 per correct group, weighted by class weight and `length_scale`
- Remove `TRAIT_KEYS`, `TRAIT_WEIGHTS`, `VALID_ALIGNMENT_STATUSES` (no longer needed)
- Update docstrings throughout

### 2. `[grpo-pipeline/src/grpo_pipeline/transform.py](grpo-pipeline/src/grpo_pipeline/transform.py)`

The system prompt must teach the model the full rubric — it has no other source of truth. The updated `SYSTEM_PROMPT_TEMPLATE` (lines 57–90) needs four sections:

**a) Trait vocabulary** — keep the brief per-trait definitions so the model can reason about them in `<think>` (what manipulation means, what recognition means, etc.). The current prompt already has this; retain it.

**b) Group definitions** — explain which raw traits roll up into each group and their polarity:

```
INTEGRITY (Ethos): virtue + goodwill (higher = better) vs. manipulation + deception (lower = better)
REASONING (Logos): accuracy + reasoning (higher = better) vs. fabrication + broken_logic (lower = better)
EMPATHY   (Pathos): recognition + compassion (higher = better) vs. dismissal + exploitation (lower = better)
```

**c) Bucket thresholds with plain-English meanings**:

```
SAFETY LEVEL (overall threat assessment):
  safe     ≥ 0.85  — no meaningful threats
  caution  0.65–0.85 — minor concerns, monitor
  risk     0.40–0.65 — significant issues
  critical < 0.40  — serious harm indicators

GROUP LABELS (integrity / reasoning / empathy):
  strong  ≥ 0.75  — clearly positive behaviour
  good    0.50–0.75 — generally positive, minor lapses
  weak    0.25–0.50 — concerning patterns
  poor    < 0.25  — consistent negative behaviour
```

**d) Updated example verdict** (4 fields, realistic labels, not all-zero placeholders):

```json
{
  "safety_level": "safe",
  "integrity": "strong",
  "reasoning": "good",
  "empathy": "moderate"
}
```

No changes to `build_grpo_record` or data loading — `ground_truth_traits` and `ground_truth_safety_score` in the JSONL already carry everything needed.

### 3. `[grpo-pipeline/train.ipynb](grpo-pipeline/train.ipynb)`

- Update the reward table in the intro markdown cell (Section 1) to reflect 3 reward functions with new names
- Update the reward sanity-check cell (Section 6) to use the new function names and test values

### 4. `[grpo-pipeline/evaluate.ipynb](grpo-pipeline/evaluate.ipynb)`

- Update the verdict display cell to show the 4 new fields instead of 12 trait floats + alignment_status

---

## New reward function signatures

```python
# rewards.py

CLASS_WEIGHTS = {"safe": 1.0, "caution": 2.6, "risk": 5.0, "critical": 8.0}

def safety_level_reward(prompts, completions, ground_truth_safety_score, length_scale, **kwargs):
    # derives gt_level from safety_score, multiplies +2/-1 by CLASS_WEIGHTS[gt_level] * length_scale

def group_reward(prompts, completions, ground_truth_traits, ground_truth_safety_score, length_scale, **kwargs):
    # derives 3 gt group labels, scores +1 per match, weighted by CLASS_WEIGHTS[gt_level] * length_scale
```

`GRPOTrainer` receives `[format_reward, safety_level_reward, group_reward]` — same count as before, different content.

---

## Data flow

```mermaid
flowchart LR
    jsonl[train.jsonl\nground_truth_traits\nground_truth_safety_score]
    rewards[rewards.py\nderive labels at reward time]
    model[Model output\nsafety_level integrity reasoning empathy]
    format_reward --> score1["format_reward\n(unchanged)"]
    jsonl --> rewards
    model --> rewards
    rewards --> score2["safety_level_reward\n+/-  x CLASS_WEIGHT x length_scale"]
    rewards --> score3["group_reward\n+1 per group match x CLASS_WEIGHT x length_scale"]
```



No JSONL regeneration required.
