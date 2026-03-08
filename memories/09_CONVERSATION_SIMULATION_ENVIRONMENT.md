---
name: Conversation Simulation Environment
overview: Live simulation layer that runs during training. ParticipantBots replay historical threads turn-by-turn into a ConversationEnvironment; SimulatedDataset wraps this as a streaming HuggingFace IterableDataset fed directly to GRPOTrainer. The ParticipantBot ABC is designed to also back LLM-powered bots (implemented in memory 10). All todos completed and unit-tested.
todos:
  - id: simulation-module
    content: "Create grpo-pipeline/src/grpo_pipeline/simulation.py: ParticipantBot, ConversationEnvironment, and SimulatedDataset (IterableDataset generator)"
    status: completed
  - id: train-py-update
    content: Add --raw-data-dir and --min-context-turns flags to train.py; when --raw-data-dir is set, load from SimulatedDataset instead of static JSONL
    status: completed
  - id: notebook-update
    content: Add USE_LIVE_SIM toggle to train.ipynb Section 2 (config) and Section 4 (data loading)
    status: completed
  - id: readme-update
    content: Update grpo-pipeline/README.md to document the live simulation mode and min_context_turns
    status: completed
  - id: unit-tests
    content: "Write tests/test_simulation.py: ReplayBot, ConversationEnvironment, min_context_turns, SimulatedDataset, equivalence with transform.py output"
    status: completed
isProject: false
---

# Conversation Simulation Environment

## Architecture

The current training loop loads a static pre-transformed JSONL. The new architecture feeds GRPOTrainer a live streaming dataset that runs the conversation simulation on each iteration.

```mermaid
flowchart TD
    subgraph current [Current]
        A[raw-data] --> B[transform.py]
        B --> C[transformed/train.jsonl]
        C --> D[Dataset.from_list]
        D --> E[GRPOTrainer]
    end

    subgraph new [New - Live Simulation]
        F[raw-data] --> G[simulation.py]
        G --> H[ParticipantBot per author]
        H --> I[ConversationEnvironment]
        I --> J["SimulatedDataset\n(IterableDataset generator)"]
        J --> K[GRPOTrainer]
    end
```



Each training step, GRPOTrainer pulls the next batch from `SimulatedDataset`, which continuously cycles through conversation threads — replaying them turn-by-turn — in shuffled order. The prompt format fed to the model is identical to the current format.

## New File: `[grpo-pipeline/src/grpo_pipeline/simulation.py](grpo-pipeline/src/grpo_pipeline/simulation.py)`

Three components:

`**ParticipantBot**` — abstract base wrapping a single author's messages, emitted chronologically. The same interface will back LLM-powered bots later.

```python
class ParticipantBot:
    author: str
    # context_so_far is optional so ReplayBot (which ignores it) and
    # LLMParticipantBot (which needs it) share the same interface
    def next_message(self, context_so_far: list[ConversationRecord] | None = None) -> ConversationRecord | None: ...
    def is_exhausted(self) -> bool: ...

class ReplayBot(ParticipantBot):
    """Replays historical ConversationRecord messages in order."""
```

`**ConversationEnvironment**` — manages one thread's state. Holds all messages pre-sorted by `created_at`; each `step()` pops the next one, appends it to the running context, and returns `(message, context_so_far)`.

```python
class ConversationEnvironment:
    def step(self) -> tuple[ConversationRecord, list[ConversationRecord]] | None: ...
    def run_to_records(self, min_context_turns: int) -> list[GRPORecord]: ...
```

`run_to_records` calls `transform.build_grpo_record` (already a pure function — no changes needed there) with a `min_context_turns` gate: records where `turn_index < min_context_turns` are skipped, and `length_scale` is recalculated over the filtered set.

`**SimulatedDataset**` — wraps the simulation as a HuggingFace `IterableDataset` suitable for direct use as `train_dataset` in `GRPOTrainer`.

```python
class SimulatedDataset:
    @staticmethod
    def generate(raw_data_dir: str, min_context_turns: int, seed: int):
        """Infinite generator: shuffles threads each epoch, yields GRPORecord dicts."""
        rng = random.Random(seed)
        while True:
            threads = load_all_threads(raw_data_dir)
            rng.shuffle(threads)
            for thread_id, source_file, messages in threads:
                env = ConversationEnvironment(thread_id, messages)
                for record in env.run_to_records(source_file, min_context_turns):
                    yield record.model_dump()

    @staticmethod
    def collect_one_epoch(raw_data_dir, min_context_turns=0) -> list[dict]:
        """Single deterministic epoch — used for SFT warmup and offline inspection."""

    @classmethod
    def create(cls, raw_data_dir, min_context_turns=0, seed=42) -> IterableDataset:
        return IterableDataset.from_generator(cls.generate, gen_kwargs={...})

    @classmethod
    def create_with_llm_bots(cls, raw_data_dir, bot_profiles_dir,
                              participant_backend, oracle_backend,
                              min_context_turns=0, seed=42) -> IterableDataset:
        """LLM bot variant — see memory 10."""
```

The generator loops infinitely; GRPOTrainer receives the exact same column schema as `train.jsonl`. `format_prompts` is applied on top unchanged. `collect_one_epoch` returns a plain `list[dict]` for SFT warmup sampling when `--raw-data-dir` is set (the `IterableDataset` is not indexable).

## Changes to `[train.py](grpo-pipeline/src/grpo_pipeline/train.py)`

Add two new CLI flags:

- `--raw-data-dir PATH` — when set, use `SimulatedDataset.create()` instead of loading TRAIN_FILE
- `--min-context-turns N` (default `0`) — passed through to `SimulatedDataset.create()`

The data-loading block becomes:

```python
if args.raw_data_dir:
    from grpo_pipeline.simulation import SimulatedDataset
    raw_dataset = SimulatedDataset.create(args.raw_data_dir, args.min_context_turns)
    dataset = raw_dataset.map(format_prompts)          # streaming map
else:
    raw_records = [json.loads(l) for l in open(TRAIN_FILE) if l.strip()]
    dataset = Dataset.from_list(raw_records)
    dataset = dataset.map(format_prompts, batched=True)
```

Because `IterableDataset` has no length, `max_steps` must be set (already the case in all current configs).

## Changes to `[train.ipynb](grpo-pipeline/train.ipynb)`

Section 4 "Load & Prepare Training Data" gets a `USE_LIVE_SIM` toggle at the top:

```python
USE_LIVE_SIM       = True          # False = load from static train.jsonl (current behaviour)
RAW_DATA_DIR       = '../raw-data' # only used when USE_LIVE_SIM = True
MIN_CONTEXT_TURNS  = 1             # skip turn 0 of each thread (no prior context)
```

The two loading paths (static vs simulated) are clearly delimited under the toggle so users can see exactly what changes.

## Minimum Context Gate

The `min_context_turns` parameter gates out early turns that carry noisy signal (discussed in prior analysis):

- `0` — identical to current behaviour, all turns included
- `1` — drop the first message in every thread (oversight agent always has at least 1 prior message)
- `2` — require at least 2 prior messages before emitting a training record

`length_scale` is recalculated over the filtered turn set so the reward scaling stays consistent.

## What `transform.py` Becomes

`transform.py` is no longer in the hot path for training. It remains useful for:

- Generating a static `test.jsonl` for evaluation (used by `evaluate.ipynb`)
- Offline inspection of the dataset distribution
- `split.py` still operates on its output for evaluation splits

No changes needed to `transform.py` for this plan.

## Unit Tests

`grpo-pipeline/tests/test_simulation.py` (58 tests, all passing). Key coverage:

- `ReplayBot` message ordering and exhaustion
- `ConversationEnvironment.step()` — context growth, turn ordering
- `run_to_records()` — chronological order, `min_context_turns` filtering, messages without evaluations skipped
- Equivalence with `transform.py` output from the oversight agent's POV (prompt format, system prompt injection, `length_scale`)
- `SimulatedDataset` schema, infinite generation, per-epoch shuffle reproducibility
- Integration test against real raw-data files

## LLM-Powered Bots Extension

Implemented — see memory 10. `ParticipantBot.next_message()` accepts `context_so_far` so `LLMParticipantBot` can use it; `ReplayBot` ignores it. `SimulatedDataset.create_with_llm_bots()` added alongside the existing `create()`.

## Files Touched

- `grpo-pipeline/src/grpo_pipeline/simulation.py` — new file; updated with `create_with_llm_bots()` and `_generate_llm()` in follow-up
- `grpo-pipeline/src/grpo_pipeline/train.py` — `--raw-data-dir`, `--min-context-turns` (+ LLM bot flags in follow-up)
- `grpo-pipeline/train.ipynb` — `USE_LIVE_SIM` toggle in Section 2 and Section 4
- `grpo-pipeline/tests/test_simulation.py` — new file (58 tests)
- `grpo-pipeline/README.md` — live simulation mode documented
- No changes to `rewards.py`, `models.py`, `split.py`, or `transform.py`
