# Simply Codebase Notes

A personal onboarding note — what I learned while first exploring this repo.
Kept separate from `experiments_note.md` (that file is for experiment logs;
this one is for understanding the code).

Last updated: 2026-06-14

---

## What Simply is

A minimal, scalable JAX research codebase from Google DeepMind for end-to-end
LLM work: pretraining, RL post-training, and inference. Design philosophy:
**minimal abstractions, few dependencies** — "learn JAX and you can read and
hack everything." Built to be forked and hacked on quickly, by humans and AI
agents. Supports Gemma / Qwen / DeepSeek families with multi-host distributed
training.

---

## The execution flow (how a run works)

Entry point: `simply/main.py`. Example:

```bash
python -m simply.main --experiment_config lm_test --experiment_dir /tmp/exp
```

1. **`main()`** (`main.py:165`) — init JAX distributed (skips gracefully on a
   single host), then loads config and dispatches to the train loop.
2. **`load_experiment_config()`** (`main.py:135`) — looks up a config by name
   from `ExperimentConfigRegistry`, applies mesh/sharding overrides from CLI
   flags, runs any `code_patch` snippets.
3. **`TrainLoopRegistry.get(config.train_loop_name)`** (`main.py:177`) — finds
   the train loop (default `'default'`) and calls it with the config.
4. **`run_experiment()`** (`model_lib.py:2963`) — the heart of a run:
   set up device mesh → `create_model()` → `get_init_state()` (init or restore
   from checkpoint) → `jax.jit`-compile `train_one_step` → build data iterator
   → loop with periodic eval + checkpointing.

Mental model: **config (data) → registry lookup → JIT'd train loop.** No
framework magic beyond that.

---

## The four core abstractions

### 1. Registry pattern — `utils/registry.py`
A name→class/function dict (`RootRegistry`). Everything pluggable is a
registered dataclass:

```python
@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class BaseExperimentConfig(ExperimentConfig): ...
```

Registered names double as serialization keys (`__registered_name__`), so
configs round-trip to JSON. Key registries: `ExperimentConfigRegistry`,
`ShardingConfigRegistry`, `ModuleRegistry`, `OptimizerRegistry`,
`TrainLoopRegistry`, `DataSourceRegistry`, `ToolRegistry`.

### 2. `SimplyModule` — `utils/module.py:38`
A stripped-down `flax.nn.Module`. Every model component is a dataclass with
three methods: `setup()`, `init(prng_key) -> params`, `apply(params, x) -> y`.
Params are **explicit PyTrees passed in and out** — no hidden state. That's the
entire neural-net abstraction.

### 3. `AnnotatedArray` — `utils/common.py`
Wraps arrays with sharding/metadata annotations that flow through the model so
the sharding layer knows how to partition each parameter across the mesh.

### 4. Config-driven everything — `config_lib.py:240`
`BaseExperimentConfig` is one big frozen dataclass (~100 fields) describing the
*entire* experiment: architecture (`model_dim`, `n_heads`, `n_layers`, MoE
settings), data (`dataset`, `batch_size`), optimization (`optimizer`, `lr`,
`num_train_steps`), checkpointing, distillation, etc. An experiment *is* a
registered instance of this. `config_lib.py` ships 60+ ready-to-use configs
selected via `--experiment_config`.

---

## Module map

| Module | Role |
|---|---|
| `main.py` | Entry point; flag parsing, config loading |
| `config_lib.py` | `ExperimentConfig` + sharding configs + 60+ experiments |
| `model_lib.py` | Big one: `Attention`, `FeedForward`, `MoEFeedForward`, `TransformerBlock`, `TransformerLM`, plus `train_one_step`, `run_experiment`, `LMInterface` (sampling/scoring) |
| `data_lib.py` | Grain-native data pipeline (sources, tokenization, packing, mixtures) |
| `rl_lib.py` | RL post-training (reward normalization, batching, RL loop) |
| `tool_lib.py` | Tool-use / execution framework |
| `utils/` | `sharding.py` (FSDP/TP/expert-parallel), `checkpoint_lib.py` (Orbax), `optimizers.py`, `sampling_lib.py`, `lm_format.py` (chat), `moe_lib.py`, position encodings |
| `agent/` | Built-in autonomous agent harness (Bash tool, memory, TUI) |
| `eval/`, `serving/`, `kernels/` | Decode-time eval; inference server (paged attention, gRPC); Pallas kernels |

---

## Where to hack (by goal)

- **New optimizer / loss / LR schedule** → `utils/optimizers.py` +
  `compute_loss` / `train_one_step` in `model_lib.py`
- **New architecture variant** → `SimplyModule` subclasses in `model_lib.py`
  (toggled by config flags like `use_moe`, `use_qk_norm`, `block_attn_pattern`)
- **New data mixture** → register a `DatasetConfig` factory in `data_lib.py`
- **New experiment** → register a `BaseExperimentConfig` subclass in
  `config_lib.py`
- **RL** → `rl_lib.py` + `RLExperimentConfig` (`config_lib.py:432`)

---

## Useful commands

```bash
# Local test run
python -m simply.main --experiment_config lm_test \
    --experiment_dir /tmp/exp --alsologtostderr

# Debug mode: disable JIT + scan so you can print arrays like normal Python
export JAX_DISABLE_JIT=True
python -m simply.main --experiment_config lm_no_scan_test \
    --experiment_dir /tmp/exp --alsologtostderr

# TensorBoard
tensorboard --logdir /tmp/exp

# Tests
pytest simply/
pytest simply/model_lib_test.py::ModelTest::test_forward_pass
```

Env vars: `SIMPLY_MODELS`, `SIMPLY_DATASETS`, `SIMPLY_VOCABS`,
`JAX_DISABLE_JIT`.

---

## Open questions / next to explore

- [ ] Trace the full `TransformerLM` forward pass end to end
      (`model_lib.py:2178`).
- [ ] Understand the mesh / sharding setup (`utils/sharding.py`): FSDP vs TP
      vs expert parallelism, and how `mesh_shape` flags map to axes.
- [ ] Walk the data pipeline: source → tokenize → pack → mixture
      (`data_lib.py`, also documented in `CLAUDE.md`).
- [ ] Read the RL loop in `rl_lib.py` and how it differs from the default loop.
- [ ] Look at how checkpoints save/restore (`utils/checkpoint_lib.py`, Orbax).
- [ ] Try the agent harness (`simply/agent/`) on an example task.
