# Experiment Log

## Index

- [2026-06-13 — local env + GPU driver upgrade](#experiment-log-5)
- [2026-06-13 — rebase, onboarding, PR #29](#experiment-log-4)
- [2026-06-11](#experiment-log-3)
- [2026-04-16](#experiment-log-1)

---

<a id="experiment-log-5"></a>
# Experiment Log

## Date

2026-06-13

## Goal

Stand up a local environment for interactively testing `simply` code in a
Jupyter notebook, and get JAX running on the box's two A100 GPUs.

## Environment

| Component | Value |
|---|---|
| Host | Azure VM, Ubuntu 22.04.5, kernel `6.8.0-1044-azure` |
| GPUs | 2 × NVIDIA A100 80GB **PCIe** (no NVSwitch) |
| Project Python | `>=3.12` (`pyproject.toml`); conda envs were all 3.8/3.10 |
| Build tool | `uv` (repo ships `uv.lock`) |

## Steps

### 1. Build the project env from the lockfile

No conda env had JAX, and all were < 3.12. Installed `uv` and synced the
locked deps into a repo-local `.venv` (uv fetched CPython 3.12):

```bash
pip install uv
uv sync --extra gpu --extra tfds --extra dev
```

Result: `.venv/` with Python 3.12, JAX 0.9.0.1, grain/orbax/TFDS/pytest.
`import simply` works.

### 2. Register a Jupyter kernel + scratch notebook

```bash
uv pip install --python .venv/bin/python ipykernel
.venv/bin/python -m ipykernel install --user --name simply \
  --display-name "Python (simply)"
```

Created `notebooks/scratch.ipynb` (setup cell adds repo root to `sys.path`,
env check, `%autoreload 2`). Kernel preset to **Python (simply)**.

### 3. GPU mismatch — driver too old for bundled CUDA

`uv sync --extra gpu` installed `jax[cuda13]` (CUDA 13 runtime), but the
host driver is **535.274.02 (max CUDA 12.2)**. JAX saw the GPUs but every
kernel failed: `CUDA_ERROR_INVALID_IMAGE`. Swapping to `jax[cuda12]` (bundles
CUDA 12.9) got the same image error — minor-version forward-compat can't
bridge 12.2 → 12.9 for compiled SASS.

Within this repo's deps JAX resolves to 0.9 (older JAX that bundles CUDA
≤12.2 would break orbax/grain), so the only real fix is a **host driver
upgrade**. As an interim, registered the kernel with `JAX_PLATFORMS=cpu` so
the notebook works CPU-only:

```bash
.venv/bin/python -m ipykernel install --user --name simply \
  --display-name "Python (simply)" --env JAX_PLATFORMS cpu
```

### 4. Driver upgrade runbook (535 → 580, for CUDA 13 / `jax[cuda13]`)

Diagnosis of the first failed `apt install cuda-drivers-580`: the 580 pkgs
*are* available, but (a) the installed **non-server** 535 stack conflicted
with apt pulling **-server** 580 variants (`nvidia-kernel-common`,
`nvidia-fabricmanager` Conflicts), and (b) a stale local repo
(`nvidia-driver-local-repo-ubuntu2204-535.183.06`, priority 600) plus a
wrong-distro `cudnn-local-repo-ubuntu2004` pinned old packages. Fix =
purge old stack + stale repos, then install one coherent branch. PCIe A100s
need no fabricmanager.

```bash
# 0. snapshot
nvidia-smi > ~/nvidia_before_upgrade.txt

# 1. stop services + remove stale/wrong-distro local repos
sudo systemctl stop nvidia-fabricmanager 2>/dev/null
sudo apt-get purge -y nvidia-driver-local-repo-ubuntu2204-535.183.06 \
                      cudnn-local-repo-ubuntu2004-9.1.1

# 2. purge the 535 driver stack (scoped to -535; leaves container toolkit)
sudo apt-get purge -y 'nvidia-*-535' 'libnvidia-*-535' \
    cuda-drivers-535 cuda-drivers-fabricmanager-535 nvidia-fabricmanager-535
sudo apt-get autoremove -y
sudo apt-get update

# 3. install clean 580 (non-server, matches pyproject 'gpu' extra)
sudo apt-get install -y cuda-drivers-580

# 4. confirm DKMS built before rebooting
dkms status        # expect nvidia/580.159.04, 6.8.0-1044-azure: installed

# 5. reboot + verify
sudo reboot
nvidia-smi         # expect Driver 580.159.04, CUDA 13.x, both A100s
```

Rollback: `sudo apt-get install -y cuda-drivers-535 && sudo reboot`.

### 5. Post-upgrade: flip the venv back to GPU (after 580 is live)

```bash
cd /home/azureuser/Projects/simply-googledeepmind
uv pip uninstall --python .venv/bin/python jax-cuda12-plugin jax-cuda12-pjrt
uv pip install   --python .venv/bin/python "jax[cuda13]"
# re-register WITHOUT the CPU lock
.venv/bin/python -m ipykernel install --user --name simply \
  --display-name "Python (simply)"
.venv/bin/python -c "import jax; print(jax.devices())"  # expect 2 CudaDevices
```

## Results

| Item | Outcome |
|---|---|
| Project `.venv` (Python 3.12) | Built from `uv.lock`; `import simply` OK |
| Jupyter kernel `Python (simply)` | Registered; CPU JAX verified (matmul OK) |
| `notebooks/scratch.ipynb` | Created, kernel preset |
| GPU execution | Blocked on driver 535 < CUDA 13; runbook above |

## Next steps

- Run the step-4 runbook (sudo) to upgrade to driver 580, then step 5 to
  re-enable GPU JAX and confirm both A100s execute.

---

<a id="experiment-log-4"></a>
# Experiment Log

## Date

2026-06-13

## Goal

1. Confirm the fork is rebased on the latest upstream.
2. Explore the codebase and write onboarding notes.
3. Make a first upstream contribution.
4. Check personal work-in-progress into the fork.

## Environment

| Component | Value |
|---|---|
| Fork (`origin`) | `github.com/chenchenpan/simply-googledeepmind` |
| Upstream | `github.com/google-deepmind/simply` |
| Commit identity | `chenchenpan <pcc.pku@gmail.com>` (local to this repo) |
| `gh` CLI account | `chenchenpan_microsoft` (Enterprise Managed User) |
| Push auth | PAT for `chenchenpan` via repo-local `credential.helper=store` |

## Steps

### 1. Verify rebase on upstream

`upstream` remote was missing. Added it, fetched, and found `main` was 1 behind
(`c9a45dc Skip jax.distributed.initialize() on single-host runs`). Rebased the
6 local commits on top — conflict-free since they don't touch `main.py`.

```bash
git remote add upstream https://github.com/google-deepmind/simply.git
git fetch upstream
git rebase upstream/main
git push --force-with-lease origin main
```

### 2. Fix push auth (EMU vs fork owner)

`git push` returned 403: the only `gh`/global identity is
`chenchenpan_microsoft`, an Enterprise Managed User with no write access to the
`chenchenpan` fork. Configured a **repo-local** credential store pointing at a
PAT for `chenchenpan`, so the global identity is untouched for other repos:

```bash
# token stored at .git/.gh-credentials (perms 600, inside .git so untrackable)
git config --local credential.https://github.com.helper ""   # reset gh helper
git config --local --add credential.https://github.com.helper \
  "store --file=$PWD/.git/.gh-credentials"
```

### 3. Explore codebase + onboarding notes

Read `main.py` → config → `TrainLoopRegistry` → `run_experiment` flow, the four
core abstractions (registry, `SimplyModule`, `AnnotatedArray`, config-driven
`BaseExperimentConfig`), and the module map. Wrote `codebase_notes.md`
(fork-only, committed to `main`).

### 4. First upstream contribution — `math_eval` fix + tests

Picked a self-contained TODO: `math_eval._str_to_int` converted via `float(x)`,
losing integer precision above 2^53 (e.g. `'9007199254740993'` →
`9007199254740992`). Fixed by parsing integers directly:

```python
def _str_to_int(x: str) -> int:
  x = x.replace(',', '')
  try:
    return int(x)            # arbitrary precision
  except ValueError:
    return int(float(x))     # fallback for '1.0', '5e3'
```

Also added `simply/utils/math_eval_test.py` — first tests for the module,
`absltest` + `parameterized`, 66 tests, all passing (run via
`python -m simply.utils.math_eval_test`; `pytest` not in this env).

Branched off **`upstream/main`** (not `main`, to avoid dragging in fork-only
commits), single commit, pushed to fork, opened **PR #29** via the REST API
(the `gh` EMU account is blocked from PRs to external repos):

```bash
git rebase --onto upstream/main main fix/math-eval-int-precision
git push -u origin fix/math-eval-int-precision
GH_TOKEN=<chenchenpan PAT> gh api -X POST repos/google-deepmind/simply/pulls \
  -f title=... -f head="chenchenpan:fix/math-eval-int-precision" -f base=main ...
```

### 5. Fix failing CLA check (email mismatch)

`cla/google` failed: the commit was authored `chenchenpan@gmail.com`, but the
signed CLA / GitHub account uses `pcc.pku@gmail.com`. The bot matches on commit
email → account → CLA. Re-authored the commit and set the repo-local email so
future commits match; CLA passed after force-push.

```bash
git config --local user.email pcc.pku@gmail.com
GIT_COMMITTER_EMAIL=pcc.pku@gmail.com \
  git commit --amend --no-edit --author="chenchenpan <pcc.pku@gmail.com>"
git push --force-with-lease origin fix/math-eval-int-precision
```

### 6. Check personal WIP into the fork

Committed to `main` (kept off the PR branch via stash → checkout → pop):
- `uv.lock` — locks the `[agent]` extra deps (litellm, openai, tiktoken, …).
- `notebooks/scratch.ipynb` — cleared-output scratch notebook.
- `.gitignore` — ignore `.ipynb_checkpoints/`.

## Results

| Item | Outcome |
|---|---|
| `main` vs `upstream/main` | Rebased; 0 behind. Fork-only commits ride on top. |
| Onboarding notes | `codebase_notes.md` on `main` |
| Upstream PR #29 | Open, `cla/google` ✅, awaiting review |
| Push auth | Repo-local PAT store for `chenchenpan`; global `gh` untouched |

## Next steps

- Watch PR #29 for maintainer review / CI.
- **Rotate the PAT** — it was pasted in plaintext this session; update the
  password in `.git/.gh-credentials` after creating a new `repo`-scoped token.
- Earlier `main` commits are still authored `chenchenpan@gmail.com`; re-author
  to `pcc.pku@gmail.com` before contributing any of them upstream.

---

<a id="experiment-log-3"></a>
# Experiment Log

## Date

2026-06-11

## Goal

Sync this fork with the upstream repo and contribute a subset of local changes
back upstream via pull requests.

## Environment

| Component | Value |
|---|---|
| Fork (`origin`) | `github.com/chenchenpan/simply-googledeepmind` |
| Upstream | `github.com/google-deepmind/simply` |
| Commit identity | `chenchenpan <chenchenpan@gmail.com>` (local to this repo) |
| Auth | PAT via `credential.helper=store`, scopes `repo` + `workflow` |

## Steps

### 1. Identify the upstream repo

The fork had only the `origin` remote. Found the parent via the GitHub API
(`api.github.com/repos/chenchenpan/simply-googledeepmind` → parent
`google-deepmind/simply`) and added it as `upstream`.

### 2. Sync `main` with upstream

`main` was 5 ahead / 6 behind. Rebased onto `upstream/main`, re-authoring the
5 local commits to `chenchenpan@gmail.com` so GitHub attributes them correctly:

```bash
git fetch upstream
git rebase upstream/main --exec 'git commit --amend --reset-author --no-edit'
git push --force-with-lease origin main
```

### 3. Set up credential caching

`gh` is not installed. Seeded a PAT into `~/.git-credentials` (perms 600) with
`credential.helper=store` so pushes don't prompt. Note: pushing changes under
`.github/workflows/` requires the token to have the `workflow` scope, not just
`repo`.

### 4. Contribute a subset of commits (fork → PR)

Created a clean branch per change off `upstream/main`, cherry-picked only the
wanted commit, pushed to the fork, and opened a PR via the cross-fork compare
URL:

```bash
git switch -c <branch> upstream/main
git cherry-pick <sha>
git push origin <branch>
# PR: github.com/google-deepmind/simply/compare/main...chenchenpan:simply-googledeepmind:<branch>?expand=1
```

Two PRs opened:
- `add-uv-setup-readme` — README "Quick setup with uv" subsection.
- `fix-duplicate-tfds-uv-lock` — dedup `tensorflow-datasets` in `uv.lock`.

### 5. Clean commit messages

Removed the `Co-Authored-By: Claude` trailer from all commits before they went
public (body was trailer-only, so the new message is just the subject):

```bash
git rebase <base> --exec 'git commit --amend -m "$(git log -1 --format=%s)"'
```

### 6. Resolve and clean up

- Signed the upstream CLA.
- README PR **merged** upstream (`5e869bd`) → deleted its branch (local + fork)
  and rebased `main`, which auto-dropped the now-upstream commit.
- uv.lock PR **abandoned** → deleted its branch; the fix stays on the fork's
  `main` only.

## Results

| Item | Outcome |
|---|---|
| `main` vs `upstream/main` | 4 ahead, 0 behind (3 experiment notes + local uv.lock fix) |
| README contribution | Merged upstream (`5e869bd`) |
| uv.lock contribution | Abandoned (kept locally) |
| Fork `origin/main` | In sync with local `main` |

## Next steps

- Keep the fork current: `git fetch upstream && git rebase upstream/main &&
  git push --force-with-lease origin main`.

---

<a id="experiment-log-1"></a>
# Experiment Log

## Date

2026-04-16

## Goal

Test the Simply codebase setup end-to-end:
1. Run `lm_test` on CPU to verify the baseline setup.
2. Enable GPU support and run the same test on a Tesla T4.

### Steps

- [Step 1: Install dependencies](#apr16-step-1)
- [Step 2: CPU run — `lm_test`](#apr16-step-2)
- [Step 3: Enable GPU — `lm_test` in a screen session](#apr16-step-3)
- [Step 4: Pretraining from scratch — `flops2e16_tfm15m_imdb`](#apr16-step-4)
- [Step 5: Monitoring with TensorBoard](#apr16-step-5)

---

## Environment

| Component | Version / Value |
|---|---|
| Machine | Azure Linux VM |
| GPU | NVIDIA Tesla T4 (15 GB) |
| NVIDIA driver | 555.42.06 (max CUDA runtime: 12.5) |
| System CUDA toolkit | 12.2 (`/usr/local/cuda-12.2`) |
| Python | 3.12.9 (via pyenv) |
| Package manager | uv 0.10.0 |
| JAX | 0.9.0.1 |

---

<a id="apr16-step-1"></a>
## Step 1: Install dependencies

```bash
uv sync --extra tfds --extra gpu
```

- `--extra tfds` installs TensorFlow and `tensorflow-datasets` (needed for the IMDB dataset used by `lm_test`).
- `--extra gpu` installs `jax[cuda13]` (CUDA 13 pip packages) — this later caused a driver mismatch (see Bug 2).

---

<a id="apr16-step-2"></a>
## Step 2: CPU run — `lm_test`

### Command

```bash
EXP=simply_local_test_1
rm -rf /tmp/${EXP}
uv run python -m simply.main \
  --experiment_config lm_test \
  --experiment_dir /tmp/${EXP} \
  --alsologtostderr
```

### Bug 1: TFDS ARRAY_RECORD format mismatch

**Error:**
```
ValueError: File format is already set to FileFormat.TFRECORD. Got FileFormat.ARRAY_RECORD
```

**Cause:** The IMDB dataset had been previously downloaded in the old `TFRECORD` format, but `tfds.data_source()` (used in `data_lib.py`) requires the newer `ARRAY_RECORD` format for random access.

**Fix:** Delete the stale TFDS cache and let it re-download in `ARRAY_RECORD` format.
```bash
rm -rf ~/tensorflow_datasets/imdb_reviews/
```

### Result

50 training steps completed successfully.

| Metric | Value |
|---|---|
| Final train loss | 12.23 |
| Speed | ~0.56 sec/step |
| Checkpoint | `/tmp/simply_local_test_1/checkpoints/50` |

---

<a id="apr16-step-3"></a>
## Step 3: Enable GPU — `lm_test` in a screen session

### Goal

Run the same `lm_test` with GPU acceleration in a detached `screen` session.

### Bug 2: `jax[cuda13]` incompatible with driver

**Error:**
```
cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version
```

**Cause:** `uv sync --extra gpu` installs `jax[cuda13]` which requires CUDA 13.x pip packages. These need a driver newer than 555.42.06 (which supports up to CUDA 12.5 only).

**Fix:** Switch to `jax[cuda12]`:
```bash
uv pip install "nvidia-cublas-cu12" "nvidia-nccl-cu12" ... # (via uv sync --extra gpu after pyproject change)
```

### Bug 3: `jax[cuda12]` pip packages too new for driver

**Error:**
```
cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version
```

**Cause:** Even `jax[cuda12]` resolves to the latest CUDA 12.x packages (12.9.x), which require driver ≥ 575.x. Our driver only supports up to CUDA 12.5.

**Root cause analysis:**
- JAX's `jax[cuda12]` extra specifies only minimum version constraints (e.g., `nvidia-cuda-runtime-cu12>=12.3`), so pip/uv resolves to the *latest* available version (12.9.79 at time of install).
- `nvidia-cuda-runtime-cu12==12.9.79` calls into `libcuda.so` (driver) at init time and checks driver API compatibility. Driver 555.42.06 fails this check.

**Fix:** Downgrade only the CUDA runtime package to a driver-compatible version. All other 12.9.x libraries can remain because they don't do the driver version check directly.

```bash
uv pip install "nvidia-cuda-runtime-cu12==12.5.82"
```

**Why this works:** `libcudart.so.12` version 12.5 checks "does the driver support CUDA 12.5?" — answer is yes (driver 555.42.06 is the *minimum* for CUDA 12.5). Higher-level libraries (cuBLAS, cuSPARSE, etc.) at 12.9 still function because they rely on `libcudart` for device management and are backward-compatible at the CUDA 12.x ABI level.

### Bug 4: Both `cuda12` and `cuda13` plugins registered simultaneously

**Error:**
```
jax.errors.JaxRuntimeError: ALREADY_EXISTS: PJRT_Api already exists for device type cuda
```

**Cause:** After switching from `cuda13` to `cuda12`, both plugin packages remained installed.

**Fix:** Uninstall all `cuda13` packages:
```bash
uv pip uninstall jax-cuda13-pjrt jax-cuda13-plugin \
  nvidia-cublas nvidia-cuda-cccl nvidia-cuda-cupti \
  nvidia-cuda-nvcc nvidia-cuda-nvrtc nvidia-cuda-runtime \
  nvidia-cudnn-cu13 nvidia-cufft nvidia-cusolver \
  nvidia-cusparse nvidia-nccl-cu13 nvidia-nvjitlink \
  nvidia-nvshmem-cu13 nvidia-nvvm
```

### Bug 5: `uv run` re-syncs from `uv.lock` on every invocation

**Symptom:** Manually installed package versions (e.g., jax 0.4.35, custom CUDA pins) were reverted on each `uv run` call because `uv run` performs an implicit `uv sync` before running.

**Fix:** Use `--no-sync` flag to skip the sync:
```bash
uv run --no-sync python -m simply.main ...
```

### Final working GPU setup

```bash
# Install CUDA 12 packages (default to latest)
uv sync --extra gpu --extra tfds

# Remove cuda13 packages (if also installed)
uv pip uninstall jax-cuda13-pjrt jax-cuda13-plugin [nvidia-cuda13-*]

# Pin CUDA runtime to driver-compatible version
uv pip install "nvidia-cuda-runtime-cu12==12.5.82"

# Verify GPU is visible
uv run --no-sync python -c "import jax; print(jax.devices())"
# Expected: [CudaDevice(id=0)]
```

### GPU `lm_test` command (in screen)

```bash
screen -dmS lm_test_gpu bash -c '
  EXP=simply_local_test_gpu
  rm -rf /tmp/${EXP}
  uv run --no-sync python -m simply.main \
    --experiment_config lm_test \
    --experiment_dir /tmp/${EXP} \
    --alsologtostderr \
    > /tmp/lm_test_gpu.log 2>&1
'
```

Monitor progress:
```bash
tail -f /tmp/lm_test_gpu.log
```

### Result

| Metric | Value |
|---|---|
| Final train loss | ~12.23 |
| Speed (after warmup) | ~0.03 sec/step |
| Speedup vs CPU | ~18x |
| Exit code | 0 |

---

<a id="apr16-step-4"></a>
## Step 4: Pretraining from scratch — `flops2e16_tfm15m_imdb`

### Goal

Run a pretraining experiment using the smallest scaling config (`flops2e16_tfm15m`, 14.86M params) without downloading C4. Replaced C4 with IMDB (already cached) to test the training loop end-to-end.

### New config added to `config_lib.py`

```python
@ExperimentConfigRegistry.register
def flops2e16_tfm15m_imdb():
  """flops2e16_tfm15m architecture trained on IMDB (no C4 download needed)."""
  config = flops2e16_tfm15m_c4_l2048()
  return dataclasses.replace(
      config,
      vocab_size=151_936,
      vocab_name='Qwen3',
      seq_len=256,
      batch_size=8,
      num_train_steps=500,
      dataset=data_lib.DatasetConfig(
          source=data_lib.TFDSSource(
              name='imdb_reviews', split='train'),
          lm_format_name='Pretrain',
      ),
      validation_dataset=data_lib.DatasetConfig(
          source=data_lib.TFDSSource(
              name='imdb_reviews', split='test'),
          lm_format_name='Pretrain',
      ),
      validation_num_eval_steps=8,
      validation_eval_interval=100,
      validation_eval_batch_size=8,
      lr=opt_lib.LinearWarmupCosineDecay(
          value=0.01,
          warmup_steps=50,
          steps_after_decay=0,
          end_decay=0.1,
      ),
      ckpt_interval=100,
      tb_log_interval=10,
  )
```

### Bug 6: GPU OOM during forward pass

**Error:**
```
Allocator (GPU_0_bfc) ran out of memory trying to allocate 9.27GiB
```

**Cause:** The logits tensor shape is `[batch_size, seq_len, vocab_size]`. With `batch_size=32`, `seq_len=512`, `vocab_size=151936` and float32, this is 32 × 512 × 151936 × 4 bytes ≈ **9.25 GiB** — too large for the 15 GB T4.

**Fix:** Reduce `batch_size` and `seq_len`:
- `batch_size`: 32 → 8
- `seq_len`: 512 → 256

Logits tensor is now 8 × 256 × 151936 × 4 bytes ≈ **580 MB**.

Also set env vars to prevent TF and JAX from competing for GPU memory at init:
```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

### Command (in screen)

```bash
screen -dmS pretrain_imdb bash -c '
  EXP=simply_pretrain_imdb_15m
  rm -rf /tmp/${EXP}
  export TF_FORCE_GPU_ALLOW_GROWTH=true
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  uv run --no-sync python -m simply.main \
    --experiment_config flops2e16_tfm15m_imdb \
    --experiment_dir /tmp/${EXP} \
    --alsologtostderr \
    > /tmp/pretrain_imdb.log 2>&1
  echo "EXIT CODE: $?" >> /tmp/pretrain_imdb.log
'
```

Monitor:
```bash
tail -f /tmp/pretrain_imdb.log | grep -E "train_loss:|secs per step"
```

### Result (in progress, 500 steps total)

| Metric | Value |
|---|---|
| Model | 14.86M params, 4 layers, 128 dim |
| Dataset | IMDB reviews (train / test split) |
| Loss at step 10 | 12.44 (random init) |
| Loss at step ~40 | ~5.8 (converging) |
| Speed (after JIT warmup) | ~0.12 sec/step |
| Checkpoint dir | `/tmp/simply_pretrain_imdb_15m/` |

---

## Summary of fixes

| Bug | Symptom | Fix |
|---|---|---|
| TFDS ARRAY_RECORD | `FileFormat` mismatch on dataset load | Delete `~/tensorflow_datasets/imdb_reviews/` |
| cuda13 driver mismatch | `cudaErrorInsufficientDriver` | Use `jax[cuda12]` instead of `jax[cuda13]` |
| cuda12 too new (12.9) | `cudaErrorInsufficientDriver` | `uv pip install "nvidia-cuda-runtime-cu12==12.5.82"` |
| Dual CUDA plugins | `PJRT_Api already exists` | Uninstall all `cuda13` packages |
| `uv run` re-syncs | Manual package pins reverted | Use `uv run --no-sync` |
| GPU OOM (logits tensor) | `bfc_allocator` OOM during forward pass | Reduce `batch_size` to 8 and `seq_len` to 256 |

---

<a id="apr16-step-5"></a>
## Step 5: Monitoring with TensorBoard

Each experiment writes TensorBoard logs to `<experiment_dir>/tb_log/`. Logged metrics include training loss, learning rate, gradient norm, and per-layer weight/gradient RMS stats.

### Start TensorBoard (on the remote machine)

Run in a detached screen session so it persists after disconnecting:

```bash
screen -dmS tensorboard bash -c '
  uv run --no-sync tensorboard --logdir /tmp/ --port 6006 --bind_all \
    > /tmp/tensorboard.log 2>&1
'
```

Using `/tmp/` as the root lets TensorBoard show all runs side-by-side (e.g., `simply_local_test_1` vs `simply_local_test_gpu`) for easy comparison.

### Access from a local machine (SSH port forwarding)

Since the VM is remote, forward port 6006 over SSH before opening the browser:

```bash
# Run this on your local machine
ssh -L 6006:localhost:6006 <your-username>@<vm-ip-or-hostname>
```

Then open **http://localhost:6006** in your browser.

### Stop TensorBoard

```bash
screen -S tensorboard -X quit
```

---

## Next steps

- Wait for `flops2e16_tfm15m_imdb` 500-step run to complete and record final loss.
- Run full pretraining on C4 with `flops2e16_tfm15m_c4_l2048` (requires C4 download):
  ```bash
  uv run --no-sync python setup/setup_assets.py --datasets-only
  ```
