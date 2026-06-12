# Experiment Log

## Date

2026-04-16

## Goal

Test the Simply codebase setup end-to-end:
1. Run `lm_test` on CPU to verify the baseline setup.
2. Enable GPU support and run the same test on a Tesla T4.

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

## Step 1: Install dependencies

```bash
uv sync --extra tfds --extra gpu
```

- `--extra tfds` installs TensorFlow and `tensorflow-datasets` (needed for the IMDB dataset used by `lm_test`).
- `--extra gpu` installs `jax[cuda13]` (CUDA 13 pip packages) — this later caused a driver mismatch (see Bug 2).

---

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

## Summary of fixes

| Bug | Symptom | Fix |
|---|---|---|
| TFDS ARRAY_RECORD | `FileFormat` mismatch on dataset load | Delete `~/tensorflow_datasets/imdb_reviews/` |
| cuda13 driver mismatch | `cudaErrorInsufficientDriver` | Use `jax[cuda12]` instead of `jax[cuda13]` |
| cuda12 too new (12.9) | `cudaErrorInsufficientDriver` | `uv pip install "nvidia-cuda-runtime-cu12==12.5.82"` |
| Dual CUDA plugins | `PJRT_Api already exists` | Uninstall all `cuda13` packages |
| `uv run` re-syncs | Manual package pins reverted | Use `uv run --no-sync` |

---

## Step 4: Monitoring with TensorBoard

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

- Run pretraining from scratch with `flops2e16_tfm15m_c4_l2048` (15M params, C4 dataset).
- Download C4 dataset first: `uv run --no-sync python setup/setup_assets.py --datasets-only`
