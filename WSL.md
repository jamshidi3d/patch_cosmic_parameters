# Running the pipeline locally through WSL (with RAM management)

`pymaster` / NaMaster is effectively Linux-only to install, so on this Windows
box the practical way to run the notebooks natively is **WSL2 + Ubuntu**. This
guide covers the WSL setup, capping and cushioning memory so the RAM-heavy
steps don't hard-kill the kernel, and the pipeline-side knobs that cut peak
usage.

Run **`patch_cosmology_fit.ipynb`** here — the inputs are already in `input/`.
(`patch_cosmology_fit_colab.ipynb` is only for Colab: its extra cells install
conda, clone the repo, and download the FITS files.)

---

## 1. Install WSL2 + Ubuntu

In an **admin PowerShell**:

```powershell
wsl --install -d Ubuntu
```

Reboot if prompted, launch **Ubuntu** from the Start menu, create the UNIX
user. Confirm you're on WSL **2**:

```powershell
wsl -l -v      # VERSION column must say 2
```

---

## 2. Cap and cushion RAM with `.wslconfig`

By default WSL2 will grow to ~50–80% of total Windows RAM and has only a small
swap file — so a memory spike in Section 2/4 gets **OOM-killed** (the repo
README calls this out). Give it an explicit ceiling and a large swap so a
spike pages instead of dying.

Create `C:\Users\<you>\.wslconfig` (Windows side):

```ini
[wsl2]
# Leave 4-8 GB for Windows itself. e.g. on a 32 GB machine:
memory=24GB
# 1.5-2x `memory` — this is the safety net for covariance spikes:
swap=40GB
swapFile=C:\\wsl-swap.vhdx
processors=8
# Hand freed RAM back to Windows instead of hoarding it:
pageReporting=true

[experimental]
autoMemoryReclaim=gradual
```

Apply it:

```powershell
wsl --shutdown
```

then reopen Ubuntu. Check inside WSL:

```bash
free -h            # "Mem" ~= memory=, "Swap" ~= swap=
nproc              # == processors=
```

---

## 3. Build the environment

### Option A — Miniforge + mamba (recommended, matches what installs cleanly)

```bash
cd ~
wget -qO Miniforge3.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3.sh -b -p ~/miniforge3
~/miniforge3/bin/conda init bash && exec bash

mamba create -n cosmo -c conda-forge python=3.12 \
  namaster healpy camb iminuit matplotlib jupyterlab nbclient ipykernel psutil
mamba activate cosmo
python -m ipykernel install --user --name cosmo --display-name "cosmo (WSL)"
```

`namaster` provides `pymaster`. On a real Linux conda env there is **no**
OpenSSL/libcurl hack needed (that was a Colab-only clash).

### Option B — apt + pip (source build)

```bash
sudo apt update
sudo apt install -y build-essential gfortran pkg-config \
  libgsl-dev libfftw3-dev libcfitsio-dev
python -m venv ~/venvs/cosmo && source ~/venvs/cosmo/bin/activate
pip install healpy camb iminuit matplotlib jupyterlab nbclient ipykernel psutil
pip install pymaster            # compiles from source; usually fine on real Ubuntu
```

Verify either way:

```bash
python -c "import pymaster, healpy, camb, iminuit; print('ok', pymaster.__version__)"
```

---

## 4. Point WSL at the project

The repo already lives on the Windows drive, reachable as:

```bash
cd /mnt/f/Science/PhD/Projects/patch_cosmic_parameters
```

**Simplest:** run in place. `output/` then lands directly in `F:\...\output\`,
no copy-back.

**Faster I/O (optional):** the `/mnt/f` bridge is slow for the ~1.6 GB FITS
reads. Put code + data on the Linux filesystem instead:

```bash
mkdir -p ~/cosmo && cd ~/cosmo
git clone https://github.com/jamshidi3d/patch_cosmic_parameters.git
cd patch_cosmic_parameters
cp -r /mnt/f/Science/PhD/Projects/patch_cosmic_parameters/input .   # ~1.6 GB, one-time
# copy results back when done:  cp -r output /mnt/f/Science/PhD/Projects/patch_cosmic_parameters/
```

Don't leave the working copy under `/mnt/c` or `/mnt/f` if you plan repeated
runs — Linux-fs I/O is many times faster.

---

## 5. Watch memory while it runs

- **Live:** `watch -n 2 free -h` in a second terminal, or `htop`
  (`sudo apt install htop`) sorted by `%MEM` (`F6` → `PERCENT_MEM`).
- **Peak of a headless run:** `/usr/bin/time -v python script.py` →
  `Maximum resident set size (kbytes)`.
- **In-notebook:** add a cell you re-run between sections —

  ```python
  import psutil, os
  print(f"RSS {psutil.Process().memory_info().rss/1e9:.1f} GB  "
        f"| avail {psutil.virtual_memory().available/1e9:.1f} GB")
  ```

- **After a crash:** `dmesg | grep -i -E "oom|killed process"` — if you see
  `Out of memory: Killed process ... python`, it was the OOM killer; raise
  `swap` in `.wslconfig` and/or apply Section 6.
- **Orphan kernels** (silently eat CPU/RAM after a timeout):
  `ps aux | grep ipykernel` → `pkill -f ipykernel`.

---

## 6. Cut the pipeline's peak memory

Anchored to cells in `patch_cosmology_fit.ipynb`:

1. **Biggest win — Section 4 `NSIDE_PATCH = 2048` → `512`.** In the cell that
   starts `NSIDE_PATCH = 2048`, set it to `512`. The section's own markdown
   says to; every downstream call already `ud_grade`s to `NSIDE_PATCH`, so
   nothing else changes. NaMaster's covariance workspace does SHTs of the full
   pixel maps, so this cuts per-patch covariance memory roughly **16×** — it's
   the difference between the 12-patch loop fitting and OOM-killing the kernel.

2. **Run the two scopes in separate kernels.** Section 4 only needs Section 1
   (data load). Do the full-sky path (Sections 2, 3, 5) in one run, let it
   save to `output/`, **restart the kernel**, then run Section 1 + Section 4.
   Neither ever holds the other's arrays.

3. **Drop the full-res maps once Section 4 has downgraded them.** Right after
   the `map_*_lo = hp.ud_grade(...)` block in Section 4:

   ```python
   del map_hm1_t, map_hm2_t, map_hm1_q, map_hm1_u, map_hm2_q, map_hm2_u
   import gc; gc.collect()
   ```

   frees ~2–2.5 GB of `nside=2048` arrays the patch loop doesn't use.

4. **Lower the multipole reach / widen bins** (README §4). Covariance size
   grows with `n_bins²`; reducing `LMAX_TT/TE/EE` or raising `bandpower_width`
   shrinks the covariance build, the dominant allocation in Section 2.

5. **Keep `compute_minos=False`** (already the default) — MINOS multiplies the
   number of expensive CAMB solves without reducing memory.

6. **Free NaMaster workspaces you're done with.** They hold C-side coupling
   matrices; `del workspace_tt, workspace_te, workspace_ee; gc.collect()` once
   the fit and residual plot are done.

---

## 7. Run it

### Interactive (JupyterLab in WSL, browser on Windows)

```bash
mamba activate cosmo          # or: source ~/venvs/cosmo/bin/activate
cd /mnt/f/Science/PhD/Projects/patch_cosmic_parameters
jupyter lab --no-browser --port 8888
```

Open the printed `http://127.0.0.1:8888/lab?token=...` URL in a Windows
browser. Or use **VS Code → "WSL: Connect to WSL"** and open the folder there.

### Headless (no orphan kernels, hard timeout)

```bash
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=10800 \
  patch_cosmology_fit.ipynb --output executed.ipynb
```

To run only a prefix of the notebook (e.g. through Section 3, skipping the
long patch loop), use the `nbclient` cell-slicing snippet in the repo README
(§3, "To iterate faster while developing").

---

## 8. Results and cleanup

- Outputs: `output/` — already on `F:\` if you ran in place, else `cp -r
  output /mnt/f/.../`.
- **Reclaim RAM:** `wsl --shutdown` (PowerShell) frees whatever the WSL VM
  (`vmmemWSL` in Task Manager) is still holding.
- **Reclaim disk:** the swap/rootfs `.vhdx` files only grow. After
  `wsl --shutdown`:

  ```powershell
  Optimize-VHD -Path "C:\wsl-swap.vhdx" -Mode Full   # needs Hyper-V module
  ```

  or compact the distro's `ext4.vhdx` via `diskpart`'s `compact vdisk`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Kernel dies mid-cell; `dmesg` shows `Out of memory: Killed process … python` | Raise `swap` in `.wslconfig` (`wsl --shutdown` to apply); set `NSIDE_PATCH=512`; run Sections 2–3 and Section 4 in separate kernels (Section 6). |
| Windows becomes sluggish / `vmmemWSL` huge during the run | Set `memory=` in `.wslconfig` to leave headroom, `wsl --shutdown`, reopen. |
| RAM not released after the run finishes | `wsl --shutdown`; ensure `pageReporting=true` and `autoMemoryReclaim=gradual` are set. |
| `pip install pymaster` fails to build | Use the Miniforge/mamba path (Option A). |
| FITS load in Section 1 is very slow | Copy `input/` onto the Linux fs (`~/…`) instead of reading over `/mnt/f` (Section 4). |
| A rerun is much slower than an earlier identical one | Orphan kernel from a prior timeout: `ps aux | grep ipykernel`, then `pkill -f ipykernel`. |
| `free -h` shows the old limits after editing `.wslconfig` | The file must be at `C:\Users\<you>\.wslconfig` (not inside WSL), and you must `wsl --shutdown` (close all WSL terminals first). |
