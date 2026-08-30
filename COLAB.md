# Running the joint TT+TE+EE fit on Google Colab

`patch_cosmology_fit_colab.ipynb` is `patch_cosmology_fit.ipynb` copied
**verbatim** (every pipeline cell unchanged) with five setup cells prepended:
bootstrap conda, install the scientific stack, (optional) Drive cache, clone
this repo, download the Planck PR3 inputs. This guide walks through running it
end to end.

For what the pipeline actually measures, see [REPORT.md](REPORT.md); for the
`cosmo_patch` API, see [README.md](README.md).

---

## 1. Open the notebook in Colab

Any one of:

- **Direct link:**
  `https://colab.research.google.com/github/jamshidi3d/patch_cosmic_parameters/blob/main/patch_cosmology_fit_colab.ipynb`
- **Colab → File → Open notebook → GitHub tab**, search `jamshidi3d/patch_cosmic_parameters`, pick `patch_cosmology_fit_colab.ipynb`.
- **Upload:** download the `.ipynb` from the repo and *File → Upload notebook*.

If you change the setup cells and want the changes to persist, use
*File → Save a copy in GitHub* (or in Drive).

---

## 2. Set the runtime

*Runtime → Change runtime type*:

- **Hardware accelerator:** None (CPU). No GPU is used.
- **Runtime shape:** **High-RAM** if available. The Section 2 joint covariance
  and the Section 4 twelve-patch loop run at `nside=2048` and can OOM-kill the
  free ~13 GB runtime (noted in the repo README). High-RAM (~25 GB) is safe.

---

## 3. Run the setup cells (1–5)

### Cell 1 — bootstrap conda (`condacolab`)

`pymaster` (NaMaster) has **no PyPI wheels** and its source build fails on
Colab, so we install it from conda-forge. `condacolab.install()` swaps in a
conda Python and **restarts the kernel once**.

> The "Your session crashed / restarted" banner after this cell is **expected**,
> not an error.

After the restart, just run the notebook again from the top
(*Runtime → Run all*). Cell 1 is idempotent: it calls `condacolab.check()`,
sees conda is already active, prints `conda is ready`, and does nothing else.

### Cell 2 — install the scientific stack

```
mamba install -c conda-forge namaster healpy camb iminuit matplotlib
```

(~2–4 min; `namaster` provides the `pymaster` module.) The cell then patches
the dynamic loader **before** `import pymaster`:

- prepends `/usr/local/lib` (conda's libdir) to `LD_LIBRARY_PATH`
- `RTLD_GLOBAL`-preloads conda's `libcrypto.so.3` + `libssl.so.3`

Without this, `import pymaster` fails with
`libssl.so.3: version OPENSSL_3.2.0 not found` — NaMaster's `_nmtlib` links
conda's `libcurl`, which needs OpenSSL ≥ 3.2, but the loader would otherwise
bind Colab's older **system** `libssl`.

Success looks like:

```
numpy 2.x | healpy 1.20.x | camb 1.5.x | iminuit 2.x | pymaster 2.x
```

The `libmamba ... files were already present` warnings above it are harmless
(condacolab overlaying conda packages onto Colab's pip packages).

### Cell 3 — (optional) Drive cache for the big FITS files

Skip it to re-download each session. To keep the ~1.6 GB across sessions,
uncomment the block:

```python
from google.colab import drive
drive.mount("/content/drive")
import os
DATA_CACHE = "/content/drive/MyDrive/planck_pr3_inputs"
os.makedirs(DATA_CACHE, exist_ok=True)
```

Cell 5 then downloads into that folder and symlinks it into `input/`.

### Cell 4 — clone the repo

Clones `https://github.com/jamshidi3d/patch_cosmic_parameters` to
`/content/patch_cosmic_parameters`, adds it to `sys.path` (there is no
`setup.py`, so `pip install -e .` is not used), `chdir`s into it, and creates
`input/` and `output/`. It prints the resolved `cosmo_patch` path as a check.

Private-repo variant (token) is in a comment in the cell.

### Cell 5 — download the Planck PR3 inputs

Four files from the IRSA mirror of the Planck Legacy Archive (direct HTTP, no
login), ~1.6 GB total, into `input/`:

| File | Size | Used for |
|---|---|---|
| `COM_CMB_IQU-smica_2048_R3.00_hm1.fits` | ~576 MB | half-mission 1 map (I, Q, U) |
| `COM_CMB_IQU-smica_2048_R3.00_hm2.fits` | ~576 MB | half-mission 2 map (I, Q, U) |
| `COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits` | ~192 MB | temperature mask |
| `COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits` | ~192 MB | polarization mask |

Downloads go to `<name>.part` and are renamed only on success, so an
interrupted cell is safe to re-run. If IRSA is unavailable, swap in the PLA
URL `https://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=<FILENAME>`.

---

## 4. Run the pipeline (Sections 1–6.1)

From the `# Repo pipeline (verbatim ...)` divider onward the cells are
identical to `patch_cosmology_fit.ipynb`. Run them top to bottom. Approximate
wall times on a High-RAM CPU runtime:

| Section | What | Time |
|---|---|---|
| 1 | Load SMICA hm1/hm2 maps + masks, one CAMB fiducial solve | ~1–2 min |
| 2 | TT/TE/EE cross-spectra + 6-block joint Gaussian covariance (NaMaster) | ~10–15 min |
| 3 | Full-sky joint fit (iminuit MIGRAD+HESSE; MINOS off by default) | a few min |
| 4 | Twelve `nside=1` patches, each a full independent spectra+cov+fit | longest — tens of min |
| 5 | Residuals + best-fit overlay plot | seconds |
| 6 / 6.1 | Fixed-parameter example fit; `H0`–`omch2` contour | ~1–2 min |

The stale outputs saved in the committed notebook are from a previous local
run — a useful reference for expected chi² and best-fit values. Do
*Runtime → Restart and run all* (twice, for the cell-1 restart) for a clean
run.

---

## 5. Getting results out

Section 3–5 cells write to `/content/patch_cosmic_parameters/output/`
(`fullsky_params_*.txt`, `patched_params_*.{png,txt}`,
`patched_bandpowers_*.png`, `fullsky_spectrum_*.png`). This is **ephemeral** —
gone when the runtime recycles. To keep them:

- **Files pane** (left sidebar) → navigate to `output/` → right-click → Download.
- **Drive:** after `drive.mount(...)`, `!cp -r output /content/drive/MyDrive/patch_cosmo_output`.
- **Git:** `git add`/`commit`/`push` from a cell (needs a PAT in the remote URL).

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ERROR: Failed building wheel for pymaster`, then `ModuleNotFoundError: No module named 'healpy'` | `pip install pymaster` cannot build NaMaster on Colab (no wheels, C build fails); pip then aborts the whole batch. Use cells 1–2 as written (conda), don't `pip install pymaster`. |
| `import pymaster` → `libssl.so.3: version OPENSSL_3.2.0 not found` | Loader bound Colab's system `libssl` instead of conda's. Cell 2's `LD_LIBRARY_PATH` + `ctypes` preload fixes it — just re-run **cell 2** (no kernel restart needed). |
| "Session crashed / restarted" right after cell 1 | Expected — that's `condacolab.install()` swapping the Python. Re-run from the top. |
| `libmamba [...] files were already present in the environment` | Harmless. condacolab layering conda packages over Colab's pip packages in the shared `site-packages`. |
| Kernel dies during Section 2 or Section 4 | Out of memory. Use a **High-RAM** runtime. If still tight: lower `LMAX_*`, lower `NSIDE_PATCH`, or raise `BANDPOWER_WIDTH_PATCH` in the Section 4 config cell. |
| A download in cell 5 produced a tiny file / HTTP error | IRSA hiccup. Re-run cell 5 (it resumes the `.part`), or switch that file to the PLA `product-action` URL. |
| `cosmo_patch` import fails after cell 4 | Cell 4 didn't run or `chdir` was undone. Re-run cell 4; confirm it prints a `/content/patch_cosmic_parameters/cosmo_patch/__init__.py` path. |
| Fit is far slower than the timings above | An earlier timed-out cell can leave an orphan kernel process. *Runtime → Restart runtime* and rerun. |

---

## 7. One-shot recipe

1. Open the Colab link, set runtime to **High-RAM**.
2. *Runtime → Run all.* Wait for the kernel to restart after cell 1.
3. *Runtime → Run all* again. Cells 1–5 do setup + download (~10 min);
   the pipeline then runs (~30–60 min, Section 4 dominates).
4. Download `output/` from the Files pane.
