# PhasePix: Quantitative SEM Microstructure Analyzer

PhasePix is a high-performance Streamlit application designed for automated deep-learning segmentation and morphological phase analysis of Scanning Electron Microscopy (SEM) micrographs (e.g., distinguishing air bubble pores from fat globules in food matrices and material science).

---

## Key Features

- **Automatic Hardware Acceleration (v1.7)**: The fastest backend is selected at startup and passed straight to Cellpose and SAM — **NVIDIA CUDA** (`torch.cuda.is_available()`, with the exact GPU model shown in the sidebar badge via `torch.cuda.get_device_name(0)`), then **Apple Silicon Metal** (`torch.backends.mps.is_available()`), then a clean **CPU** fallback where every feature keeps working.
- **Windows CPU Threading Hotfix (v1.7.1)**: MKL / OpenMP oversubscription could freeze the CPU backend for 10+ minutes on small images (e.g. 700×700) on Windows. `cpu_thread_guard.py` now exports `OMP_NUM_THREADS` / `MKL_NUM_THREADS` (4 threads by default) before `numpy` and `torch` are imported — from both `app.py` and `cellpose_model.py` — and finishes with `torch.set_num_threads(4)` so PyTorch's own intra-op pool is capped too. A lower value exported through `PHASEPIX_CPU_THREADS` (or `OMP_NUM_THREADS`) is honoured, while larger values are clamped back to the safe ceiling.
- **Precise Scale Bar Detection with Vertical Ticks (`scale_detector.py`)**: Reliably locates the SEM calibration scale bar in the upper half of the info banner (excluding full-width table dividers and lower sample metadata). Measures exact pixel span between vertical end-ticks and visualizes it with a crisp green centerline and red end markers.
- **Additive Manual Correction Workflow & Tab Persistence (`app.py`, `analyzer.py`)**: Manual polygons and pores drawn in the **🖌️ Manual Correction** tab are accumulated additively into `saved_manual_items_{image_name}`. Switching tabs never erases manual annotations. Previously saved manual objects are pre-rendered directly onto the canvas background with their unique IDs so users can seamlessly sketch new structures.
- **Smart Snap: Robust Boundary-Band Refinement (`analyzer.py`)**: Sketched rough contours can be automatically refined using **✨ Auto-Snap Rough Contour to Particle Edges**. Specifically engineered for foam and emulsion SEM micrographs containing fine fat droplets inside large pores: refinement operates strictly within a narrow boundary band using Sobel gradient + Watershed while preserving the internal core 100% solid, backed by a ±45% safety area sanity check.
- **Multi-Variant Cellpose Training Dataset Export (`.tif`)**: One-click download button (`🎓 Export Image + Combined Masks (.tif) for Cellpose Training`) packages the original micrograph and 3 specialized 16-bit TIFF mask variants into a ZIP archive: `_masks_combined_fat_on_top.tif` (retaining fine fat inclusions over large pores), `_masks_air_only.tif` (pores only), and `_masks_fat_only.tif` (fat globules only).
- **Strict Z-Order Layering & Pure Color Blending (`visualizer.py`)**: Individual opacities are clipped strictly to user slider values (0.0 to 1.0) and blended directly against base RGB micrograph values, eliminating muddy color contamination when fat droplets overlay air pores in `Fat over Air` mode.
- **Scientifically Rigorous Granulometry Reporting (`analyzer.py`)**: Computes Sauter ($D[3,2]$) and De Brouckere ($D[4,3]$) mean diameters strictly for individual microstructural phases (Fat Particles, Air Bubbles), omitting scientifically invalid mixed global averages.
- **Automated Cellpose Particle Diameter (`app.py`)**: Calculates the recommended Cellpose rescaling diameter directly from your calibrated physical size range ($\mu\text{m}$) and image scale, while providing a clear toggle between Auto, dynamic Cellpose internal estimation (`0`), and manual pixel override.
- **Instant Streamlit Caching**: Uses `@st.cache_resource` for the neural network model and `@st.cache_data` for segmentation masks, ensuring that adjusting brightness thresholds or spatial scales updates the dashboard instantaneously without re-running model inference.
- **High-Speed Engine (v1.5)**: The whole deterministic analysis is memoised, not just the neural network: image decoding (`decode_image_bytes`), SEM scale-bar probing (`cached_detect_scale_bar`), `regionprops` geometry (`cached_extract_metrics` — convex-hull `solidity`/`perimeter` for thousands of particles) and both matryoshka de-duplication passes (`cached_prune_nested_auto_objects`, `cached_remove_nested_matryoshka_masks`). Mask filtering uses an O(1) NumPy look-up table instead of `np.isin`, the Rule-2 centroid test is a single vectorised gather instead of a row-wise `DataFrame.apply`, PNG exports use `compress_level=1`, and `segment_air_bubbles_auto` runs Pass 1 at a ~42 px target scale (`42.0 / air_target_px`, clamped to `>= 0.35`) with exact `cv2.INTER_NEAREST` mask restoration whenever `air_target_px > 55 px`, while the dark-pore fallback stays at native resolution. Interfacial contact-angle metrics are skipped entirely when no `adsorbed_fat` object exists.
- **Precise Quantitative Metrics**: Calculates particle area ($\mu\text{m}^2$), equivalent diameter ($\mu\text{m}$) using $\sqrt{\frac{4 \cdot \text{Area}}{\pi}}$, mean pixel intensity, spatial centroids (`centroid_x`, `centroid_y`), and circularity shape factor via `skimage.measure.regionprops`.
- **Morphological Shape Filtering (Circularity)**: Interactive slider `Min Circularity` ($0.0 - 1.0$) to filter out irregular artifacts and sharp-cornered non-spherical structures using $\frac{4 \pi \cdot \text{Area}}{\text{Perimeter}^2}$ with instant zeroing out of rejected objects from masks and overlays.
- **Automated Scale Bar Detection**: Automatically detects horizontal SEM calibration scale bars in the status banner prior to any cropping.
- **Size-Based Phase Classification**: Replaces simple intensity thresholding with dedicated micrometer size boundaries (`Min Air Bubble Size (µm)`, `Max Air Bubble Size (µm)`, `Min Fat Particle Size (µm)`, `Max Fat Particle Size (µm)`). Features automatic overlap conflict resolution and customizable out-of-range particle handling (`Unclassified`, `Fat Particles`, `Air Bubbles`).
- **Advanced Granulometry Metrics ($D[3,2], D[4,3]$)**: Calculates surface area moment mean (Sauter mean diameter $D[3,2] = \sum d^3 / \sum d^2$), volume moment mean (De Brouckere mean diameter $D[4,3] = \sum d^4 / \sum d^3$), cumulative distribution percentiles ($D_{10}$, $D_{50}$ median, $D_{90}$), and Polydispersity Span per phase and overall.
- **Dedicated Granulometry Export**: Interactive results table in the **Metrics** tab with an instant one-click download button (`📥 Download Results Table (CSV)`) for food science and colloidal dispersion publications.
- **Interactive Table Editing & Manual Reclassification (`st.data_editor`)**: Researchers can check off artifact particles via the `Excluded` boolean checkbox column AND manually reclassify particles using the editable `phase_class` dropdown (`st.column_config.SelectboxColumn` with `['air_bubbles', 'fat_particles', 'Unclassified']`). Clicking `Apply Changes` synchronizes modifications to `st.session_state`, immediately recoloring the particle on the overlay, recalculating phase KPI counts, and updating granulometry metrics in real time (with a `Reset Modifications` option).
- **Lime Green Unclassified Masks**: Unclassified particles (those not matching the Min/Max size boundaries) are highlighted on the segmentation overlay in distinct semi-transparent Lime Green (`RGB 50, 205, 50` / `#32cd32`), making them instantly recognizable for visual inspection and manual reclassification.
- **Integrated Visual Ruler (Zero-Latency HTML5 Caliper)**: The **📊 Interactive Dashboard** column 1 exposes a third `Display Mode` — **📏 Interactive Ruler** — which renders the micrograph (or the segmentation overlay, switchable via *Ruler Base Image*) inside a self-contained HTML5 `<canvas>` component (`st.components.v1.html`), so a click-and-drag measurement is calculated **entirely in the browser** — no server rerun, no websocket round-trip, identical behaviour on Windows, macOS and Linux. The component is styled to match Streamlit's own cards (white background, `1px solid #e6e9ef` borders, 8 px radius, `Source Sans Pro` typography, `#31333F` text), the canvas always spans 100 % of the column width (no dark letterboxing and identical geometry to the *Segmentation Overlay* / *Raw Micrograph* views), and the background micrograph is embedded as a fast `quality=92` JPEG data URL. Euclidean line length $\sqrt{\Delta x^2 + \Delta y^2}$ is mapped from CSS pixels → embedded bitmap pixels → **native micrograph pixels** using the exact `native/bitmap` ratio, then multiplied by the active calibration to report micrometres ($\mu\text{m}$). A compact top bar shows **Measured Size** (µm and px), **Suggested Min (80 %)** and **Suggested Max (120 %)** next to the *Line Color* / *Line Width* pickers and the *↩️ Undo* / *🗑️ Clear* buttons, while a live floating badge follows the drag. The measurement history table appears only after the first line is drawn (keeping the dashboard compact), and the suggested Min/Max values can be copied straight into **Section 4** of the sidebar while the Phase Classifications table stays visible on the right.
- **Manual Particle ID Exclusion**: Allows researchers to exclude specific particle IDs (e.g. `4, 12, 45`) via the sidebar text input.
- **Flexible Phase Layering & Z-Order (Layering)**: Dedicated visualization controls to tune individual opacities (`Air Bubbles Opacity` and `Fat Particles Opacity`, default 0.3), toggle class visibility (`Show Air Bubbles` and `Show Fat Particles`), and control layering priority (`Fat over Air` vs. `Air over Fat`) so that overlying fat globule crusts do not obstruct air pore analysis on whipped emulsions.
- **Sample Type Modes & Dynamic Sidebar UI**: Supports 3 distinct sample types via `st.radio('Sample Type', options=['Emulsion', 'Whipped Emulsion', 'Foam'])`:
  - **`Emulsion`**: Single-phase fat emulsion. Hardcodes all detected objects to `fat_particles` (orange overlay), completely hides air bubble size/opacity controls in the sidebar, renders only fat particles on the KDE distribution, and exposes a single dedicated `📥 Download Fat Particles (CSV)` button.
  - **`Whipped Emulsion`**: Multi-phase colloidal system. Uses size-based classification (`Min/Max Air Bubble Size` and `Min/Max Fat Particle Size`) with overlap priority and out-of-range fallback resolution, displaying full multi-phase sidebar controls.
  - **`Foam`**: Single-phase gaseous foam. Hardcodes all detected objects to `air_bubbles` (sky blue overlay), hides all fat particle size/opacity controls, plots only the air bubble KDE curve, and displays a single dedicated `📥 Download Air Bubbles (CSV)` button.
- **Reproducibility & Data Export**: Export full tabular metrics as CSV, publication-ready KDE distributions as PNG, session manifest JSON, and full-resolution segmented colored and numbered overlays (`segmented_overlay.png`).
- **Publication-Ready Cropped Export (v1.6)**: Whenever a SEM info banner is excluded, both the **📊 Interactive Dashboard** (`✂️ Download Cropped Overlay — No Footer (PNG)` / `✂️ Download Cropped Raw — No Footer (PNG)`) and the **💾 Reproducibility & Export** tab (`✂️ Export Cropped Overlay (No Footer, PNG)`) offer a footer-free PNG (`segmented_overlay_cropped_*.png` / `raw_micrograph_cropped_*.png`) sliced with `overlay[:-effective_footer, :]`. The exported figure contains only the clean microstructure region — the bottom SEM databar is gone while a burned-in scale bar drawn inside the analysis region (e.g. `10 um`) is preserved — ready for direct journal submission.
- **Parameter Preset Manager with Deletion (v1.6)**: Sidebar **Section 5 → 📂 Parameter Presets (Save / Load Recipes)** stores complete Section 4 & 5 recipes per modality (SEM / Light) in `parameter_presets.json`. Recipes can be saved (`💾 Save Current Settings as Preset`), applied (`📂 Load Preset`) and permanently removed (`🗑️ Delete Preset`) side-by-side. Because a modality written to `parameter_presets.json` is authoritative, deleted presets stay deleted across restarts, while `🔄 Restore Built-in Presets` re-seeds the laboratory recipes without touching custom ones.

---

## Directory Structure

```text
PhasePix/
├── app.py                            # Streamlit web application & UI dashboard
├── analyzer.py                       # Morphological, photometric & D[3,2]/D[4,3] feature extraction
├── cellpose_model.py                 # Cellpose inference wrapper with MPS / CUDA / CPU handling
├── scale_detector.py                 # Automated SEM scale bar detection
├── visualizer.py                     # Size-based phase sorting, overlay rendering & KDE visualization
├── requirements.txt                  # Clean runtime dependencies
├── light_calibration_presets.json    # Light microscopy objective calibration presets
├── parameter_presets.json            # Built-in SEM / Light parameter presets
├── Icon.ico                          # Application icon
├── phasepix.bat                      # Native Windows virtual environment launcher
├── phasepix_run.command              # Native macOS Finder launcher script
├── build_portable_windows.bat        # One-time builder: zero-install dist\PhasePix_Portable folder
├── portable\                         # Launcher templates copied into the portable distribution
│   ├── PhasePix_Start.bat            # End-user auto-updating launcher
│   ├── Create_Desktop_Shortcut.bat   # Desktop shortcut helper with the custom .ico
│   └── PORTABLE_README.txt           # Instructions shipped inside the portable folder
├── images\                           # Micrographs - empty in the public repo (.gitkeep only)
│   ├── sem\                          # SEM micrographs
│   └── light\                        # Light microscopy images
├── models_cache\                     # Downloaded AI weights - local only (.gitkeep only in git)
├── SOP_SEM.docx                      # Standard operating procedure - SEM workflow
├── SOP_light.docx                    # Standard operating procedure - light microscopy workflow
├── Methodology_SEM.docx              # Segmentation methodology - SEM
├── Methodology_light.docx            # Segmentation methodology - light microscopy
├── METHODOLOGY_AI Parameter_Advisor .docx   # AI Parameter Advisor methodology
├── .gitignore                        # Keeps micrographs, weights and caches out of Git
└── README.md                         # Project documentation & run guide
```

---

## Running PhasePix

The public repository ships **source code, presets and documentation only** — no laboratory micrographs are published. Put your own data into `images/sem/` or `images/light/` (both folders are preserved in Git by a `.gitkeep` placeholder), or upload files directly from the sidebar. Manual mask JSONs written by the *Manual Correction* tab land in the same folders and are git-ignored as well.

### Windows — zero-install portable folder (recommended)

1. Build the distribution once with **`build_portable_windows.bat`** on a Windows build PC.
2. Hand the resulting `PhasePix_Portable` folder to any Windows 10/11 x64 PC (USB stick, network share, zip).
3. Double-click **`PhasePix_Start.bat`** — the dashboard opens at `http://localhost:8501`.
4. Drop micrographs into `app\images\sem` / `app\images\light`, or double-click **`Open_Images_Folder.bat`** in the root of the portable folder to open that directory directly.

### Windows / macOS / Linux — native Python

Install Python 3.11, then `pip install -r requirements.txt` and launch with `phasepix.bat` (Windows) or `phasepix_run.command` (macOS Finder) — see [Native Installation & Setup](#native-installation--setup-alternative).

### Local data folders

- **`images/`** — your micrographs (`images/sem`, `images/light`) plus the manual mask JSONs. Never committed.
- **`models_cache/`** — Segment Anything and Cellpose weights downloaded on first run. Never committed.
- **`parameter_presets.json` / `light_calibration_presets.json`** — recipes created in the sidebar persist here.

---

## 📦 Building & Distributing PhasePix Portable (Windows Zero-Install)

`PhasePix_Portable` is a self-contained folder that runs PhasePix on **any Windows 10/11 x64 PC without Python, Git, CUDA or administrator rights** — yet it still receives automatic code updates from GitHub on every launch.

### 1. Build the folder once (on one Windows PC)

```bat
build_portable_windows.bat
```

The builder produces `dist\PhasePix_Portable\` with:

| Folder / file | What it is |
| --- | --- |
| `python\` | Portable **CPython 3.11.9** (`indygreg`/`astral-sh` `python-build-standalone`, `x86_64-pc-windows-msvc-shared-install_only`) extracted with the Windows built-in `tar`. Full standard library, headers and a working `pip`. |
| `git\` | Portable **MinGit 2.47.0** (`cmd\git.exe`) — no Git installation on the target PC. |
| `app\` | A clean `git clone` of this repository **including `.git`**, so the launcher can self-update. The `origin` remote is force-set to the public GitHub HTTPS URL, never to the local folder the builder ran from. |
| `app\images\sem\` / `app\images\light\` | Created empty — no laboratory micrograph is ever shipped inside the distribution. |
| `app\models_cache\` | Pre-downloaded `sam_vit_b_01ec64.pth` and Cellpose weights (`cyto2` / Cellpose-4 `cpsam`), so colleagues do not wait on first run. |
| `Icon.ico` | Copied from the repository root and used by the Desktop shortcut. |
| `PhasePix_Start.bat` | The end-user launcher (copied from `portable\`). |
| `Create_Desktop_Shortcut.bat` | Creates a Desktop `.lnk` pointing at the launcher, with `$Shortcut.IconLocation` set to `Icon.ico`. |
| `Open_Images_Folder.bat` | Opens `app\images` in Windows Explorer (`explorer.exe "%~dp0app\images"`) straight from the root of `PhasePix_Portable`. |
| `PORTABLE_README.txt` | Plain-language instructions for colleagues. |

Build steps executed by the script: download + extract portable Python → download + extract MinGit → resolve the public GitHub URL from the builder's `origin` remote → `git clone` the repo → pin `origin` inside `app\` to that URL → `pip install --upgrade pip setuptools wheel` → `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` → `pip install --prefer-binary --upgrade-strategy only-if-needed -r app\requirements.txt` (torch / torchvision are unpinned in `requirements.txt`, so the `+cu121` wheels installed in the previous step are kept and never swapped for PyPI CPU wheels; the builder then re-checks `torch.version.cuda` and reinstalls the CUDA 12.1 wheels automatically if a CPU build slipped in) → copy `Icon.ico` → pre-download AI checkpoints → install launchers, create the empty `images` folders and generate `Open_Images_Folder.bat`. Expect **20–40 minutes and ~6 GB** of downloads; every step is idempotent, so re-running the builder resumes where it stopped.

### 2. Hand it to colleagues

Copy the whole `dist\PhasePix_Portable` folder (USB stick, network share, zip). On the target PC:

1. Double-click **`PhasePix_Start.bat`** → the dashboard opens at `http://localhost:8501`.
2. Optional: run **`Create_Desktop_Shortcut.bat`** once to get a PhasePix icon on the Desktop.
3. Put micrographs into `app\images\sem` / `app\images\light`, or double-click **`Open_Images_Folder.bat`** to open that directory in Explorer.

### 3. Automatic updates on every launch

`PhasePix_Start.bat` performs a **smart, non-destructive** update in three numbered phases:

* **`[1/3] Checking for PhasePix updates...`** — backs up `parameter_presets.json` and `light_calibration_presets.json` to `.tmp` files, runs `git fetch --all`, then `git reset --hard origin/main` (falling back to `origin/master`) and restores the user's preset files. Lab recipes are therefore never lost. Without a network connection it prints `[INFO] Offline mode - starting current local version.` and continues with the local copy.
* **`[2/3] Verifying Python dependencies...`** — re-runs `pip install -r requirements.txt` only when `.portable_req_stamp` shows that `requirements.txt` changed in the fetched update.
* **`[3/3] Starting PhasePix...`** — launches `python -m streamlit run app.py --server.port=8501 --browser.gatherUsageStats=false`.

The launcher also exports the portable environment: `PATH` (bundled Python + Git first), `CELLPOSE_LOCAL_MODELS_PATH` and `TORCH_HOME` (both pointing at `app\models_cache`, so **no weights are ever written to `C:\Users\<user>\.cellpose`**), `STREAMLIT_SERVER_HEADLESS=false` and `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`.

> `cellpose_model.py` resolves the checkpoint folder from `CELLPOSE_LOCAL_MODELS_PATH` (then `PHASEPIX_MODELS_DIR`, then `models_cache/`) and exports it before importing Cellpose, whose `MODEL_DIR` is only read at import time. The same folder holds the SAM checkpoints (`ensure_sam_checkpoint`) and the Cellpose weights (`ensure_cellpose_weights`) — the builder calls both to pre-seed the cache.

`dist/` is git-ignored; only the builder, the `portable/` templates and the documentation are committed.

---

## Native Installation & Setup (Alternative)

### Windows
1. Install Python 3.11:
   ```cmd
   winget install -e --id Python.Python.3.11
   ```
2. Create & activate environment:
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate.bat
   ```
3. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```
   *(For NVIDIA GPU acceleration: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` prior to requirements)*
4. Run:
   ```cmd
   phasepix.bat
   ```

### macOS / Linux
1. **Clone or navigate to the project directory**:
   ```bash
   cd /path/to/PhasePix
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch**:
   - Double-click `phasepix_run.command` in macOS Finder, or:
   ```bash
   streamlit run app.py
   ```

---

## Running the Application

### Option 1: Quick Launch
- **Windows**: Double-click the **`phasepix.bat`** file.
- **macOS**: Double-click the **`phasepix_run.command`** file in Finder, or execute:
  ```bash
  ./phasepix_run.command
  ```

### Option 2: Manual Terminal Launch
- **Windows**:
  ```cmd
  .venv\Scripts\activate.bat
  streamlit run app.py
  ```
- **macOS / Linux**:
  ```bash
  source .venv/bin/activate
  streamlit run app.py
  ```

The application will automatically open in your default browser at `http://localhost:8501`.

### Usage Workflow:
1. **Upload Micrograph**: Upload your SEM image (`.jpg`, `.png`, or `.tif`) or select a pre-loaded micrograph from the local `images/` folder.
2. **Set Spatial Scale**: Enter the nominal scale bar size in micrometers ($\mu\text{m}$) or use the auto-detected scale bar pixel length.
3. **Configure Particle Size Ranges**: Specify the min and max diameter ranges for Air Bubbles (e.g. $10 - 100\,\mu\text{m}$) and Fat Particles (e.g. $1 - 5\,\mu\text{m}$), along with conflict resolution strategies.
4. **Inspect Granulometry & Export**: Switch to the **Metrics** tab to inspect $D[3,2]$, $D[4,3]$, $D_{10}$, $D_{50}$, $D_{90}$, and Span, and export the table via **"📥 Download Results Table (CSV)"**.
