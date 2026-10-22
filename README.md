# PhasePix: Quantitative SEM Microstructure Analyzer

PhasePix is a high-performance Streamlit application designed for automated deep-learning segmentation and morphological phase analysis of Scanning Electron Microscopy (SEM) micrographs (e.g., distinguishing air bubble pores from fat globules in food matrices and material science).

---

## Key Features

- **Apple Silicon Hardware Acceleration**: Built with native PyTorch Metal Performance Shaders (`mps`) support for macOS (M-series processors).
- **Precise Scale Bar Detection with Vertical Ticks (`scale_detector.py`)**: Reliably locates the SEM calibration scale bar in the upper half of the info banner (excluding full-width table dividers and lower sample metadata). Measures exact pixel span between vertical end-ticks and visualizes it with a crisp green centerline and red end markers.
- **Additive Manual Correction Workflow & Tab Persistence (`app.py`, `analyzer.py`)**: Manual polygons and pores drawn in the **🖌️ Manual Correction** tab are accumulated additively into `saved_manual_items_{image_name}`. Switching tabs never erases manual annotations. Previously saved manual objects are pre-rendered directly onto the canvas background with their unique IDs so users can seamlessly sketch new structures.
- **Smart Snap: Robust Boundary-Band Refinement (`analyzer.py`)**: Sketched rough contours can be automatically refined using **✨ Auto-Snap Rough Contour to Particle Edges**. Specifically engineered for foam and emulsion SEM micrographs containing fine fat droplets inside large pores: refinement operates strictly within a narrow boundary band using Sobel gradient + Watershed while preserving the internal core 100% solid, backed by a ±45% safety area sanity check.
- **Multi-Variant Cellpose Training Dataset Export (`.tif`)**: One-click download button (`🎓 Export Image + Combined Masks (.tif) for Cellpose Training`) packages the original micrograph and 3 specialized 16-bit TIFF mask variants into a ZIP archive: `_masks_combined_fat_on_top.tif` (retaining fine fat inclusions over large pores), `_masks_air_only.tif` (pores only), and `_masks_fat_only.tif` (fat globules only).
- **Strict Z-Order Layering & Pure Color Blending (`visualizer.py`)**: Individual opacities are clipped strictly to user slider values (0.0 to 1.0) and blended directly against base RGB micrograph values, eliminating muddy color contamination when fat droplets overlay air pores in `Fat over Air` mode.
- **Scientifically Rigorous Granulometry Reporting (`analyzer.py`)**: Computes Sauter ($D[3,2]$) and De Brouckere ($D[4,3]$) mean diameters strictly for individual microstructural phases (Fat Particles, Air Bubbles), omitting scientifically invalid mixed global averages.
- **Automated Cellpose Particle Diameter (`app.py`)**: Calculates the recommended Cellpose rescaling diameter directly from your calibrated physical size range ($\mu\text{m}$) and image scale, while providing a clear toggle between Auto, dynamic Cellpose internal estimation (`0`), and manual pixel override.
- **Instant Streamlit Caching**: Uses `@st.cache_resource` for the neural network model and `@st.cache_data` for segmentation masks, ensuring that adjusting brightness thresholds or spatial scales updates the dashboard instantaneously without re-running model inference.
- **Precise Quantitative Metrics**: Calculates particle area ($\mu\text{m}^2$), equivalent diameter ($\mu\text{m}$) using $\sqrt{\frac{4 \cdot \text{Area}}{\pi}}$, mean pixel intensity, spatial centroids (`centroid_x`, `centroid_y`), and circularity shape factor via `skimage.measure.regionprops`.
- **Morphological Shape Filtering (Circularity)**: Interactive slider `Min Circularity` ($0.0 - 1.0$) to filter out irregular artifacts and sharp-cornered non-spherical structures using $\frac{4 \pi \cdot \text{Area}}{\text{Perimeter}^2}$ with instant zeroing out of rejected objects from masks and overlays.
- **Automated Scale Bar Detection**: Automatically detects horizontal SEM calibration scale bars in the status banner prior to any cropping.
- **Size-Based Phase Classification**: Replaces simple intensity thresholding with dedicated micrometer size boundaries (`Min Air Bubble Size (µm)`, `Max Air Bubble Size (µm)`, `Min Fat Particle Size (µm)`, `Max Fat Particle Size (µm)`). Features automatic overlap conflict resolution and customizable out-of-range particle handling (`Unclassified`, `Fat Particles`, `Air Bubbles`).
- **Advanced Granulometry Metrics ($D[3,2], D[4,3]$)**: Calculates surface area moment mean (Sauter mean diameter $D[3,2] = \sum d^3 / \sum d^2$), volume moment mean (De Brouckere mean diameter $D[4,3] = \sum d^4 / \sum d^3$), cumulative distribution percentiles ($D_{10}$, $D_{50}$ median, $D_{90}$), and Polydispersity Span per phase and overall.
- **Dedicated Granulometry Export**: Interactive results table in the **Metrics** tab with an instant one-click download button (`📥 Download Results Table (CSV)`) for food science and colloidal dispersion publications.
- **Interactive Table Editing & Manual Reclassification (`st.data_editor`)**: Researchers can check off artifact particles via the `Excluded` boolean checkbox column AND manually reclassify particles using the editable `phase_class` dropdown (`st.column_config.SelectboxColumn` with `['air_bubbles', 'fat_particles', 'Unclassified']`). Clicking `Apply Changes` synchronizes modifications to `st.session_state`, immediately recoloring the particle on the overlay, recalculating phase KPI counts, and updating granulometry metrics in real time (with a `Reset Modifications` option).
- **Lime Green Unclassified Masks**: Unclassified particles (those not matching the Min/Max size boundaries) are highlighted on the segmentation overlay in distinct semi-transparent Lime Green (`RGB 50, 205, 50` / `#32cd32`), making them instantly recognizable for visual inspection and manual reclassification.
- **Visual Ruler (Interactive SEM Caliper)**: Powered by `streamlit-drawable-canvas`, researchers can switch to the **📏 Visual Ruler** tab to draw measurement lines directly across pores, bubbles, or fat globules on the micrograph. The tool automatically computes Euclidean line length in pixels $\sqrt{\Delta x^2 + \Delta y^2}$, multiplies by the active spatial calibration scale, and displays the exact physical dimension in micrometers ($\mu\text{m}$) via a prominent metric card and history table, providing immediate guidance for configuring Min/Max boundary thresholds.
- **Manual Particle ID Exclusion**: Allows researchers to exclude specific particle IDs (e.g. `4, 12, 45`) via the sidebar text input.
- **Flexible Phase Layering & Z-Order (Layering)**: Dedicated visualization controls to tune individual opacities (`Air Bubbles Opacity` and `Fat Particles Opacity`, default 0.3), toggle class visibility (`Show Air Bubbles` and `Show Fat Particles`), and control layering priority (`Fat over Air` vs. `Air over Fat`) so that overlying fat globule crusts do not obstruct air pore analysis on whipped emulsions.
- **Sample Type Modes & Dynamic Sidebar UI**: Supports 3 distinct sample types via `st.radio('Sample Type', options=['Emulsion', 'Whipped Emulsion', 'Foam'])`:
  - **`Emulsion`**: Single-phase fat emulsion. Hardcodes all detected objects to `fat_particles` (orange overlay), completely hides air bubble size/opacity controls in the sidebar, renders only fat particles on the KDE distribution, and exposes a single dedicated `📥 Download Fat Particles (CSV)` button.
  - **`Whipped Emulsion`**: Multi-phase colloidal system. Uses size-based classification (`Min/Max Air Bubble Size` and `Min/Max Fat Particle Size`) with overlap priority and out-of-range fallback resolution, displaying full multi-phase sidebar controls.
  - **`Foam`**: Single-phase gaseous foam. Hardcodes all detected objects to `air_bubbles` (sky blue overlay), hides all fat particle size/opacity controls, plots only the air bubble KDE curve, and displays a single dedicated `📥 Download Air Bubbles (CSV)` button.
- **Reproducibility & Data Export**: Export full tabular metrics as CSV, publication-ready KDE distributions as PNG, session manifest JSON, and full-resolution segmented colored and numbered overlays (`segmented_overlay.png`).

---

## Directory Structure

```text
PhasePix/
├── .agents/
│   ├── rules/
│   │   └── rules.md             # Project architectural rules and constraints
│   └── skills/
│       └── developing-with-streamlit/
├── images/                      # Put your SEM micrographs here (jpg, png, tif)
├── analyzer.py                  # Morphological, photometric & D[3,2]/D[4,3] feature extraction
├── app.py                       # Streamlit web application & UI dashboard
├── cellpose_model.py            # Cellpose inference wrapper with MPS / CPU handling
├── install_python.txt           # Windows setup commands and dependency guide
├── phasepix.bat                 # Quick launch script for Windows
├── phasepix_run.command         # Quick launch script for macOS Finder
├── Phasepix_code.txt            # Complete single-file code snapshot of the PhasePix codebase
├── requirements.txt             # Pinned project dependencies
├── scale_detector.py            # Automated SEM scale bar detection
├── visualizer.py                # Size-based phase sorting, overlay rendering & KDE visualization
└── README.md                    # Project documentation & run guide
```

---

## Installation & Setup

### Windows
Follow the quick commands provided in [install_python.txt](install_python.txt):
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

3. **Install pinned dependencies**:
   ```bash
   pip install -r requirements.txt
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
