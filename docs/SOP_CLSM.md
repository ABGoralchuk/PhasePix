# 📋 Standard Operating Procedure — Quantitative CLSM Analysis of Colloidal Microspheres & Aggregates

> **Document:** `docs/SOP_CLSM.md` · **Downloadable Word twin:** `docs/SOP_CLSM.docx`
> **System:** PhasePix — Quantitative Microstructure & Colloid Analyzer
> **Modality:** 🔦 Confocal Laser Scanning Microscopy (CLSM)
> **Dispersed system:** 🟣 Colloidal Microspheres & Aggregates (Silica)
> **Revision:** 2.0.1 · **Validated on build:** v2.0.0 and later

| Field | Value |
| --- | --- |
| Procedure owner | Microscopy & Colloid Analytics Laboratory |
| Target analyte | Silica microspheres ($d \approx 1\text{–}20\,\mu\text{m}$) and their flocs |
| Calibration route | CLSM objective preset ($\mu\text{m}/\text{px}$) |
| Segmentation engine | Cellpose `cyto2` / `cyto3` + distance-transform watershed catchers |
| Deliverables | PSD, granulometry, cluster metrics ($N_{agg}$, $R_g$, $D_f$), SHA-256 manifest |
| Archive | CSV tables + PNG figures + JSON session manifest |

## 1. 🎯 Purpose and Scope

This SOP defines the complete operational protocol for quantifying **colloidal
silica microspheres, their interfacial shell and their aggregate networks** from
Confocal Laser Scanning Microscopy frames in PhasePix. It covers multi-channel
image preparation, objective calibration, AI segmentation setup,
human-in-the-loop quality control, cluster/flocculation analytics and
cryptographic archiving.

The procedure applies to:

- single-phase colloidal dispersions imaged as a **composite RGB TIFF**
  (green negative-contrast matrix + magenta fluorescent shell);
- single-plane acquisitions exported as **`c1`** (dark spheres in a bright
  matrix) or **`c0`** (bright fluorescent shells on a dark background);
- Pickering-stabilised silica networks where spheres touch and must be split
  into physical primary particles before granulometry.

**Out of scope:** air-bubble / foam analytics (those phases do not exist in a
single-phase CLSM dispersion, so the 🛡️ Interfacial Analytics tab is hidden
automatically in CLSM mode), and any measurement performed without a verified
spatial calibration.

## 2. 🧰 Before You Start — Prerequisites

| # | Requirement | Acceptance check |
| --- | --- | --- |
| 1 | PhasePix launched and reachable on `localhost` | Sidebar shows the build badge and a hardware backend |
| 2 | Objective metadata known | Magnification and pixel size recorded in the acquisition log |
| 3 | Composite RGB export available | `.tif` (preferred) or `.png` with both channels preserved |
| 4 | Frame free of saturation | No clipped pixels inside particle cores |
| 5 | Reference material | One field with a certified size standard per session |

## 3. 🖼️ Image Preparation Rules for CLSM Exports

1. **Export one composite RGB TIFF per field of view.** Do **not** pre-merge
   channels in image-editing software and do **not** apply gamma, sharpening or
   JPEG compression: PhasePix performs its own deterministic decomposition, and
   an upstream tone curve destroys the physical intensity relation between the
   shell and the matrix.
2. **Keep the native bit depth** (8-bit or 16-bit). 16-bit exports are normalised
   once, globally, so the channel balance is preserved.
3. **No stitching and no maximum-intensity collapse** of a z-stack unless the
   collapse is documented; a collapsed stack changes the apparent diameter of
   partially embedded spheres.
4. **One scale bar per session is enough**, because the calibration lives in the
   objective preset — but the preset must be re-verified whenever the objective,
   zoom or camera binning changes.
5. **Name files descriptively** (`CLSM_20x_SiO2_batch07_field03.tif`) so the
   exported session manifest is self-documenting.

## 4. 🪜 Step-by-Step Procedure

### 4.1 Select the imaging modality

In Sidebar **0. Imaging Modality**, choose **🔦 Confocal Fluorescence**. This
single switch:

- hides the air-bubble / adsorbed-fat controls that cannot act on a single-phase
  dispersion;
- enables the CLSM objective presets in Section 3;
- exposes the **🟣 Colloidal Microspheres & Aggregates (Silica)** dispersed phase;
- removes 🛡️ **Interfacial Analytics** from the navigation bar.

### 4.2 Load the micrograph

In Section **1. Image Input**, drop a file into the CLSM image folder shown in
the caption or use the uploader. Verify that the preview shows **both** the green
matrix and the magenta shell before continuing.

### 4.3 Verify spatial calibration (critical)

Open Section **3. Spatial Calibration** and select the objective preset that
matches the acquisition:

| Preset | $\mu\text{m}/\text{px}$ | Scale bar | Typical use |
| --- | --- | --- | --- |
| 10x CLSM Objective | 1.59800 | 50 µm / 31.29 px | Overview of large flocs |
| 20x CLSM Objective | 0.79900 | 50 µm / 62.58 px | Default quantitative work |
| 40x CLSM Objective | 0.39950 | 20 µm / 50.06 px | Single-cluster morphology |
| 63x CLSM Objective | 0.09900 | 10 µm / 101.01 px | Primary-particle counting |
| 100x CLSM Objective | 0.06200 | 10 µm / 161.29 px | Sub-micron shells |
| Custom CLSM Calibration | user | user | Non-standard optics |

The conversion used everywhere downstream is:

$$
s = \frac{L_{\text{bar}}\,[\mu\text{m}]}{N_{\text{px,bar}}}\quad[\mu\text{m}/\text{px}],
\qquad d\,[\mu\text{m}] = s \cdot d_{\text{px}}.
$$

**Purpose:** convert every pixel-domain measurement (mask area, equivalent
diameter, peak spacing) into SI-traceable micrometres before any statistic is
computed.

**Where:**

- $s$ — spatial sampling, the calibrated micrometres per pixel of the active
  objective preset;
- $L_{\text{bar}}$ — physical length engraved on the reference scale bar
  ($\mu$m), taken from the preset, **not** from OCR;
- $N_{\text{px,bar}}$ — pixel length of that scale bar in the exported frame;
- $d_{\text{px}}$ — any diameter measured in pixels on the mask;
- $d$ — the same diameter expressed in micrometres.

> ⚠️ If $s$ is wrong by 10 %, **every** size, area, coverage and cluster metric
> is wrong by 10 %, 21 %, 10 % and 10 % respectively. Re-verify the preset after
> every objective change.

### 4.4 Confirm the dispersed system and the size window

In Section **4. Phase Classification**, keep the single CLSM option
**🟣 Colloidal Microspheres & Aggregates (Silica)** and set the physical size
window that defines one microsphere. Every object whose equivalent diameter
falls inside the window and passes the circularity gate is counted as a sphere;
everything else is reported as aggregate debris.

### 4.5 Configure Cellpose AI segmentation

In Section **5. Cellpose AI Segmentation**:

| Control | Recommended start | Effect |
| --- | --- | --- |
| Channel / Optical Mode | 🔄 Auto-Detect | Picks composite / `c1` / `c0` automatically |
| Pretrained Model | `cyto2` (default) or `cyto3` | `cyto3` for textured agglomerates |
| 🎯 Typical Microsphere Diameter for AI | 8 µm (20x) | Sets the neural target scale |
| `flow_threshold` | 0.70 | Higher values accept irregular boundaries |
| `cellprob_threshold` | −1.0 | Lower values recover faint spheres |
| 🟣 Microsphere Min Circularity | 0.55 | Rejects crescents and merged clusters |

The effective network diameter is derived automatically and the caption must be
read before running:

$$
d_{\text{eff}} = \operatorname{clip}\!\left(\frac{d_{\text{target}}}{s},\;8\;\text{px},\;220\;\text{px}\right),
\qquad d_{\text{catcher}} \le 0.8\,d_{\text{eff}} .
$$

**Purpose:** express the operator's physical intention ("spheres of about
8 µm") as the single pixel-scale parameter the neural network expects, and bound
that request so one inference pass always costs approximately the same time.

**Where:**

- $d_{\text{target}}$ — physical microsphere diameter entered in the sidebar
  (1–60 µm, default 8 µm);
- $s$ — active calibration in µm/px;
- $d_{\text{eff}}$ — effective Cellpose diameter in pixels (clamped to
  8–220 px);
- $d_{\text{catcher}}$ — upper bound of the residual distance-transform catcher
  diameter, which recovers spheres smaller than the receptive field.

Click **Run Segmentation** once and wait for the overlay. Changing any
downstream slider (size window, circularity, opacity) must **not** re-run the
network: masks are cached per image + parameters.

### 4.6 Quality control — human-in-the-loop verification

1. Open **📊 Interactive Dashboard** and inspect the overlay at 100 % zoom.
2. Confirm that touching spheres are separated (no "peanut" blobs) and that no
   single sphere carries two contours.
3. Switch the backdrop between 🎨 Composite RGB and the single channels to check
   that the segmentation followed the physical shell, not a colour artefact.
4. Correct residual errors in **🖌️ Manual Correction** — erase false positives,
   merge split halves or add missed spheres with the SAM smart box — then return
   to the dashboard; every downstream table updates immediately.
5. Never accept a field in which the rejected-object log shows > 5 % of the
   population dropping out only because of the circularity gate.

### 4.7 Read the results

| Tab | What to verify |
| --- | --- |
| 🔬 **Metrics** | Count, $D_{10}$/$D_{50}$/$D_{90}$, $D[3,2]$, $D[4,3]$, Span per phase |
| 🍇 **Cluster & Coalescence Analytics** | $N_{agg}$ distribution, $R_g$, $D_f$, aspect ratio |
| 🛡️ **Interfacial Analytics** | Hidden in CLSM mode — expected behaviour |

Sanity rules: $D[3,2] \le D[4,3]$ always; a monodisperse silica batch should
report Span < 1.0; a flocculated batch reports $D_f \approx 1.8\text{–}2.5$.

### 4.8 Reproducibility and export

1. Go to **💾 Reproducibility & Export**.
2. Confirm the recorded SHA-256 of the source image, the build badge and the
   complete parameter snapshot.
3. Download the CSV tables, the PNG figures and the JSON manifest **before**
   closing the session; the manifest is the only record that ties a figure to
   its exact parameter set.

## 5. ✅ Acceptance Criteria and QC Checklist

- [ ] Modality is 🔦 Confocal Fluorescence and the calibration preset matches
      the objective printed on the acquisition log.
- [ ] The preview shows a green matrix **and** a magenta shell.
- [ ] AI Target Scale caption reports a plausible pixel diameter for the chosen
      objective (≈ 10 px at 20x for 8 µm spheres).
- [ ] No sphere is measured twice; touching spheres are split.
- [ ] Object count is stable (± 2 %) when `cellprob_threshold` is nudged by
      ±0.5, otherwise the field is marginal and must be re-acquired.
- [ ] $D[3,2] \le D[4,3]$ and both lie inside the Section 4 size window.
- [ ] Cluster metrics are reported only for clusters with $N_{agg} \ge 3$.
- [ ] Session manifest exported and archived with the raw micrograph.

## 6. 🧯 Troubleshooting

| Symptom | Probable cause | Corrective action |
| --- | --- | --- |
| Almost no spheres detected | Wrong channel mode / dark matrix | Set the Advanced CLSM Channel Override explicitly to `c1` or Composite RGB |
| Bright rings measured as hollow objects | `c0` shell plane used without flood filling | Keep Auto-Detect on — it hole-fills and domes the rings |
| Touching spheres merged into one blob | Shell ridge not subtracted / circularity too high | Use Composite RGB mode and lower Min Circularity to ≈ 0.45 |
| Huge object split into shards | Target diameter far below the real sphere size | Raise 🎯 Typical Microsphere Diameter |
| Diameters off by a constant factor | Stale objective preset | Re-select the preset or recalibrate the custom entry |
| Cluster metrics show $D_f = 0$ | Fewer than three multi-particle clusters | Image more fields; single spheres carry no fractal information |

## 7. 📜 Records and Archiving

Retain, for each reported dataset: the raw composite TIFF, the JSON session
manifest, the exported CSV tables, and this SOP revision. The manifest alone is
sufficient to reproduce the run because it stores the build identity, the
parameter snapshot, the calibration and the image hash.

## 8. 🔗 Related Documents

- [`docs/Methodology_CLSM.md`](Methodology_CLSM.md) — every equation used here,
  with variable definitions and assumptions.
- [`docs/SOP_SEM.md`](SOP_SEM.md) — electron-microscopy procedure.
- [`docs/SOP_Light.md`](SOP_Light.md) — transmitted-light emulsion procedure.
- [`README.md`](../README.md) — installation and usage overview.

