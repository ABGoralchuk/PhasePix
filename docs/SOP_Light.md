# 📋 Standard Operating Procedure — Quantitative Light-Microscopy Analysis of Emulsion Droplets

> **Document:** `docs/SOP_Light.md` · **Downloadable Word twin:** `docs/SOP_Light.docx`
> **System:** PhasePix — Quantitative Microstructure & Colloid Analyzer
> **Modality:** 💡 Light (transmitted-light / brightfield) Microscopy
> **Dispersed system:** 🟧 Fat droplets — 💧 liquid or ❄️ crystallised
> **Revision:** 2.1.0 · **Validated on build:** v2.0.0 and later

| Field | Value |
| --- | --- |
| Procedure owner | Microscopy & Colloid Analytics Laboratory |
| Target analyte | Fat globules / droplets, 0.5–50 µm (liquid or crystallised) |
| Calibration route | Stored objective preset (µm/px), optional scale-bar override |
| Segmentation engine | Cellpose `cyto2` / `cyto3`, optional multi-scale (coarse + fine) |
| Deliverables | Droplet PSD, $D_{10}/D_{50}/D_{90}$, $D[3,2]$, $D[4,3]$, Span, counts |
| Archive | CSV tables + PNG figures + JSON session manifest |

## 1. 🎯 Purpose and Scope

This SOP defines how to quantify **fat-droplet size distributions in
transmitted-light micrographs** of liquid and crystallised emulsions with
PhasePix. It covers the brightfield-specific preparation rules, objective
calibration, emulsion-profile selection (liquid vs crystallised), AI
segmentation, human verification and export.

The procedure applies to:

- **💧 Liquid Fat (Smooth Globules)** — homogeneous, near-spherical droplets;
- **❄️ Crystallised Fat (Textured / Internal Crystals)** — globules containing
  internal crystal needles that must be smoothed away before segmentation,
  otherwise the network splits one globule into several shards.

**Out of scope:** air-bubble, adsorbed-fat and Pickering-corona analytics (a
single-phase transmitted-light emulsion has no air phase, so 🛡️ Interfacial
Analytics is hidden in this modality), and any measurement without a calibrated
objective.

## 2. 🧰 Before You Start — Prerequisites

| # | Requirement | Acceptance check |
| --- | --- | --- |
| 1 | Objective and camera recorded | Magnification and pixel size written in the acquisition log |
| 2 | Preset available | The matching objective preset exists in `light_calibration_presets.json` |
| 3 | Uniform illumination | No vignetting or lamp flicker across the field |
| 4 | Focus at the droplet equator | Droplet edges are sharp, not haloed |
| 5 | Representative sampling | ≥ 3 fields and ≥ 300 droplets per sample |

## 3. 🖼️ Image Preparation Rules

1. **Do not saturate the highlights.** Droplet rims that clip to pure white lose
   their gradient and the network cannot locate the boundary.
2. **Keep the native resolution.** Down-sampling an image before upload destroys
   sub-micron droplets; PhasePix performs its own bounded downscale if needed.
3. **Export without sharpening or auto-contrast.** Both change the
   perimeter-to-area ratio and therefore circularity.
4. **Preserve colour** if the acquisition is colour; the inversion option can then
   be used for inverted brightfield.
5. **Avoid coverslip pressure artefacts.** Flattened droplets have $AR > 1.3$ and
   will be rejected by the shape gates — re-prepare the slide instead of lowering
   the gates.

## 4. 🪜 Step-by-Step Procedure

### 4.1 Select the imaging modality

In Sidebar **0. Imaging Modality**, choose **💡 Light Microscopy**. This:

- hides the SEM databar/footer panel and the air/corona controls;
- enables the objective presets stored for light microscopy;
- enables the **Emulsion Droplet Profile** selector in Section 5;
- removes 🛡️ Interfacial Analytics from the navigation bar.

### 4.2 Load the micrograph

In Section **1. Image Input**, load the file from the light-micrograph folder or
upload it. Check the preview: droplet interiors must be visible and the
background must be uniform.

### 4.3 Verify the objective calibration

In Section **3. Spatial Calibration**, select the preset that matches the optics
used — including any 1.6× intermediate magnification:

| Preset | $\mu\text{m}/\text{px}$ | Scale bar |
| --- | --- | --- |
| 10x Objective | 0.72098 | 10 µm / 13.87 px |
| 10x + 1.6x Objective | 0.45998 | 10 µm / 21.74 px |
| 20x Objective | 0.35348 | 10 µm / 28.29 px |
| 20x + 1.6x Objective | 0.22989 | 10 µm / 43.50 px |
| 40x Objective | 0.17915 | 10 µm / 55.82 px |
| 40x + 1.6x Objective | 0.11662 | 10 µm / 85.75 px |
| 63x Objective | 0.11424 | 10 µm / 87.54 px |
| 63x + 1.6x Objective | 0.07420 | 10 µm / 134.77 px |
| 100x Objective | 0.07149 | 10 µm / 139.87 px |
| 100x + 1.6x Objective | 0.04667 | 10 µm / 214.28 px |
| Custom Calibration | user | user |

$$
s = \frac{L_{\text{bar}}}{N_{\text{px,bar}}}\quad[\mu\text{m}/\text{px}],
\qquad
d\,[\mu\text{m}] = s\,d_{\text{px}} .
$$

**Purpose:** convert every pixel measurement into micrometres with the optics
that actually produced the frame; unlike SEM, there is no databar in the image,
so the preset *is* the metrological reference.

**Where:**

- $L_{\text{bar}}$ — physical length of the calibration bar used to build the
  preset (µm);
- $N_{\text{px,bar}}$ — its pixel length;
- $s$ — µm/px sampling stored in the preset;
- $d_{\text{px}}$, $d$ — diameter before and after calibration.

Custom presets can be saved and are restored on the next session; presets are
stored in `light_calibration_presets.json` next to `app.py`.

### 4.4 Choose the emulsion profile and size window

In Section **4. Phase Classification** set the physical droplet window
(default 0.5–50 µm). In Section **5** choose the profile:

| Profile | When to use | Key effects |
| --- | --- | --- |
| 💧 Liquid Fat (Smooth Globules) | Homogeneous liquid droplets | No pre-smoothing, default AI diameter 5 µm, solidity gate 0.72 |
| ❄️ Crystallised Fat (Textured) | Visible crystal needles inside droplets | Edge-preserving smoothing, default AI diameter 9 µm, solidity gate 0.84, multi-scale on |

The physical-to-pixel conversion of the AI target scale is:

$$
d_{\text{eff}} = \operatorname{clip}\!\left(\frac{d_{\text{target}}}{s},\;12\;\text{px},\;220\;\text{px}\right),
\qquad
d_{\text{target}} \in [1,\,40]\ \mu\text{m}.
$$

**Purpose:** let the operator express the physical droplet size they care about
and translate it into the single pixel scale the network requires, while
preventing a pathological preset (e.g. 1 µm at 10x) from producing a huge,
slow inference.

**Where:**

- $d_{\text{target}}$ — 🎯 Typical Droplet Diameter for AI (µm); default 5 µm
  for 💧 liquid and 9 µm for ❄️ crystallised fat;
- $s$ — active calibration (µm/px);
- $d_{\text{eff}}$ — effective Cellpose diameter in pixels, clamped to
  12–220 px.

### 4.5 Configure Cellpose AI segmentation

| Control | Default | Notes |
| --- | --- | --- |
| Pretrained Model | `cyto2` | `cyto3` for clumped or textured populations |
| `flow_threshold` | 0.70 | Raise for soft, low-contrast boundaries |
| `cellprob_threshold` | −1.0 | Lower to −2.0 to recover faint droplets |
| Shared: Enable CLAHE | off | Brightfield halos are amplified by CLAHE |
| 🟧 Matrix Fat Min Circularity | 0.60 | Rejects fragments and elongated debris |
| 🛡️ Matrix Fat Min Solidity | 0.72 (liquid) / 0.84 (crystallised) | Rejects shards and crescent rims |
| ❄️ Internal Crystal Smoothing Strength | 3 (crystallised only) | 0 = off, 2–3 recommended |
| 🔍 Multi-Scale Globule Detection | on (crystallised only) | Coarse pass for large globules + fine pass for small ones |

The solidity gate is the decisive brightfield filter:

$$
S = \frac{A}{A_{\text{hull}}} \ge S_{\min},
\qquad
S_{\min} = 0.72\ (\text{liquid}),\quad 0.84\ (\text{crystallised}).
$$

**Purpose:** distinguish a real droplet from a crescent-shaped fragment of a
ruptured droplet or from an inter-droplet background sliver, which circularity
alone cannot separate.

**Where:**

- $A$ — droplet mask area;
- $A_{\text{hull}}$ — area of its convex hull;
- $S_{\min}$ — the sidebar gate, raised for crystallised samples because their
  internal texture produces more non-convex fragments.

### 4.6 Quality control

1. Inspect the overlay at 100 % zoom; every droplet must carry exactly one
   contour.
2. For ❄️ crystallised samples, verify that a large globule is **one** mask and
   not several shards along its crystal lines. If shards persist, raise the
   smoothing strength or the AI target diameter.
3. Use **🖌️ Manual Correction** for residual merges and splits; the tables update
   immediately.
4. Compare the measured $D_{50}$ with the microscope's own scale bar as a
   sanity check before exporting.

### 4.7 Read the results

| Tab | What to verify |
| --- | --- |
| 🔬 **Metrics** | Count, $D_{10}/D_{50}/D_{90}$, $D[3,2]$, $D[4,3]$, Span |
| 🍇 **Cluster & Coalescence Analytics** | Coalescence indicators for flocculated droplets |
| 🛡️ **Interfacial Analytics** | Hidden in Light mode — expected behaviour |

For a well-prepared emulsion, $D[3,2] \le D[4,3]$ and the reported $D_{50}$ must
agree with the visual impression at the known magnification. A Span below about
1.0 indicates a narrow distribution; above 2.0 the sample is broadly distributed
or the segmentation mixed two populations.

### 4.8 Export and archive

In **💾 Reproducibility & Export**, download the CSV tables, PNG figures and the
JSON manifest. The manifest stores the objective preset, the profile, the gates
and the image SHA-256 needed to regenerate the result.

## 5. ✅ Acceptance Criteria and QC Checklist

- [ ] Modality is 💡 Light and the objective preset matches the optics
      (including any 1.6× intermediate magnification).
- [ ] The emulsion profile matches the physical state of the sample.
- [ ] No droplet is counted twice; no droplet is split along a crystal line.
- [ ] One contour per droplet at 100 % zoom.
- [ ] $D[3,2] \le D[4,3]$ and both inside the declared size window.
- [ ] $\ge 300$ droplets measured across $\ge 3$ fields.
- [ ] Counts stable (± 2 %) when `cellprob_threshold` is nudged by ±0.5.
- [ ] Manifest exported with the raw micrographs.

## 6. 🧯 Troubleshooting

| Symptom | Probable cause | Corrective action |
| --- | --- | --- |
| Droplets detected as dark rings only | Inverted brightfield without inversion enabled | Enable the brightfield inversion option |
| One globule split into shards | Crystal texture / AI diameter too small | Raise ❄️ smoothing to 3 and 🎯 diameter to 9–14 µm |
| Many droplets missing | Low contrast / threshold too strict | Lower `cellprob_threshold` to −2.0 and raise `flow_threshold` to 0.8 |
| Everything merged into one blob | Thresholds too permissive | Raise `cellprob_threshold` towards 0 and reinstate the circularity gate |
| Diameters off by a constant factor | Wrong objective preset (missing 1.6×) | Re-select the correct preset |
| Only fragments reported | Solidity gate rejecting flattened droplets | Re-prepare the slide; do not lower the gate below 0.70 |
| Counts differ between fields | Different parameters per field | Apply one saved preset to all fields |

## 7. 📜 Records and Archiving

Archive: raw micrographs, the objective preset name and $\mu\text{m}/\text{px}$
value, the JSON session manifest, exported CSV/PNG tables and this SOP revision.
Where droplets were edited manually, keep the manual-correction table so the
human decisions remain auditable.

## 8. 🔗 Related Documents

- [`docs/Methodology_Light.md`](Methodology_Light.md) — equations, gates and
  assumptions for transmitted-light quantification.
- [`docs/SOP_SEM.md`](SOP_SEM.md) — electron-microscopy procedure.
- [`docs/SOP_CLSM.md`](SOP_CLSM.md) — confocal microsphere procedure.
- [`README.md`](../README.md) — installation and usage overview.


