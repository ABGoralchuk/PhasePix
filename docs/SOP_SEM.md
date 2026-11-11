# 📋 Standard Operating Procedure — Quantitative SEM Analysis of Emulsions, Foams & Pickering Systems

> **Document:** `docs/SOP_SEM.md` · **Downloadable Word twin:** `docs/SOP_SEM.docx`
> **System:** PhasePix — Quantitative Microstructure & Colloid Analyzer
> **Modality:** 🔬 Scanning Electron Microscopy (SEM)
> **Dispersed systems:** 🫧 air bubbles · 🟧 matrix fat · 🟨 adsorbed fat · 👑 Pickering corona
> **Revision:** 2.0.1 · **Validated on build:** v2.0.0 and later

| Field | Value |
| --- | --- |
| Procedure owner | Microscopy & Colloid Analytics Laboratory |
| Target analytes | Air bubbles (10–100 µm), fat globules (0.2–5 µm), interfacial corona globules |
| Calibration route | Embedded databar / scale bar (OCR-assisted, operator-verified) |
| Segmentation engine | Cellpose `cyto2` / `cyto3` + two-pass foam recovery + deep-scan |
| Deliverables | Per-phase PSD, granulometry, interfacial coverage, contact angle $\theta$ |
| Archive | CSV tables + PNG figures + JSON session manifest |

## 1. 🎯 Purpose and Scope

This SOP defines how to obtain **quantitative, reproducible microstructure
statistics from SEM micrographs** of food emulsions, whipped emulsions (foams
with a fat phase) and Pickering-stabilised systems. It covers databar removal,
scale-bar calibration, sample-type selection, AI segmentation, interfacial
(Pickering corona) analytics and archiving.

The procedure applies to:

- **Whipped Emulsion** — the full multi-phase set: air bubbles, matrix fat,
  adsorbed fat inside bubbles and the equatorial corona on the bubble rim;
- **Emulsion** — single-phase fat droplets in a continuous matrix;
- **Foam** — single-phase air bubbles without a measurable fat phase.

**Out of scope:** any measurement without a verified scale (the SEM databar is
mandatory in this modality) and elemental/EDX composition, which PhasePix does
not perform.

## 2. 🧰 Before You Start — Prerequisites

| # | Requirement | Acceptance check |
| --- | --- | --- |
| 1 | Micrograph with the instrument databar visible | Databar text is legible at 100 % zoom |
| 2 | Nominal magnification and pixel size known | Printed in the acquisition log |
| 3 | No charging artefacts over the region of interest | No bright streaks or drifting bands |
| 4 | Frame representative of the batch | At least 3 fields per sample, ≥ 200 objects total |
| 5 | Same preparation protocol across compared samples | Freeze-fracture / sputter settings recorded |

## 3. 🖼️ Image Preparation Rules

1. **Keep the native greyscale export** (`.tif` preferred). Do not convert to
   JPEG: block artefacts bias both the circularity gate and the adsorbed-fat
   area.
2. **Do not crop the databar before upload** — PhasePix removes it itself and
   needs the databar text for OCR-based calibration. Cropping manually is only
   allowed when the user then enters the scale manually.
3. **Preserve the original bit depth**; 16-bit exports are normalised globally
   once.
4. **Avoid stitching** unless the stitching is documented, because stitching
   seams create artificial edges that the neural network reads as particle
   boundaries.
5. **Do not upscale or sharpen**; both change the perimeter-to-area relation and
   therefore circularity.

## 4. 🪜 Step-by-Step Procedure

### 4.1 Select the imaging modality

In Sidebar **0. Imaging Modality**, choose **🔬 SEM Micrograph**. This enables
the databar-exclusion panel, the databar calibration route and the full
four-phase classification.

### 4.2 Load the micrograph and exclude the footer databar

In Section **1. Image Input**, load the file. Then, in Section
**2. Footer Panel Exclusion**, confirm the crop that removes the instrument
databar. The exclusion rule is:

$$
\Omega_{\text{ROI}} = \{(x,y) \;|\; y \le H\,(1 - f)\},
\qquad
f \in [0,\;0.25],
$$

**Purpose:** guarantee that databar text, logos and the scale bar itself never
enter the segmentation, because they are high-contrast rectangles that the model
would otherwise count as particles.

**Where:**

- $\Omega_{\text{ROI}}$ — analysed region of interest;
- $H$ — frame height in pixels;
- $f$ — excluded footer fraction selected in the sidebar (0 = keep everything,
  0.25 = drop the bottom quarter of the frame).

> 👑 Tip: the scale bar must still be **read** before it is excluded. PhasePix
> probes the databar first and only then crops.

### 4.3 Calibrate the spatial scale

In Section **3. Spatial Calibration**, either accept the auto-detected scale bar
or type the physical length in micrometres and let PhasePix measure the bar's
pixel length. The calibration is:

$$
s = \frac{L_{\text{bar}}}{N_{\text{px,bar}}}\quad[\mu\text{m}/\text{px}],
\qquad
d\,[\mu\text{m}] = s\,d_{\text{px}} .
$$

**Purpose:** anchor every downstream length, area, coverage and contact-angle
number to the instrument scale; the SEM databar is the only metrological
reference in the file, so an error here invalidates the whole measurement.

**Where:**

- $L_{\text{bar}}$ — nominal bar length printed by the microscope (µm);
- $N_{\text{px,bar}}$ — measured pixel length of that bar (auto-detected, then
  operator-verified);
- $s$ — resulting µm/px sampling;
- $d_{\text{px}}$, $d$ — a diameter before and after calibration.

Always cross-check: the printed magnification and $s$ must agree within about
5 % for a well-calibrated instrument.

### 4.4 Select the sample type and the size windows

Section **4. Phase Classification** drives which controls exist:

| Sample type | Active phases | Default size windows |
| --- | --- | --- |
| 🍦 **Whipped Emulsion** | 🫧 air + 🟧 fat + 🟨 adsorbed + 👑 corona | air 10–100 µm, fat 0.20–5.0 µm |
| 🥛 **Emulsion** | 🟧 matrix fat | fat 0.20–5.0 µm |
| 🫧 **Foam** | 🫧 air | air 10–100 µm |

A candidate is assigned to the air phase when its equivalent diameter lies in
the air window **and** its circularity exceeds the air gate; otherwise it is
tested against the fat window and the fat gate. For Whipped Emulsion the
detection focus offers:

- **🫧 Air Bubbles Only (Fast & Precise)** — single fast pass, air metrics only;
- **🫧+🟧 Both Air Bubbles & Fat Particles (Two-Pass)** — air pass plus a
  fat-optimised pass with a smaller network diameter.

### 4.5 Configure Cellpose AI segmentation

| Control | Default | Notes |
| --- | --- | --- |
| Pretrained Model | `cyto2` | `cyto3` for strongly textured fracture surfaces |
| `flow_threshold` | 0.70 | Raise to 0.85 for ragged bubble rims |
| `cellprob_threshold` | −1.0 | Lower to −2.0 to recover shadowed craters |
| Shared: Enable CLAHE | on | Recommended for SEM contrast recovery |
| 🫧 Air Min Circularity | 0.50 | Rejects torn rim fragments |
| 🟧 Matrix Fat Min Circularity | 0.60 | Rejects shards |
| 🟨 Adsorbed Fat Min Circularity | 0.30 | Permissive: globules sit on curved walls |
| 👑 Corona Rim Capture Distance | 1.2 µm | Band width around the bubble rim |
| 👑 Corona Fat Min Circularity | 0.60 | Combined with $S \ge 0.78$, $AR \le 1.65$ |
| 🔬 Deep-Scan Adsorbed Fat inside Bubbles | off | Slower; use for Pickering studies |

The network diameter is **derived**, never typed directly:

$$
d_{\text{eff}} =
\begin{cases}
\operatorname{clip}\!\big(1.25\,d_{\min}^{\text{fat}}/s,\;18,\;45\big), & \text{fat pass},\\[4pt]
\operatorname{clip}\!\big(\sqrt{d_{\min}^{\text{air}} d_{\max}^{\text{air}}}/s,\;25,\;220\big), & \text{air pass}.
\end{cases}
$$

**Purpose:** derive a single, physically meaningful pixel scale from the size
window the operator already declared, so the network is neither blind to small
globules nor fragmenting large bubbles.

**Where:**

- $d_{\min}^{\text{fat}}$ — minimum fat diameter from Section 4 (µm);
- $d_{\min}^{\text{air}}, d_{\max}^{\text{air}}$ — air size window bounds (µm);
- $s$ — calibration (µm/px);
- $d_{\text{eff}}$ — effective Cellpose diameter (px), clamped to the stated
  range for each pass.

### 4.6 Quality control

1. Inspect the overlay at 100 % zoom; reclassify individual objects in
   **🖌️ Manual Correction** when a globule is systematically misassigned.
2. Verify the phase-colour legend: 🫧 blue = air, 🟧 orange = matrix fat,
   🟨 gold = adsorbed fat, 👑 amber = corona.
3. Confirm that the bubble rim is intact around the whole perimeter; a broken
   rim means the fracture plane cut through the bubble and the contact-angle
   measurement is invalid for that bubble.
4. Keep the same parameter set for every field of the same sample; changing the
   circularity gate between fields invalidates the comparison.

### 4.7 Read the interfacial (Pickering) analytics

Open **🛡️ Interfacial Analytics** (Whipped Emulsion only). The tab reports:

- **Surface Coverage (%)** — the fraction of the bubble floor occupied by
  adsorbed and corona globules;
- **Packing Density (glob/µm²)** — adsorbed globules per unit bubble area;
- **Rim Coverage (%)** — coverage measured inside the equatorial rim band;
- **Contact angle $\theta$** — the three-phase angle from the chord recipe, with
  a protrusion-based cross-check.

$$
\Theta_{\text{cov}} = \frac{\left|\Omega_b \cap \bigcup_i G_i\right|}
{\left|\Omega_b\right|}\times 100,
\qquad
\theta_i = \arcsin\!\left(\operatorname{clip}\!\frac{L_{c,i}}{D_i},\,0,\,1\right).
$$

**Purpose:** convert a two-dimensional projection of the bubble surface into the
three quantities that control Pickering stability: how much interface is
covered, how densely, and how strongly each globule is wetted.

**Where:**

- $\Omega_b$ — segmented air-bubble mask (the "floor" of the crater);
- $G_i$ — mask of the $i$-th adsorbed / corona globule;
- $L_{c,i}$ — visible chord length of globule $i$ where it meets the interface;
- $D_i$ — full diameter of globule $i$;
- $\theta_i$ — apparent contact angle in degrees (reported as NaN for interior
  globules, which never touch the interface).

The independent cross-check is $\theta_i^{\text{prot}} =
\arccos(\operatorname{clip}(d_i/R_i,0,1))$, where $d_i$ is the distance of the
globule centre from the interface plane and $R_i$ its own radius; agreement
within ±10° is required before a value is reported as valid.

### 4.8 Export and archive

Go to **💾 Reproducibility & Export** and download the per-phase CSV tables, the
PNG figures and the JSON manifest. The manifest records the image SHA-256, the
build identity, the calibration and every parameter value used.

## 5. ✅ Acceptance Criteria and QC Checklist

- [ ] Modality is 🔬 SEM and the databar was excluded without cutting particles.
- [ ] Calibration verified against the printed magnification (± 5 %).
- [ ] Air, fat, adsorbed and corona gates are the documented defaults or a
      justified deviation is recorded.
- [ ] $\ge 200$ objects measured per sample; $\ge 3$ fields per condition.
- [ ] No object is counted in two phases.
- [ ] Rim integrity confirmed for every bubble used in $\theta$ statistics.
- [ ] $D[3,2] \le D[4,3]$ for each phase.
- [ ] Manifest exported and stored with the micrographs.

## 6. 🧯 Troubleshooting

| Symptom | Probable cause | Corrective action |
| --- | --- | --- |
| Databar counted as particles | Footer exclusion set to 0 | Increase the excluded footer fraction |
| Scale wildly wrong | OCR read a different number block | Enter the bar length manually |
| Bubbles split into arcs | `cellprob_threshold` too high or rim partly cut | Lower to −2.0 and re-acquire an intact bubble |
| Fat globules merged with the bubble rim | Corona band too narrow | Raise 👑 Corona Rim Capture Distance to 1.5–2.0 µm |
| Corona shards reported as globules | Circularity gate too low | Raise 👑 Corona Fat Min Circularity or rely on the $S$/$AR$ gates |
| $\theta$ missing for most globules | Globules classified as interior | Check that the rim band actually intersects the globule centroids |
| Counts unstable between reruns | Parameters changed between fields | Re-apply one saved preset to all fields |

## 7. 📜 Records and Archiving

Archive per dataset: the raw SEM micrographs, the JSON session manifest, the
exported CSV/PNG files, the acquisition log with magnification and pixel size,
and this SOP revision. Retain the manual-correction table whenever objects were
reclassified by hand, because it documents the human decisions behind the
numbers.

## 8. 🔗 Related Documents

- [`docs/Methodology_SEM.md`](Methodology_SEM.md) — equations, thresholds and
  assumptions behind this procedure.
- [`docs/SOP_CLSM.md`](SOP_CLSM.md) — confocal microsphere procedure.
- [`docs/SOP_Light.md`](SOP_Light.md) — transmitted-light emulsion procedure.
- [`README.md`](../README.md) — installation and usage overview.


