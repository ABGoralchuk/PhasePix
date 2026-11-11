# 📐 Methodology — SEM Microstructure, Granulometry and Interfacial Analytics

> **Document:** `docs/Methodology_SEM.md` · **Downloadable Word twin:** `docs/Methodology_SEM.docx`
> **Companion procedure:** [`SOP_SEM.md`](SOP_SEM.md)
> **Modality:** 🔬 Scanning Electron Microscopy
> **Phases:** 🫧 air · 🟧 matrix fat · 🟨 adsorbed fat · 👑 Pickering corona
> **Revision:** 2.1.0 · **Validated on build:** v2.0.0 and later

## 1. 🎯 Purpose and Scope

This document states how PhasePix turns a greyscale SEM micrograph into
traceable microstructure numbers: databar exclusion, scale calibration, hybrid
neural/morphological segmentation, phase assignment rules, granulometry and the
interfacial (surface coverage, packing density, contact angle) analytics. It is
the reference for the thresholds exposed in the sidebar, and every equation is
implemented in `app.py` / `analyzer.py` / `cellpose_model.py`.

The methodology is **deterministic** and **auditable**: each classification
decision is a closed-form inequality on measurable geometry, so any object in
the exported table can be traced back to the rule that accepted or rejected it.

## 2. 🔤 Symbols and Notation

| Symbol | Meaning | Unit |
| --- | --- | --- |
| $\hat{I}$ | globally normalised intensity | 0–255 |
| $f$ | excluded footer fraction | – |
| $s$ | spatial sampling | µm/px |
| $A$ | mask area | px² / µm² |
| $P$ | mask perimeter | px / µm |
| $d_{\text{eq}}$ | equivalent circular diameter | µm |
| $C$ | circularity | – |
| $S$ | solidity (convexity) | – |
| $AR$ | aspect ratio of the fitted ellipse | – |
| $\Omega_b$ | air-bubble mask | – |
| $G_i$ | $i$-th adsorbed / corona globule mask | – |
| $L_c$ | visible chord of a globule on the interface | µm |
| $D$ | globule diameter | µm |
| $\theta$ | apparent three-phase contact angle | ° |
| $\rho_p$ | packing density | glob/µm² |

## 3. 🖼️ Image Formation and Pre-Processing

### 3.1 Region of interest and databar exclusion

$$
\Omega_{\text{ROI}} = \{(x,y) \mid 0 \le x < W,\; 0 \le y \le H(1-f)\},
\qquad
f \in [0,\,0.25].
$$

**Purpose:** remove the instrument databar — a high-contrast rectangle with text
and a scale bar — from the segmentation domain, because the network would
otherwise return it as a large, perfectly solid "particle" and inflate the size
distribution.

**Where:**

- $W$, $H$ — frame width and height in pixels;
- $f$ — footer fraction excluded in Sidebar Section 2;
- $\Omega_{\text{ROI}}$ — analysed region.

The scale bar is detected **before** the crop, so calibration survives the
exclusion.

### 3.2 Normalisation and local contrast enhancement

$$
\hat{I} = 255\cdot\frac{I - I_{\min}}{I_{\max} - I_{\min}},
\qquad
I_{\text{pre}} = \operatorname{CLAHE}_\lambda^{\tau\times\tau}(\hat{I}).
$$

**Purpose:** make one fixed set of thresholds valid across detectors and
sessions, and recover shadowed crater interiors without globally stretching the
contrast.

**Where:**

- $I$ — raw greyscale intensity (8-bit or 16-bit);
- $I_{\min}$, $I_{\max}$ — global extrema of the frame;
- $\lambda$ — CLAHE clip limit (higher = more local contrast, more noise);
- $\tau$ — CLAHE tile grid size in pixels;
- $I_{\text{pre}}$ — the plane handed to the neural network.

CLAHE is enabled by default in SEM mode because fractured surfaces are strongly
shadowed; it is disabled by default in Light mode, where it amplifies brightfield
halos.

## 4. 🧠 Segmentation

### 4.1 Neural pass

$$
\mathcal{S} = \operatorname{Cellpose}\!\big(I_{\text{pre}},\, d_{\text{eff}},\, f_{\text{flow}},\, p_{\text{cell}}\big)
$$

with $d_{\text{eff}}$ derived from the declared size window (SOP §4.5), the flow
threshold $f_{\text{flow}}$ (default 0.70) bounding the tolerated boundary-flow
error, and $p_{\text{cell}}$ (default −1.0) the cell-probability floor.

**Purpose:** obtain object candidates without hand-tuned edge thresholds; the two
thresholds only trade boundary tolerance against recall, they do not change the
physical definition of a particle.

### 4.2 Two-pass foam recovery

In **🫧+🟧 Both Air Bubbles & Fat Particles** mode the frame is segmented twice:

$$
\mathcal{S} = \mathcal{S}_{\text{air}}\big(d_{\text{eff}}^{\text{air}}\big)
\;\cup\;
\Big(\mathcal{S}_{\text{fat}}\big(d_{\text{eff}}^{\text{fat}}\big)
\setminus \mathcal{S}_{\text{air}}\Big),
$$

**Purpose:** a single network diameter cannot resolve 10–100 µm bubbles and
0.2–5 µm globules simultaneously; the union–difference keeps every object in
exactly one phase population.

**Where:**

- $\mathcal{S}_{\text{air}}$ — masks from the air-optimised pass (25–220 px);
- $\mathcal{S}_{\text{fat}}$ — masks from the fat-optimised pass (18–45 px),
  with air masks subtracted so no object is double-counted;
- $d_{\text{eff}}^{\text{air}}, d_{\text{eff}}^{\text{fat}}$ — as defined in the
  SOP.

### 4.3 Deep-scan adsorbed-fat recovery

When 🔬 **Deep-Scan Adsorbed Fat inside Bubbles** is enabled, each confirmed
bubble interior is re-segmented with shadow recovery, and the recovered masks
are kept only inside $\Omega_b$ minus a rim margin.

**Purpose:** globules adsorbed on the *inside* wall of a bubble are shadowed by
the rim and are systematically missed by a global pass; the local pass restores
them without changing the global thresholds.

## 5. 🏷️ Phase Assignment Rules

Every candidate mask is evaluated against the following ordered rules. The first
rule that matches wins, so each object belongs to exactly one phase.

$$
\text{air}: \; d_{\min}^{\text{air}} \le d_{\text{eq}} \le d_{\max}^{\text{air}}
\;\wedge\; C \ge C_{\text{air}},
$$

$$
\text{adsorbed}: \; \frac{A_{i \cap b}}{A_i} \ge 0.15
\;\wedge\; C \ge C_{\text{ads}},
$$

$$
\text{corona}: \; \operatorname{dist}\!\left(\partial\Omega_b,\, \mathbf{r}_i\right) \le w_{\text{corona}}
\;\wedge\; C \ge C_{\text{corona}} \;\wedge\; S \ge 0.78 \;\wedge\; AR \le 1.65,
$$

$$
\text{matrix fat}: \; d_{\min}^{\text{fat}} \le d_{\text{eq}} \le d_{\max}^{\text{fat}}
\;\wedge\; C \ge C_{\text{fat}} .
$$

**Purpose:** replace subjective "this looks adsorbed to me" judgements with a
closed-form, reproducible rule set, so two operators analysing the same field
obtain the same phase populations.

**Where:**

- $d_{\text{eq}}$ — equivalent circular diameter (µm);
- $A_{i \cap b}$ — the part of globule $i$ that overlaps the bubble mask;
- $\partial\Omega_b$ — the bubble rim;
- $w_{\text{corona}}$ — 👑 Corona Rim Capture Distance (default 1.2 µm), the
  band width around the rim;
- $C_{\text{air}}, C_{\text{fat}}, C_{\text{ads}}, C_{\text{corona}}$ — the
  circularity gates (defaults 0.50, 0.60, 0.30 and 0.60);
- $S$, $AR$ — solidity and aspect ratio, which reject crescent rim strips and
  elongated rim-wall shards that a circularity gate alone would accept.

Objects that satisfy no rule are retained as *unclassified* and are visible in
the rejected-object log.

## 6. 📏 Morphometry

For an accepted mask with area $A$ (px²), perimeter $P$ (px) and calibrated
sampling $s$:

$$
d_{\text{eq}} = 2\sqrt{\frac{A}{\pi}}\,s,
\qquad
C = \frac{4\pi A}{P^2},
\qquad
S = \frac{A}{A_{\text{hull}}},
\qquad
A\,[\mu\text{m}^2] = A\,[\text{px}^2]\cdot s^{2}.
$$

**Purpose:** express size and shape in scale-free, physically interpretable
terms so gates are portable between magnifications.

**Where:**

- $d_{\text{eq}}$ — diameter of the circle with the same area;
- $C$ — circularity, $1$ for a circle, $\approx 0.55$ for a typical fractured
  bubble rim, $< 0.4$ for torn fragments;
- $A_{\text{hull}}$ — convex hull area of the mask (solidity denominator);
- $A\,[\mu\text{m}^2]$ — calibrated area used by coverage metrics.

## 7. 📊 Granulometry

Per phase, with $d_i$ the equivalent diameters:

$$
D[3,2] = \frac{\sum_i d_i^{3}}{\sum_i d_i^{2}},
\qquad
D[4,3] = \frac{\sum_i d_i^{4}}{\sum_i d_i^{3}},
\qquad
\text{Span} = \frac{D_{90} - D_{10}}{D_{50}} .
$$

**Purpose:** report every phase with the same surface- and volume-weighted means
used by laser diffraction, so SEM and diffraction results for the same emulsion
can be compared directly.

**Where:**

- $D[3,2]$ — Sauter mean (specific-surface equivalent);
- $D[4,3]$ — De Brouckere mean (volume equivalent);
- $D_{10}, D_{50}, D_{90}$ — decile diameters from the empirical CDF;
- $\text{Span}$ — relative width of the distribution.

Two invariants are checked before a table is displayed: $D[3,2] \le D[4,3]$, and
every $d_i$ lies inside the declared window of its phase (otherwise the object
would have been reassigned).

## 8. 🛡️ Interfacial (Pickering) Analytics

### 8.1 Surface and rim coverage

$$
\Theta_{\text{cov}} = \frac{\left|\Omega_b \cap \bigcup_i G_i\right|}{\left|\Omega_b\right|}\times 100,
\qquad
\Theta_{\text{rim}} = \frac{\left|\Omega_b \cap R_w \cap \bigcup_i G_i\right|}
{\left|\Omega_b \cap R_w\right|}\times 100 .
$$

**Purpose:** quantify how much of the bubble interface is stabilised by fat
globules, both over the whole visible floor ($\Theta_{\text{cov}}$) and in the
equatorial band that the fracture plane actually cut ($\Theta_{\text{rim}}$, the
statistically safer estimate).

**Where:**

- $\Omega_b$ — segmented bubble mask;
- $G_i$ — adsorbed and corona globule masks;
- $R_w$ — equatorial rim band of width $w_{\text{rim}}$ inside the bubble;
- $\Theta_{\text{cov}}$, $\Theta_{\text{rim}}$ — percentages.

### 8.2 Packing density

$$
\rho_p = \frac{N_{\text{ads}}}{A_b\,[\mu\text{m}^2]} .
$$

**Purpose:** separate "are the globules there?" from "are they packed tightly?",
which are different physical questions when the interface is only partially
covered.

**Where:**

- $N_{\text{ads}}$ — number of adsorbed (gold) globules inside the bubble;
- $A_b$ — calibrated bubble area in µm²;
- $\rho_p$ — globules per µm².

### 8.3 Apparent contact angle

$$
\theta_i = \arcsin\!\left(\operatorname{clip}\!\left(\frac{L_{c,i}}{D_i},\,0,\,1\right)\right),
\qquad
\theta_i^{\text{prot}} = \arccos\!\left(\operatorname{clip}\!\left(\frac{d_i}{R_i},\,0,\,1\right)\right).
$$

**Purpose:** estimate the three-phase wetting angle from a two-dimensional
cross-section, and provide an independent geometric cross-check so a wrong
assignment (a globule that only *touches* the rim) cannot bias the mean angle.

**Where:**

- $L_{c,i}$ — visible chord of globule $i$ on the interface plane;
- $D_i$ — its full diameter;
- $d_i$ — distance from the globule centre to the interface plane;
- $R_i$ — its radius;
- $\theta_i$ — chord-based angle (reported for corona globules);
- $\theta_i^{\text{prot}}$ — protrusion-based angle (cross-check, ±10°
  tolerance).

Interior globules receive $\theta = \text{NaN}$ and are excluded from the mean,
because a globule that never intersects the interface carries no wetting
information.

## 9. 🎯 Uncertainty Propagation

$$
\frac{\delta d}{d} = \frac{\delta s}{s},
\qquad
\frac{\delta \Theta_{\text{cov}}}{\Theta_{\text{cov}}} \approx
2\,\frac{\delta s}{s},
\qquad
\delta\theta \approx \frac{\delta s/s}{\sqrt{1 - (L_c/D)^2}} .
$$

**Purpose:** show which reported quantities are sensitive to the calibration and
which are not. Coverage is an area ratio and therefore scales with $s^2$ in the
numerator and denominator, cancelling to first order — but mask-boundary bias
still enters, so it is reported with the pixel-level confidence interval.

**Where:**

- $\delta s/s$ — relative calibration uncertainty (typically 1–3 % for SEM
  databars);
- $\delta\theta$ — angle uncertainty, which grows steeply as the chord
  approaches the full diameter (a globule seen almost equator-on carries an
  ill-conditioned angle).

## 10. ⚠️ Assumptions and Limitations

1. **Planar proxy of a 3D interface.** Coverage and $\theta$ are derived from the
   visible fracture cross-section; they describe the exposed bubble floor, not
   the whole sphere surface.
2. **Rim integrity.** A bubble whose rim is cut by the fracture cannot be used for
   rim coverage or $\theta$.
3. **Greyscale signal.** Fat, protein and ice are all grey; phase assignment is
   geometric, not compositional. EDX or staining is required to prove chemistry.
4. **Neural model domain.** Cellpose was trained on optical microscopy; strongly
   textured fracture surfaces may need `cyto3` and manual review.
5. **Threshold sensitivity.** Gates are step functions: a globule exactly at the
   circularity gate can flip between phases if the segmentation boundary moves by
   one pixel. Report gates with every table.
6. **Two-pass ordering.** In two-pass mode the air pass always wins, so a small
   bubble that also satisfies the fat window is reported as air.

## 11. 🔁 Reproducibility Contract

Archive the raw micrograph, its SHA-256, the build identity, the calibration,
the saved parameter preset and the exported tables. With those five items the
same masks and the same statistics can be regenerated exactly; nothing in the
pipeline depends on UI order or on the current session state.

## 12. 📚 References

1. Stringer, C. et al. *Cellpose: a generalist algorithm for cellular
   segmentation.* Nature Methods 18, 100–106 (2021).
2. Weaire, D. & Hutzler, S. *The Physics of Foams.* Oxford University Press
   (1999).
3. Binks, B. P. & Horozov, T. S. *Colloidal Particles at Liquid Interfaces.*
   Cambridge University Press (2006) — coverage and contact-angle definitions.
4. ISO 13320:2020 — particle-size analysis by laser diffraction.
5. Russ, J. C. *The Image Processing Handbook*, 7th ed. — circularity, convexity
   and perimeter measurement.
6. Zuiderveld, K. *Contrast limited adaptive histogram equalization.* Graphics
   Gems IV, 474–485 (1994).

---

*This methodology is mirrored by the downloadable Word document
[`docs/Methodology_SEM.docx`](Methodology_SEM.docx). Editing it by hand makes your
Word file authoritative — PhasePix will serve it and will not regenerate it from
the Markdown.*


