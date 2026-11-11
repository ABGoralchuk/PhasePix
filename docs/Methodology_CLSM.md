# 📐 Methodology — Confocal Fluorescence (CLSM) Microsphere & Aggregate Quantification

> **Document:** `docs/Methodology_CLSM.md` · **Downloadable Word twin:** `docs/Methodology_CLSM.docx`
> **Companion procedure:** [`SOP_CLSM.md`](SOP_CLSM.md)
> **Modality:** 🔦 Confocal Laser Scanning Microscopy · **Dispersed system:** 🟣 Silica microspheres
> **Revision:** 2.1.0 · **Validated on build:** v2.0.0 and later

## 1. 🎯 Purpose and Scope

This document states **exactly** how PhasePix converts a confocal micrograph
into numbers: the channel algebra, the hybrid neural–morphological segmentation
cascade, the morphometric and granulometric definitions, the cluster (fractal)
analytics and the propagation of calibration uncertainty. Every equation below
is implemented in `app.py` / `analyzer.py` / `cellpose_model.py` and is written
in the same notation the application displays.

The methodology is *deterministic*: identical pixels plus identical parameters
produce identical masks and identical statistics, independent of UI state, and
all randomness in the watershed seeding is replaced by explicit distance-
transform maxima.

## 2. 🔤 Symbols and Notation

| Symbol | Meaning | Unit |
| --- | --- | --- |
| $I_c(x,y)$ | raw intensity of channel $c$ at pixel $(x,y)$ | counts |
| $\hat{I}$ | globally normalised 8-bit intensity | 0–255 |
| $s$ | spatial sampling (calibration) | µm/px |
| $A$ | mask area | px² (and µm² after calibration) |
| $P$ | mask perimeter | px (µm) |
| $d_{\text{eq}}$ | equivalent circular diameter | µm |
| $C$ | circularity | – |
| $S$ | solidity (convexity) | – |
| $AR$ | aspect ratio of the fitted ellipse | – |
| $N$ | number of measured objects | – |
| $N_{agg}$ | aggregation number of one cluster | – |
| $R_g$ | radius of gyration of a cluster | µm |
| $D_f$ | 2D fractal dimension | – |
| $\kappa$ | shell-barrier subtraction coefficient | – |
| $\lambda$ | CLAHE clip limit | – |
| $\tau$ | tile grid size of the CLAHE kernel | px |

## 3. 🎨 Image Formation and Channel Decomposition

### 3.1 Global normalisation

$$
\hat{I}(x,y,c) = \frac{I(x,y,c) - \min_{x,y} I(\cdot,c)}
{\max_{x,y} I(\cdot,c) - \min_{x,y} I(\cdot,c)}\cdot 255 .
$$

**Purpose:** remove per-acquisition gain and offset differences so that one
fixed set of thresholds (circularity, contrast ratios, median brightness) is
valid across sessions, objectives and detectors.

**Where:**

- $I(x,y,c)$ — raw 8-bit or 16-bit intensity at pixel $(x,y)$ in channel $c$;
- $\min_{x,y} I(\cdot,c)$, $\max_{x,y} I(\cdot,c)$ — global extrema of that
  channel over the whole frame;
- $\hat{I}(x,y,c)$ — normalised intensity with fixed channel balance
  ($0 \le \hat{I} \le 255$).

### 3.2 Composite RGB decomposition

A colour CLSM export carries two physical signals: a **negative-contrast
matrix** (green; silica cores are dark) and a **fluorescent interfacial shell**
(magenta/red/blue). PhasePix extracts both planes analytically from the single
composite TIFF:

$$
M(x,y) = 255 - \hat{I}_G(x,y),
\qquad
H(x,y) = \max\!\big(\hat{I}_R(x,y),\,\hat{I}_B(x,y)\big).
$$

**Purpose:** reconstruct the physically meaningful dark-core mask $M$ and shell
mask $H$ without asking the operator to export separate monochrome channels, so
the measurement can never be biased by a manual channel merge.

**Where:**

- $\hat{I}_G,\hat{I}_R,\hat{I}_B$ — normalised green, red and blue planes of the
  composite image;
- $M(x,y)$ — negative-contrast core signal, high inside a silica sphere;
- $H(x,y)$ — interfacial shell signal, high on the magenta rim between spheres.

### 3.3 Shell-barrier core enhancement

Adding the shell to $M$ bridges touching cores into "peanut" blobs. PhasePix
therefore **subtracts** the inter-particle shell excess, turning every contact
neck into a separation furrow:

$$
X(x,y) = \operatorname{clip}\!\left(M(x,y) - \kappa \cdot H_{\text{excess}}(x,y),\;0,\;255\right),
\qquad
H_{\text{excess}} = \max\!\big(H - \overline{H}_{\text{in}},\;0\big).
$$

**Purpose:** convert bright inter-particle shell necks into dark grooves so the
neural network and the watershed see genuinely separate spheres, which is the
pre-condition for a correct primary-particle size distribution.

**Where:**

- $X(x,y)$ — final pre-processed plane fed to the segmentation engine;
- $\kappa$ — shell-barrier subtraction coefficient (0–1); $0$ disables the
  barrier, larger values deepen the furrow and raise the risk of over-splitting;
- $\overline{H}_{\text{in}}$ — local median shell intensity used as the
  "excess" reference, so a uniformly bright field is not penalised;
- $\operatorname{clip}(\cdot,0,255)$ — saturation guard that keeps the plane in
  the valid 8-bit range.

## 4. 🔀 Optical-Mode Auto-Detection

PhasePix classifies the input before any enhancement is applied:

$$
\text{mode} =
\begin{cases}
\text{Composite RGB}, & \text{if the array has } 3 \text{ channels},\\[2pt]
\text{Channel } c1, & \text{if single-channel and } \operatorname{median}(\hat{I}) > 90,\\[2pt]
\text{Channel } c0, & \text{otherwise}.
\end{cases}
$$

**Purpose:** choose the physically correct enhancement chain without operator
input, while still exposing a manual override for non-standard exports.

**Where:**

- $\operatorname{median}(\hat{I})$ — frame median brightness of the single
  plane; a bright-field negative-contrast matrix is bright by construction, a
  fluorescent-shell plane is dark;
- mode `c1` → invert directly, $X = 255 - \hat{I}$;
- mode `c0` → hole-fill the enclosed cavities of the bright rings and weight the
  result by its Euclidean distance transform, so a hollow shell is measured as
  one solid dome rather than as an annulus.

## 5. 🧠 Hybrid Neural–Morphological Segmentation Cascade

### 5.1 Stage 1 — Cellpose neural inference

The pre-processed plane is optionally contrast-limited-adaptive-histogram
equalised (CLAHE, clip limit $\lambda$, tile grid $\tau \times \tau$) and then
evaluated by the Cellpose network at the calibrated target diameter:

$$
\mathcal{S}_1 = \operatorname{Cellpose}\!\big(\operatorname{CLAHE}_\lambda^\tau(X),\; d_{\text{eff}},\; f_{\text{flow}},\; p_{\text{cell}}\big).
$$

**Purpose:** obtain a first, topologically consistent set of sphere masks from a
generalist deep-learning model that does not depend on hand-tuned thresholds.

**Where:**

- $d_{\text{eff}} = \operatorname{clip}(d_{\text{target}}/s,\;8,\;220)$ px —
  effective network diameter;
- $f_{\text{flow}}$ — `flow_threshold` (default 0.70), the maximum tolerated flow
  error; higher accepts more irregular boundaries;
- $p_{\text{cell}}$ — `cellprob_threshold` (default −1.0); negative values raise
  recall for faint spheres;
- $\mathcal{S}_1$ — label image of Stage-1 masks.

### 5.2 Stage 2 — dense agglomerate recovery

Inside intensely fluorescent agglomerates ($\hat{I}_{c0} \ge 0.6$) that Stage 1
left unlabelled, dark core cavities are recovered when

$$
C_{\text{core}} = \frac{\overline{X}_{\text{rim}} - \overline{X}_{\text{core}}}
{\overline{X}_{\text{rim}}} \ge 0.20
\quad\text{and}\quad
\frac{A_{\text{rim}}}{A_{\text{core,hole-filled}}} \ge 0.50 ,
$$

and then split with marker-controlled watershed seeded by Euclidean
distance-transform maxima separated by at least $d_{\min}$.

**Purpose:** measure small spheres ($1\text{–}4\,\mu\text{m}$) that sit inside a
bright shell agglomerate and are below the receptive field of the network.

**Where:**

- $\overline{X}_{\text{core}}$, $\overline{X}_{\text{rim}}$ — mean core and rim
  intensity of the candidate cavity;
- $A_{\text{rim}}$, $A_{\text{core,hole-filled}}$ — rim band area and the
  hole-filled cavity area;
- $d_{\min}$ — minimum Euclidean peak separation used by the watershed seeds
  (pixels).

### 5.3 Stage 3 — post-segmentation cluster splitting

Any mask that still looks like two or more touching spheres is partitioned:

$$
\text{split if } C < C_{\min} \;\lor\; S < S_{\min},
\qquad
\text{seeds} = \operatorname{localmax}\!\big(\operatorname{EDT}(\text{mask})\big),\;
\Delta_{\text{seed}} \ge d_{\min}.
$$

**Purpose:** restore the physical primary-particle population from merged
doublets without inventing particles in genuinely non-spherical debris.

**Where:**

- $C_{\min}$ — 🟣 Microsphere Min Circularity (default 0.55);
- $S_{\min}$ — solidity gate applied to the filled mask;
- $\operatorname{EDT}$ — Euclidean distance transform of the mask;
- $\Delta_{\text{seed}}$ — enforced minimum distance between two watershed
  seeds.

### 5.4 Stage 4 — "Matryoshka" de-duplication

Candidate masks are sorted by descending area and a candidate is rejected when
more than 90 % of its hole-filled footprint is already covered by an accepted
larger mask.

**Purpose:** remove the concentric duplicates that Stage 1 and Stage 2 can both
produce around the same physical sphere, guaranteeing that no particle is
counted twice.

## 6. 📏 Morphometry

For every accepted mask with area $A$ (px²) and perimeter $P$ (px):

$$
d_{\text{eq}} = 2\sqrt{\frac{A}{\pi}}\cdot s,
\qquad
C = \frac{4\pi A}{P^2},
\qquad
S = \frac{A}{A_{\text{hull}}},
\qquad
AR = \frac{a_{\text{major}}}{a_{\text{minor}}} .
$$

**Purpose:** describe each sphere by a size, a boundary-smoothness and two shape
descriptors, so the size distribution can be filtered to *physically spherical*
particles only.

**Where:**

- $d_{\text{eq}}$ — equivalent circular diameter in µm (the diameter of a circle
  with the same area);
- $C$ — circularity, $1$ for a perfect circle, $\approx 0.8$ for a compact
  sphere projected at an oblique angle, $< 0.6$ for crescents and shards;
- $A_{\text{hull}}$ — area of the convex hull of the mask;
- $a_{\text{major}}$, $a_{\text{minor}}$ — major and minor axes of the fitted
  ellipse.

Masks whose $C$, $S$ or $AR$ fall outside the configured gates are reported as
debris and are excluded from the size distribution, but remain visible in the
rejected-object log for audit.

## 7. 📊 Granulometry

With $d_i$ the equivalent diameter of the $i$-th accepted sphere:

$$
D[3,2] = \frac{\sum_i d_i^{3}}{\sum_i d_i^{2}},
\qquad
D[4,3] = \frac{\sum_i d_i^{4}}{\sum_i d_i^{3}},
\qquad
\text{Span} = \frac{D_{90} - D_{10}}{D_{50}} .
$$

**Purpose:** summarise the distribution with the two engineering means used in
emulsion and colloid science (surface-weighted Sauter and volume-weighted De
Brouckere) plus a scale-free width, so results are comparable with laser
diffraction and sieving data.

**Where:**

- $D[3,2]$ — Sauter mean diameter, the diameter of the monodisperse population
  with the same specific surface area;
- $D[4,3]$ — De Brouckere (volume) mean diameter;
- $D_{10}, D_{50}, D_{90}$ — the diameters below which 10 %, 50 % and 90 % of the
  measured spheres lie;
- $\text{Span}$ — dimensionless width; $\text{Span} < 1$ is a narrow
  (monodisperse) batch.

Because $\sum d^4/\sum d^3 \ge \sum d^3/\sum d^2$ by Cauchy–Schwarz, the
inequality $D[3,2] \le D[4,3]$ is a built-in correctness test.

## 8. 🍇 Cluster, Aggregation and Fractal Analytics

Spheres are grouped into clusters by centroid proximity and mask dilation. For a
cluster of $N_{agg}$ primary particles with areas $A_i$ at positions
$\mathbf{r}_i$:

$$
\mathbf{r}_{cm} = \frac{\sum_i A_i \mathbf{r}_i}{\sum_i A_i},
\qquad
R_g = \sqrt{\frac{\sum_i A_i\,\lVert \mathbf{r}_i - \mathbf{r}_{cm}\rVert^2}{\sum_i A_i}} .
$$

**Purpose:** provide an area-weighted (intensity-proxy) centre of mass and radius
of gyration, so a large sphere inside a floc cannot dominate the geometry the
way an unweighted centroid would.

**Where:**

- $\mathbf{r}_i$ — centroid of particle $i$ in µm;
- $A_i$ — area of particle $i$ in µm², used as the mass proxy;
- $\mathbf{r}_{cm}$ — area-weighted centre of mass of the cluster;
- $R_g$ — radius of gyration in µm.

The 2D fractal dimension is the slope of the power law

$$
N_{agg} \propto \left(\frac{R_g}{r_0}\right)^{D_f}
\quad\Longrightarrow\quad
D_f = \frac{\mathrm{d}\log N_{agg}}{\mathrm{d}\log (R_g/r_0)} ,
$$

fitted over all multi-particle clusters, with $r_0$ the mean primary-particle
radius.

**Purpose:** quantify how compactly the silica network packs; $D_f \to 2$ means
space-filling (dense) flocs, $D_f \to 1$ means chain-like, open aggregates.

**Where:**

- $N_{agg}$ — number of primary particles per cluster;
- $r_0$ — mean primary-particle radius of the measured population (µm);
- $D_f$ — fitted slope; reported as $0$ when fewer than three multi-particle
  clusters exist, because the fit is then statistically meaningless.

## 9. 🎯 Calibration and Uncertainty Propagation

All linear metrics scale with the first power of $s$ and all areas with its
square:

$$
\frac{\delta d}{d} = \frac{\delta s}{s},
\qquad
\frac{\delta A}{A} = 2\,\frac{\delta s}{s},
\qquad
\frac{\delta R_g}{R_g} = \frac{\delta s}{s} .
$$

**Purpose:** make the sensitivity of each reported quantity explicit, so a
reviewer can bound a result from the objective's calibration tolerance.

**Where:**

- $\delta s$ — uncertainty of the spatial sampling (from the scale-bar or
  stage-calibration tolerance);
- $\delta d$, $\delta A$, $\delta R_g$ — induced uncertainties of a size, an
  area and a radius of gyration.

A 1 % calibration error therefore produces a 1 % size error but already a 2 %
area error; the pixel-domain shape metrics ($C$, $S$, $AR$, $D_f$) are
dimensionless and unaffected.

## 10. ⚠️ Assumptions and Limitations

1. **Planar projection.** Spheres are measured on a 2D projection; a sphere whose
   equator lies outside the optical section is underestimated by
   $\sqrt{1 - (z/R)^2}$.
2. **Optical resolution floor.** Particles below ≈ 3 px in diameter cannot be
   measured reliably; the Stage-2 catcher extends the usable range but never
   below the diffraction limit of the objective.
3. **One optical section.** Differential z-drift during the scan is not
   corrected; acquisitions with visible focus drift must be discarded.
4. **Linear fluorescence.** Saturated shells violate the intensity assumptions of
   the excess-shell subtraction.
5. **Cluster definition.** The dilation radius used to link spheres defines a
   cluster; changing it changes $N_{agg}$ and therefore $D_f$, so the value must
   be reported with the data.
6. **No stereo correction for touching spheres.** Overlapping projections are
   separated by watershed, which can under-split strongly sintered necks.

## 11. 🔁 Reproducibility Contract

A PhasePix result is reproducible when the following are archived together: the
source image, its SHA-256, the build identity (`PHASEPIX_BUILD_VERSION`,
`PHASEPIX_BUILD_DATE`), the calibration preset, the full parameter snapshot and
the exported tables. The JSON session manifest written in
**💾 Reproducibility & Export** contains all six.

## 12. 📚 References

1. Stringer, C. et al. *Cellpose: a generalist algorithm for cellular
   segmentation.* Nature Methods 18, 100–106 (2021).
2. Weaire, D. & Hutzler, S. *The Physics of Foams.* Oxford University Press
   (1999) — granulometry conventions.
3. ISO 13320:2020 — *Particle size analysis, laser diffraction methods*
   (Sauter / De Brouckere means, span definition).
4. Russ, J. C. *The Image Processing Handbook*, 7th ed. — circularity, solidity
   and convexity definitions.
5. Meakin, P. *Fractal aggregates.* Advances in Colloid and Interface Science 28,
   249–331 (1988) — $N \propto (R_g/r_0)^{D_f}$ scaling.
6. Gonzalez, R. C. & Woods, R. E. *Digital Image Processing*, 4th ed. — CLAHE
   and distance-transform watershed segmentation.

---

*This methodology is mirrored by the downloadable Word document
[`docs/Methodology_CLSM.docx`](Methodology_CLSM.docx). Editing that Word file by
hand makes it authoritative: PhasePix serves your version and will not
regenerate it from the Markdown.*



