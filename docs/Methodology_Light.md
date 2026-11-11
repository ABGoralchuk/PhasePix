# 📐 Methodology — Transmitted-Light Droplet Quantification

> **Document:** `docs/Methodology_Light.md` · **Downloadable Word twin:** `docs/Methodology_Light.docx`
> **Companion procedure:** [`SOP_Light.md`](SOP_Light.md)
> **Modality:** 💡 Light (transmitted-light / brightfield) Microscopy
> **Dispersed system:** 🟧 Fat droplets — 💧 liquid or ❄️ crystallised
> **Revision:** 2.0.1 · **Validated on build:** v2.0.0 and later

## 1. 🎯 Purpose and Scope

This document specifies how PhasePix converts a transmitted-light micrograph of
an emulsion into a droplet size distribution: intensity normalisation, optional
brightfield inversion, edge-preserving crystal smoothing, multi-scale neural
segmentation, solidity/circularity gates, granulometry and uncertainty
propagation. It is the reference for every light-mode control in the sidebar.

Transmitted-light imaging is the hardest of the three modalities because the
contrast mechanism is **refraction and absorption along the optical path**, not
a physical interface signature: droplet edges are bright or dark rims whose
position depends on focus, aperture and refractive-index difference. The
methodology below is therefore built around shape gates rather than intensity
thresholds.

## 2. 🔤 Symbols and Notation

| Symbol | Meaning | Unit |
| --- | --- | --- |
| $I_0$ | incident (background) intensity | counts |
| $I(x,y)$ | transmitted intensity at a pixel | counts |
| $\hat{I}$ | normalised intensity | 0–255 |
| $s$ | spatial sampling | µm/px |
| $d_{\text{eff}}$ | effective Cellpose diameter | px |
| $A$ | mask area | px² / µm² |
| $P$ | mask perimeter | px / µm |
| $C$ | circularity | – |
| $S$ | solidity (convexity) | – |
| $AR$ | aspect ratio | – |
| $\sigma_s$ | bilateral spatial sigma | px |
| $\sigma_r$ | bilateral range sigma | grey levels |
| $S_{\min}$ | solidity gate (profile dependent) | – |

## 3. 🖼️ Image Formation and Pre-Processing

### 3.1 Normalisation and optional inversion

$$
\hat{I} = 255\cdot\frac{I - I_{\min}}{I_{\max} - I_{\min}},
\qquad
\hat{I}_{\text{inv}} = 255 - \hat{I}\ \ (\text{if brightfield inversion is on}).
$$

**Purpose:** remove lamp intensity and exposure differences so one set of gates
is valid across sessions, and let the same pipeline accept both "bright droplets
on a dark background" and "dark droplets on a bright background" geometries.

**Where:**

- $I$ — raw transmitted intensity;
- $I_{\min}$, $I_{\max}$ — global extrema of the frame;
- $\hat{I}$ — normalised plane, droplets bright after inversion;
- the inversion checkbox is off by default because most transmitted-light
  exports already show dark rims on a bright field, which the network segments
  correctly without inversion.

### 3.2 Edge-preserving crystal smoothing (❄️ profile only)

Internal crystal needles are high-frequency ridges that a segmentation network
reads as object boundaries, splitting one globule into many fragments. PhasePix
applies a bilateral filter that smooths *within* a grey level but keeps *across*
strong edges:

$$
I_s(p) = \frac{1}{W_p}\sum_{q\in\Omega}
I(q)\,G_{\sigma_s}\!\left(\lVert p-q\rVert\right)\,
G_{\sigma_r}\!\left(\left|I(p)-I(q)\right|\right),
\qquad
W_p = \sum_{q\in\Omega} G_{\sigma_s}G_{\sigma_r}.
$$

**Purpose:** erase the crystal texture inside a globule while preserving the
globule's true outer boundary, which is the only boundary that carries size
information.

**Where:**

- $p$ — the pixel being smoothed, $q$ — its neighbours in window $\Omega$;
- $G_{\sigma_s}$ — Gaussian weight on spatial distance (kernel radius in px);
- $G_{\sigma_r}$ — Gaussian weight on intensity difference (grey levels);
- $W_p$ — normalisation so a flat region keeps its intensity exactly;
- $I_s$ — smoothed plane handed to the network.

The sidebar exposes this as an integer strength 0–4 that maps onto the pair
$(\sigma_s,\sigma_r)$; **0 disables the filter**, and 2–3 is the recommended
range for crystallised fat.

### 3.3 Multi-scale globule detection

For crystallised samples the frame is segmented twice and the results merged:

$$
\mathcal{S} = \operatorname{dedup}\!\left(\mathcal{S}_{\text{coarse}} \cup \mathcal{S}_{\text{fine}}\right),
$$

where $\mathcal{S}_{\text{coarse}}$ uses a large target diameter (large
globules, whole) and $\mathcal{S}_{\text{fine}}$ a small one (small globules),
and $\operatorname{dedup}$ removes a mask when more than 90 % of its area is
already covered by a larger accepted mask.

**Purpose:** a single diameter cannot capture a population that spans one order
of magnitude without either fragmenting large globules or missing small ones;
the merge keeps both extremes while guaranteeing that no droplet is counted
twice.

**Where:**

- $\mathcal{S}_{\text{coarse}}$ — masks from the large-diameter pass;
- $\mathcal{S}_{\text{fine}}$ — masks from the small-diameter pass;
- $\operatorname{dedup}$ — containment-based de-duplication (descending area
  order).

### 3.4 Bounded downscale

Very large frames are downscaled so that one inference pass stays inside a fixed
pixel budget; the scale factor is carried into every measurement:

$$
X_{\downarrow} = \operatorname{resize}\!\left(X,\;1/k\right),
\qquad
d\,[\mu\text{m}] = s\,k\,d_{\downarrow}\,[\text{px}] .
$$

**Purpose:** keep inference time bounded on large camera frames without ever
changing the physical calibration of the reported diameters.

**Where:**

- $k$ — integer downscale factor actually applied (1 when the frame fits the
  budget);
- $d_{\downarrow}$ — diameter measured on the downscaled plane;
- $s$ — calibration of the original frame.

## 4. 🧠 Segmentation

$$
\mathcal{S} = \operatorname{Cellpose}\!\big(I_{\text{pre}},\, d_{\text{eff}},\, f_{\text{flow}},\, p_{\text{cell}}\big),
\qquad
d_{\text{eff}} = \operatorname{clip}\!\left(\frac{d_{\text{target}}}{s},\,12,\,220\right).
$$

**Purpose:** obtain droplet candidates from a generalist model whose only
physical input is the droplet scale the operator declared, so the result does not
depend on a hand-tuned edge threshold that would have to be re-tuned per
microscope.

**Where:**

- $I_{\text{pre}}$ — normalised (and, for ❄️, bilateral-smoothed) plane;
- $d_{\text{target}}$ — typical droplet diameter entered in µm (1–40 µm);
- $s$ — calibration in µm/px;
- $d_{\text{eff}}$ — effective network diameter in pixels (12–220 px);
- $f_{\text{flow}}$ — `flow_threshold` (default 0.70), the tolerated boundary
  flow error;
- $p_{\text{cell}}$ — `cellprob_threshold` (default −1.0), the object-probability
  floor.

## 5. 🏷️ Shape Gates

A candidate becomes a measured droplet only when all three gates pass:

$$
C = \frac{4\pi A}{P^2} \ge C_{\min},
\qquad
S = \frac{A}{A_{\text{hull}}} \ge S_{\min},
\qquad
AR = \frac{a_{\text{major}}}{a_{\text{minor}}} \le AR_{\max}.
$$

**Purpose:** remove the three artefact families that transmitted light produces —
thin fragments (low $C$), crescents of ruptured droplets (low $S$) and flattened
droplets under coverslip pressure (high $AR$) — without ever touching the true
droplet population.

**Where:**

- $C_{\min}$ — 🟧 Matrix Fat Min Circularity, default 0.60;
- $S_{\min}$ — 🛡️ Matrix Fat Min Solidity, default 0.72 for 💧 liquid fat and
  0.84 for ❄️ crystallised fat;
- $AR_{\max}$ — aspect-ratio ceiling applied to the fitted ellipse;
- $A_{\text{hull}}$ — convex hull area of the mask;
- $a_{\text{major}}, a_{\text{minor}}$ — fitted-ellipse axes.

Because crystallised droplets contain non-convex internal texture, their
*pixel* mask is less convex than a liquid droplet's even when the physical
droplet is perfectly spherical; that is exactly why the solidity default is
raised for the ❄️ profile instead of being lowered to admit fragments.

## 6. 📏 Morphometry

$$
d_{\text{eq}} = 2\sqrt{\frac{A}{\pi}}\,s,
\qquad
A\,[\mu\text{m}^2] = A\,[\text{px}^2]\cdot s^2,
\qquad
P\,[\mu\text{m}] = P\,[\text{px}]\cdot s .
$$

**Purpose:** convert mask geometry into the physical size and area that the
granulometry and the exported tables require.

**Where:**

- $d_{\text{eq}}$ — equivalent circular diameter (µm);
- $A$, $P$ — mask area and perimeter, in pixels and in micrometres;
- $s$ — calibration (µm/px).

## 7. 📊 Granulometry

With $d_i$ the equivalent diameter of the $i$-th accepted droplet:

$$
D[3,2] = \frac{\sum_i d_i^{3}}{\sum_i d_i^{2}},
\qquad
D[4,3] = \frac{\sum_i d_i^{4}}{\sum_i d_i^{3}},
\qquad
\text{Span} = \frac{D_{90} - D_{10}}{D_{50}} .
$$

**Purpose:** express the distribution with the same surface- and volume-weighted
means used by laser diffraction so light-microscopy and diffraction results for
the same emulsion are directly comparable, plus a dimensionless width.

**Where:**

- $D[3,2]$ — Sauter mean diameter (specific-surface equivalent);
- $D[4,3]$ — De Brouckere volume mean;
- $D_{10}, D_{50}, D_{90}$ — decile diameters from the empirical CDF;
- $\text{Span}$ — relative width; $< 1$ is narrow, $> 2$ is broad or bimodal.

The inequality $D[3,2] \le D[4,3]$ is guaranteed mathematically and is surfaced
in the UI as a correctness check; a violation indicates a corrupted table, never
a physical result.

## 8. 🍇 Coalescence Indicators

Droplets that share a mask boundary or overlap after dilation form a coalescence
group. For each group of $n \ge 2$ droplets:

$$
\bar{d}_{\text{group}} = \frac{1}{n}\sum_{i=1}^{n} d_i,
\qquad
\rho_{\text{coal}} = \frac{N_{\text{groups}}}{N_{\text{droplets}}} .
$$

**Purpose:** quantify how much of the population is already coalesced, which is
the practical stability indicator for a stored emulsion, without requiring
time-resolved acquisitions.

**Where:**

- $d_i$ — equivalent diameter of droplet $i$ inside the group;
- $N_{\text{groups}}$ — number of multi-droplet groups;
- $N_{\text{droplets}}$ — total number of accepted droplets;
- $\rho_{\text{coal}}$ — dimensionless coalescence fraction ($0$ = fully
  dispersed, $\to 1$ = heavily coalesced).

A rising $\rho_{\text{coal}}$ together with a rising $D[4,3]$ is the signature of
ongoing coalescence; a rising $D[4,3]$ at constant $\rho_{\text{coal}}$ indicates
Ostwald ripening or a formulation change instead.

## 9. 🎯 Uncertainty Propagation

$$
\frac{\delta d}{d} = \frac{\delta s}{s},
\qquad
\frac{\delta A}{A} = 2\,\frac{\delta s}{s},
\qquad
\delta \text{Span} \approx \frac{\delta s}{s}\cdot\frac{D_{90}+D_{10}}{D_{50}} .
$$

**Purpose:** state how a calibration error moves each reported quantity. Because
all lengths scale linearly with $s$, a systematic offset cancels in the
dimensionless shape metrics ($C$, $S$, $AR$) but not in the size distribution.

**Where:**

- $\delta s/s$ — relative calibration uncertainty of the objective preset;
- $\delta d$, $\delta A$ — induced size and area uncertainties;
- $\delta\text{Span}$ — worst-case width uncertainty, which is the most
  calibration-sensitive reported number.

Additional, non-calibration uncertainty sources are the segmentation boundary
(≈ 1 px per side, i.e. ≈ 2 px per diameter) and the subjective choice of gates;
both are documented by archiving the parameter snapshot with the data.

## 10. ⚠️ Assumptions and Limitations

1. **Projection of a 3D droplet.** A droplet outside the focal plane is blurred
   symmetrically and measured with an inflated diameter; only in-focus droplets
   are valid.
2. **Refractive-index contrast.** A droplet whose refractive index matches the
   continuous phase is invisible, so the measured population is biased against
   such droplets.
3. **Coverslip flattening.** Compressed droplets violate sphericity; they are
   rejected rather than corrected.
4. **Crystal texture.** Smoothing removes internal structure but can also soften a
   genuinely irregular boundary — always confirm the ❄️ profile visually.
5. **Multi-scale merge order.** The coarse pass wins containment; a small droplet
   whose mask is fully inside a large (incorrectly merged) mask disappears.
6. **Bounded downscale.** When $k > 1$, droplets smaller than about 3 px on the
   downscaled plane fall below the measurable limit.

## 11. 🔁 Reproducibility Contract

Archive the raw micrographs, the objective preset name and its µm/px value, the
emulsion profile, the smoothing strength, the gate values, the image SHA-256 and
the build identity. These items are all captured in the JSON session manifest, so
the exact droplet table can be regenerated without relying on memory or on the
current sidebar state.

## 12. 📚 References

1. Stringer, C. et al. *Cellpose: a generalist algorithm for cellular
   segmentation.* Nature Methods 18, 100–106 (2021).
2. Tomasi, C. & Manduchi, R. *Bilateral filtering for gray and color images.*
   ICCV 1998, 839–846.
3. ISO 13320:2020 — *Particle size analysis, laser diffraction methods.*
4. McClements, D. J. *Food Emulsions: Principles, Practices and Techniques*,
   3rd ed. — droplet-size measurement and coalescence.
5. Russ, J. C. *The Image Processing Handbook*, 7th ed. — solidity, convexity
   and perimeter measurement.
6. Zuiderveld, K. *Contrast limited adaptive histogram equalization.* Graphics
   Gems IV, 474–485 (1994) — why CLAHE is disabled by default here.

---

*This methodology is mirrored by the downloadable Word document
[`docs/Methodology_Light.docx`](Methodology_Light.docx). Editing it by hand makes
your Word file authoritative — PhasePix serves it and will not regenerate it from
the Markdown.*


