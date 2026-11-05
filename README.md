# PhasePix: Quantitative SEM Microstructure Analyzer

PhasePix is a high-performance Streamlit application designed for automated deep-learning segmentation and morphological phase analysis of Scanning Electron Microscopy (SEM) micrographs (e.g., distinguishing air bubble pores from fat globules in food matrices and material science).

---

## Key Features

> **Current build:** `📦 Build: 2026.11.04 (v1.8.0 - Dual-Mode Interfacial Analytics (Coverage vs Corona θ))` — the exact string shown in the sidebar badge; the session manifest reports its `PhasePix Build Version` (`v1.8.0`) next to the source-image SHA-256.

- **Interior Crater Coverage vs. Corona Contact Angle Split (`analyzer.py`, `app.py`, v1.8.0)**: `compute_interfacial_adsorption_metrics()` now keeps the two globule populations of a cryo-SEM freeze-fractured bubble strictly apart, because they sit on two different surfaces. **Interior adsorbed globules** (`is_corona_globule == False`) sit on the inner crater floor and are the *only* population behind the 2D areal metrics: `Surface Coverage (%) = |Ω_b ∩ ⋃ interior globule masks| / |Ω_b| · 100` and `Packing Density (glob/µm²) = n_interior / Bubble Area` — per bubble and in `ads_summary` (`Overall Surface Coverage (%)`, `Mean Packing Density (glob/µm²)`, `Overall Packing Density (glob/µm²)`). **Corona rim globules** (`is_corona_globule == True`) sit on the fractured equatorial perimeter cross-section and are the *only* population behind the three-phase contact angle `θ_i = arcsin(clip(L_c,i / D_i, 0, 1))` and `Rim Coverage (%)`, measured on the corona union inside the equatorial rim band. A corona globule straddles the rim, so counting it as crater area used to inflate Surface Coverage / Packing Density with particles that never sit on the crater floor; the split is now enforced inside the analyzer instead of relying on the UI. `contact_angle_mode` is locked to `CONTACT_ANGLE_MODE_INDIVIDUAL` — the retired global `D_ref` recipe is still *accepted* (so old scripts keep running) but ignored, so a legacy caller can never re-mix interior globules into θ. The per-globule table gains `used_for_surface_coverage`, the per-bubble table reports `Interior Adsorbed Globules`, `Interior Adsorbed Area (µm²)`, `Surface Coverage (%)`, `Packing Density (glob/µm²)`, `Corona Globules`, `Corona Valid for θ`, `Excluded from θ`, `Rim Coverage (%)`, `Mean Contact Angle θ (°)`, `Mean Corona Chord (µm)`, `Mean Corona Globule Diam (µm)` and `Total Adsorbed Globules`, and the ambiguous legacy columns (`Adsorbed Count`, `Corona Count`, `Interior Count`, `Adsorbed Area (µm²)`, `Mean Ads Globule Diam (µm)`) are gone.
- **🔬 2-Mode Inspection Switch & Role-Filtered Overlays (`app.py`, `visualizer.py`, v1.8.0)**: The `Contact Angle θ Geometry / Reference Globule Diameter` radio is gone; in its place the top of **🛡️ Interfacial Analytics** carries `🔬 Interfacial Inspection Mode:` with two options — `🫧 Surface Coverage Mode (Interior Crater Globules)` and `👑 Contact Angle θ Mode (Equatorial Corona Globules)`. `render_selected_bubbles_inspection_overlay()` gained `globule_role` (`GLOBULE_ROLE_ALL` / `GLOBULE_ROLE_INTERIOR` / `GLOBULE_ROLE_CORONA`, normalised by `normalize_globule_role()`) and `show_globule_ids`: in **coverage mode** the cyan bubble contour is drawn together with **only** the interior crater globules in gold (corona rim globules are hidden, the gold mask is clipped strictly to Ω_b, and `🔢 Show Particle IDs on Image` toggles their bare particle IDs — no θ labels), and the crop-inspector table lists that bubble's interior globules (`particle_id`, `diameter_um`, `area_um2`, `circularity`); in **θ mode** only the equatorial corona globules are drawn in gold (interior globules are hidden, the corona rim band is kept so rim-straddling globules are complete), `Show Contact Angle Labels (°) on Image` and `🏷️ Show Particle ID next to θ (e.g. #142: 68°)` annotate them, and the interactive `Exclude from θ` editor (`render_theta_exclusion_editor()`, `st.data_editor`) is fed **strictly the corona globules (`is_corona_globule == True`)** of the inspected bubble so fractured/incomplete rim particles can be excluded before `✅ Apply θ Exclusions for Bubble #N`. The annotation pass is role-aware as well, so a coverage overlay can never print a θ and a θ overlay can never label an interior globule; `GLOBULE_ROLE_ALL` keeps the historic interior + corona rendering for legacy callers.
- **Five Interfacial KPI Cards (`app.py`, v1.8.0)**: The tab now opens with exactly five cards — `Selected Bubbles`, `Interior Globules (Coverage)` (delta `Corona rim globules: N`), `Crater Surface Coverage (%)` (delta `Interior globules only · Rim: X.X%`), `Crater Packing Density (glob/µm²)` (delta `interior globules per µm² of bubble area`) and `Mean Contact Angle θ (°)` (delta `Corona valid for θ: N · X excluded`) — so the areal crater metrics and the corona contact-angle statistics are separated at a glance instead of being mixed in one card.

- **Three-Phase Contact Angle on Corona Rim Globules Only (`analyzer.py`, `app.py`, v1.7.9)**: `compute_interfacial_adsorption_metrics()` now splits every adsorbed globule into a **corona rim** role (straddles / touches / overlaps the bubble perimeter — the globule centre lies within one own radius of the interface, or ≥ 15 % of its own area lies in the contact strip) and an **interior** role (fully inside the eroded crater, where only a 2D cap projection is visible). Surface coverage, rim coverage, packing density and the adsorbed count still include **all** adsorbed globules, but θ is now computed **exclusively for corona globules and with each globule's own geometry**: `D_i = max(major_axis_um, diameter_um)` and `L_c,i` = chord of the globule's own mask cut by the bubble boundary (measured inside a narrow tangent-line strip, falling back to the exact circle/plane cross-section `2√(R_i² − d_i²)` when the segmentation gap exceeds the strip), so `θ_i = arcsin(clip(L_c,i / D_i, 0, 1))` stays physically consistent in [0°, 90°]. The independent radial-protrusion estimator `θ_i = arccos(clip(d_i / R_i, 0, 1))` is reported in `contact_angle_protrusion_deg` with `protrusion_distance_um` for cross-checking. Interior globules now carry `contact_angle_deg = NaN`, so the NaN-aware mean / median / SD can never be dragged away from the real three-phase value; the new columns `is_corona_globule`, `globule_diam_um` and `corona_chord_um` document every number. The legacy `arcsin(L_c / D_ref)` recipe remained selectable (`contact_angle_mode="d_ref"`) for reproducing historical results — **retired in v1.8.0**, which locks θ to the individual-corona recipe and ignores the argument.
- **Interactive θ Exclusions in the Single Bubble Crop Inspector (`app.py`, `analyzer.py`, `visualizer.py`, v1.7.9)**: Cryo-SEM freeze-fracture leaves some corona globules cleaved off-centre or partially occluded, so their visible cross-section is not their true diameter. The static detail table under **🛡️ Interfacial Analytics → 🔍 Single Bubble Crop Inspector** is now an `st.data_editor` with a leading **`Exclude from θ`** checkbox plus `Particle ID (on crop)`, `Globule Role` (`👑 Corona` / `Interior`), `True Globule Ø (µm)`, `Chord L_c (µm)` and `θ (°)`, and two buttons — **`✅ Apply θ Exclusions for Bubble #N`** (stores the ticked IDs of that bubble in `st.session_state["theta_excluded_particle_ids_{image_name}"]`, keeping the exclusions of the other bubbles) and **`🔄 Reset θ Exclusions`**. Excluded IDs are passed to `compute_interfacial_adsorption_metrics(..., excluded_theta_particle_ids=...)`, flagged in the `excluded_from_theta` column and dropped from the mean, median, SD, the `n` of the distribution histogram and the inspection/crop **and dashboard** θ labels (drawn in dim grey instead of white). They remain `adsorbed_fat`, so Surface Coverage, Packing Density and the adsorbed count are untouched unless the particle is deleted globally in Tab 1 / Tab 2. A new `🏷️ Show Particle ID next to θ (e.g. #142: 68°)` toggle prefixes every label with the particle ID, so a glyph on the crop can be matched one-to-one with a row of the editor.
- **Recommended Corona Mode & Valid-θ Counts (`app.py`, `analyzer.py`, v1.7.9)**: The **🛡️ Interfacial Analytics** selector now leads with **`👑 Individual Corona Globule Size (True Profile — Recommended)`** as the default, followed by the three legacy reference options (`Mean Matrix Fat Diameter`, `Matrix Fat Sauter D[3,2]`, `Custom Value (µm)`); choosing any of them switches the analyzer back to `contact_angle_mode="d_ref"`. The KPI row shows `Total Adsorbed Globules` with a `Corona valid for θ: N · interior: M` delta and a θ card whose delta is `n = N valid · X excluded`, a caption line adds mean / median / SD, and the per-bubble table now carries `Corona Count`, `Corona Valid for θ`, `Interior Count` and `Excluded from θ` next to `Adsorbed Count`. *(Superseded in v1.8.0: the legacy reference options, the radio and those four legacy column names are gone — see the locked-methodology entries above.)*

- **Automatic Hardware Acceleration (v1.7)**: The fastest backend is selected at startup and passed straight to Cellpose and SAM — **NVIDIA CUDA** (`torch.cuda.is_available()`, with the exact GPU model shown in the sidebar badge via `torch.cuda.get_device_name(0)`), then **Apple Silicon Metal** (`torch.backends.mps.is_available()`), then a clean **CPU** fallback where every feature keeps working.
- **Platform-Aware CPU Threading Guard (v1.7.1 → v1.7.3)**: MKL / OpenMP / OpenCV oversubscription could freeze the **Windows CPU-only** backend for 10+ minutes on small images (e.g. 700×700) — even at `model.eval()` — so v1.7.1 / v1.7.2 capped every thread pool in the process. Because that also throttled macOS (Apple Silicon / MPS) and NVIDIA CUDA machines, `cpu_thread_guard.py` is now platform- and hardware-aware: on **macOS (`sys.platform == "darwin"`) and Linux** `apply_cpu_thread_env()`, `apply_opencv_thread_limits()` and `apply_torch_thread_cap()` are **complete no-ops** (no `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `NUMEXPR_NUM_THREADS` exports, no `cv2.setNumThreads(0)`, no `torch.set_num_threads` cap), and on **Windows with an NVIDIA CUDA GPU** (`torch.cuda.is_available()`) OpenCV is restored to its default multi-threaded pool (`cv2.setNumThreads(-1)`) while PyTorch receives `os.cpu_count()` intra-op threads. The strict guard — 4 threads exported to OMP / MKL / OpenBLAS / NumExpr plus `KMP_DUPLICATE_LIB_OK=TRUE` (defuses the Intel `libiomp5md.dll` vs. LLVM `libomp.dll` OpenMP conflict), `cv2.setNumThreads(0)` and `torch.set_num_threads(4)` — now runs **only** on Windows *without* CUDA, keeping the deadlock protection exactly where it is needed. A lower value exported through `PHASEPIX_CPU_THREADS` (or `OMP_NUM_THREADS`) is honoured on that path, `PHASEPIX_FORCE_THREAD_CAP=1` forces the strict cap anywhere for diagnostics, and every path logs the active mode at startup (calls still happen before `numpy` / `torch` are imported, from both `app.py` and `cellpose_model.py`).
- **Split Databar Scale-Bar Detection (`scale_detector.py`, v1.7.4)**: FEI / Thermo Fisher databars draw the scale bar as `|---- 10 µm ----|`, i.e. two collinear horizontal rays interrupted by the magnification label. v1.7.3 measured only one half on real 10 µm micrographs (237 px instead of the full ~557 px) because the left ray was clipped by the right-hand ROI window and discarded at the border. The detector now collects every horizontal run on the dominant baseline (±3 px), merges two collinear halves when the gap (typically 20–120 px) holds bright label glyphs (`_gap_has_label_text` + `_merge_split_bar_runs`), recovers a half clipped by the ROI edge by walking outward along the same baseline in the full image (`_recover_clipped_span`), and rejects full-width table dividers/full-width rules. The reported length is always measured from the outermost left tick to the outermost right tick, the green centerline and red end marks of the preview annotation span that full tick-to-tick width across the text gap, and a ray-density coverage check (`rays / span ≥ 0.25`) keeps sparse text rows and metadata table rules from being mistaken for a bar.
- **Label-Free Single Bubble Crop Inspector (`app.py`, `visualizer.py`, v1.7.4)**: The **🛡️ Interfacial Analytics** tab → **🔍 Single Bubble Crop Inspector** now exposes a `🔢 Show Labels / Contact Angles on Crop` checkbox right above the close-up (default synced with the *Show Contact Angle Labels (°) on Image* toggle). When unchecked, the crop is rendered from a label-free inspection overlay (`render_selected_bubbles_inspection_overlay(..., show_contact_angles=False, show_labels=False)`) that keeps only the cyan bubble boundary and the gold adsorbed-fat masks, so the adsorption corona and the completeness of the interfacial-particle identification can be verified without θ values / bubble ID text clutter. The caption marks the mode (`— close-up (labels hidden)`) and the `📥 Download Crop #N (PNG)` button exports exactly the image shown.
- **Phase Classifications ID Search / Filter (`app.py`, v1.7.4)**: The **📊 Interactive Dashboard** → **2. Phase Classifications** panel now exposes a compact `🔍 Search / Filter by Particle ID` box directly above the `st.data_editor` table. It accepts a single ID (`142`), a comma / semicolon / space separated list (`12, 45, 142`) or an inclusive range (`100-120`), shows a live `showing X of Y particles` caption, warns (instead of silently emptying the table) when no numeric ID can be parsed, and disables **Apply Changes** when nothing matches. Only the rows currently visible are written on apply (`_apply_particle_editor_changes`), so checking `Excluded` or changing `Phase Class` for a handful of filtered particles never resets the exclusions or phase overrides of the particles hidden by the filter; the `particle_data_editor` key carries a short filter signature so a different row subset can never receive positional edits from another view.
- **AI Parameter Advisor Honours the Active Sample Type (`app.py`, v1.7.4)**: `_diagnose_sem_micrograph()` / `_sem_comparison_rows()` no longer recommend or overwrite `sem_sample_type_radio` — the `Sample Type` row and key were removed from `recommended_state` / `comparison_rows`, so an explicit **Emulsion** or **Foam** choice can never be overridden by `🪄 Analyze Image & Recommend Parameters`. The proposal is now a pure function of `(pixels, calibration, sample_type)`: **Emulsion** gets only fat-particle bounds + Matrix Fat circularity, **Foam** only air-bubble bounds + Air circularity, and **Whipped Emulsion** keeps the full multi-phase recipe (`sem_detection_focus_radio`, air + fat bounds, air/adsorbed circularity, Deep-Scan). A `ℹ️ Reminder` banner in the advisor section (and inside the review card) asks the user to confirm the correct Sample Type before running, the card states which mode the proposal was computed for, and switching the Sample Type invalidates a stale proposal instead of applying a foreign phase recipe.
- **Pickering Corona Adsorbed-Fat Capture (`app.py`, `analyzer.py`, `visualizer.py`, v1.7.4)**: Corona (Pickering) fat globules sit ON the air-bubble boundary, so the segmented blue contour frequently runs along their inner edge and their geometric centroid ends up a few pixels *outside* the bubble mask — where the old centroid-only test silently demoted them to orange matrix fat. Sidebar **Section 5** now exposes `👑 Corona Rim Capture Distance (µm)` (0.0–5.0 µm, default 1.2 µm, key `sem_corona_Capture_um`, Shipped for SEM **Whipped Emulsion**). The cumulative bubble mask is dilated by `corona_px = max(2, round(corona_um / µm_per_px))` with an elliptical kernel, and a fat-range particle with `circularity ≥ Adsorbed Fat Min Circularity` becomes `in_bubble = True` / `adsorbed_fat` when **either** its centroid lies in that band **or** ≥ 15 % of its own segmented area (`masks == id`) overlaps it (`_corona_rim_flags`, two O(N) `bincount` passes). `compute_interfacial_adsorption_metrics(..., corona_margin_px=...)` applies the same rule per bubble, so corona globules are counted in **rim coverage**, **surface coverage**, **packing density** and the apparent **contact angle θ** of the bubble they touch, and the inspection overlay (`corona_margin_px`) no longer clips their gold masks at the rim.
- **Fast Localized ROI SAM for Smart Box (`cellpose_model.py`, `analyzer.py`, v1.7.5)**: `📦 AI Smart Box` no longer pushes the full high-resolution micrograph through the SAM ViT encoder for every rectangle. Each drawn box is served from a **padded local ROI** (box + 40 % per side, never below 256×256 px, clamped to the image) and the ROI-local mask is shifted back into full-image coordinates, so downstream contours are unchanged. Neighbouring rectangles reuse one cached embedding (`get_cached_sam_predictor_for_roi` only re-encodes when the requested ROI leaves the cached region) and `prefetch_sam_roi_union` encodes a single tight union ROI for several close boxes, turning N encoder passes into one. On Apple Silicon the ViT encoder now runs on **MPS** (≈2.2× faster than CPU) with an automatic, transparent CPU fallback if an op is unsupported. Measured on a real SEM frame (vit_b): full-frame embedding 0.40–1.23 s per box → **0.087 s** per additional box with a warm ROI embedding.
- **Stable Manual-Correction Canvas (`app.py`, v1.7.5)**: `st_canvas` used to be keyed with a version suffix, so every Add / Undo / Clear remounted the component and could flash a blank white canvas. The widget key is now **stable** (`manual_correction_canvas_{image}`) and a reset is requested through `initial_drawing={"version": "4.4.0", "objects": []}` carrying `pp_reset_rev` (the `manual_canvas_ver` revision), so fabric clears only the drawn objects while the micrograph background bitmap — and its Streamlit media URL — stays mounted. The background is also normalised to a clean 3-channel RGB PIL image.
- **Point-and-Click Group Phase Reclassifier (`app.py`, v1.7.5)**: A fourth Selection Method — `🎨 Click to Reclassify Existing Particles (Group Click Mode)` — paints every segmented particle on the canvas (green Unclassified emphasised, orange fat, gold adsorbed fat, sky-blue air bubbles) and lets the user pick a target in `Assign Clicked Particles As` (`adsorbed_fat`, `fat_particles`, `air_bubbles`, `Unclassified`, `🗑️ Delete (Exclude)`; Emulsion mode offers fat/Unclassified/Delete only). Clicks are drawn as dots in the target colour and committed by `✅ Apply Phase to Clicked Particles`: `_resolve_clicked_particles` maps each click through `masks[py, px]` (plus a 3×3 window), then `cv2.pointPolygonTest` against saved manual polygons, then the nearest `classified_df` centroid within `0.6 × diameter_px`. Matches are written to `manual_phase_overrides` (or `interactive_excluded_ids` for Delete) with a confirmation flash.
- **Manual `adsorbed_fat` Overrides Are Final (`app.py`, v1.7.5)**: Rule 4 no longer deletes a researcher's `adsorbed_fat` assignment when the particle centroid lies outside the segmented bubble mask. `_respect_manual_adsorbed_overrides` keeps the override, forces `in_bubble = True` / `is_partial_in_bubble = True` (so the particle renders gold and enters the interfacial metrics) and applies the automatic "outside bubble" demotion strictly to non-overridden ids.
- **Self-Contained (Inline) Manual-Correction Canvas (`app.py`, visualizer.py, v1.7.6)**: `st_canvas` used to fetch the micrograph from Streamlit's `MediaFileManager` (`/media/<hash>.png`); those transient refs could be evicted or lose the asynchronous `<img>` load race after a tab switch, leaving a blank white box. The bitmap is now embedded as a self-contained `data:image/jpeg;base64,...` URI (`encode_image_data_url(..., max_width=0, quality=93)` at native 1:1, cached per background digest) inside the fabric `initial_drawing.backgroundImage` payload, so it is decoded from memory and can never expire; the widget key stays stable and entering `🖌️ Manual Correction` bumps `manual_canvas_ver`, forcing an immediate clean re-render. The media-URL background is kept as a belt-and-braces fallback, and the legacy raster fallback is disabled because the background now lives inside the canvas snapshot.
- **Phase Visibility Toggles for Click-to-Reclassify (`app.py`, `visualizer.py`, v1.7.6)**: Sidebar **Section 6** gained `Show Unclassified (Green)` (key `show_unclassified_toggle`), honoured by `create_mask_overlay(show_unclassified=...)` for the green underlay and its ID labels. Inside `🎨 Click to Reclassify` the sticky toolbar now shows a **Canvas Phase Visibility** row — `🟢 Show Unclassified`, `🟧 Show Matrix Fat`, `🟨 Show Adsorbed Fat`, `🟦 Show Air Bubbles` (seeded from the sidebar) — plus `🎯 Isolate Unclassified Only (Hide Fat & Air on Canvas)`, which paints only the green Unclassified particles onto a clean raw micrograph (`_canvas_phase_visibility`). All other Selection Methods keep their previous clean rendering.
- **Short Selection-Method Labels & Dashboard Quick Jump (`app.py`, v1.7.6)**: The four Manual-Correction modes are now `📦 AI Smart Box (SAM)`, `🎨 Click to Reclassify (Group Click)`, `📐 Exact Smooth Curve (Points)` and `🪄 1-Click Dark Pore`, so they fit on a single row (widget key `manual_assist_mode`). The dashboard **⚡ Smart Group Bulk Actions for Unclassified** block exposes `🎨 Click-to-Reclassify on Image`, which jumps straight to `🖌️ Manual Correction` with the Group Click mode pre-selected through deferred `_pending_navigation_tab` / `_pending_manual_mode` state.
- **Phase-Aware Sidebar Tools (`app.py`, v1.7.7)**: Section **5. Cellpose AI Segmentation** now shows only the controls that can act on the active Phase Classification mode. **Emulsion** renders just `🟧 Matrix Fat Min Circularity` (air-bubble, adsorbed-fat, Deep-Scan and Corona controls are hidden), **Foam** renders just `🫧 Air Bubbles Min Circularity` (matrix-fat, adsorbed-fat and Deep-Scan hidden), and **Whipped Emulsion** keeps the full multi-phase panel. Hidden controls keep working through their stored / canonical values (`st.session_state` → `PARAMETER_WIDGET_DEFAULTS`), so classification results, presets and the AI Parameter Advisor are unaffected; each trimmed panel shows a short ℹ️ caption explaining which tools are available.
- **Vectorised Label Loops & Exact Bubble-Source Counters (`cellpose_model.py`, v1.7.8)**: Three hot paths used to allocate a full-image (or full-ROI) boolean mask **per candidate label** — `masks_inv == inv_label` in the Step-A dual-pass merge, `cc_labels == cc_id` in the Step-B dark-pore fallback and `m_roi_2d == lbl` inside the fat deep-scan ROI loop — so merging L labels cost `L × image area` time and memory (~2 MB per candidate on a 1440×1440 micrograph). Every candidate is now evaluated strictly inside its tight bounding box: `scipy.ndimage.find_objects` gives the inverted-pass and deep-scan slices, `cv2.connectedComponentsWithStats` supplies each dark component's area/bbox in the same C pass (so out-of-range pores are rejected before a single pixel is touched), and `np.bincount` produces the deep-scan candidate-area and overlap-with-existing-fat tables in one vectorised pass. Measured on a 1440×1440 frame: Step-A merge **38.9×** faster, Step-B component filter **≈4000×** faster, deep-scan ROI loop **3.4×** faster, with masks bit-identical to v1.7.7 on every adversarial case (off-by-`None` label gaps, IoU/intensity gates, large-bubble downscale path). The Step-A/B log line now reports exact `direct` / `inv_added` / `fallback` counters instead of deriving them from `next_label`, which always printed `fallback=0` and folded every fallback bubble into `inv_added`.
- **⚡ Pipeline Runtime Profiler (`app.py`, v1.7.8)**: Every analysis run is timed with `time.perf_counter()` across the four core stages — `Decode & Crop`, `AI Segmentation` (Cellpose single/two-pass + bubble deep-scan), `Metrics & Classification` (metrics extraction, matryoshka pruning, phase classification) and `Overlay Render`. Because the sidebar is drawn before the pipeline executes, the badge is written into an `st.empty()` placeholder placed directly under the **Hardware Acceleration** badge and filled at the end of the run: `⚡ Total: X.XX s (AI: X.XX s · Metrics: X.XX s · Render: X.XX s)`, with a `⏱️ Pipeline Runtime` expander listing all four stages plus the total. The same numbers are exported in the `💾 Reproducibility & Export` manifest (`Pipeline Runtime (s)` and `Pipeline Stage Breakdown (s)`), so a slow run can be diagnosed without a profiler.
- **⚠️ QC Filter for Borderline & Unclassified Particles (`app.py`, v1.7.8)**: Tab 1 → **2. Phase Classifications** now offers `⚠️ QC Filter: Show Only Borderline / Unclassified Particles` (key `qc_borderline_filter_{image}`) next to the `🔍 Search / Filter by Particle ID` box. It narrows the editable table to the objects that actually need a human look: `Unclassified` particles, particles whose `circularity` lies within **+0.05** of their own phase's minimum circularity threshold (borderline shape), and particles whose diameter lies within **10 %** of the active Min/Max size boundary of their phase (borderline size) — all computed by the vectorised `_qc_borderline_mask`. The filter only restricts the *editable view*: its signature is appended to the data-editor key so the widget state of the filtered subset never leaks into the full table, and `_apply_particle_editor_changes()` still writes **only the visible rows**, so exclusions and phase overrides of hidden particles are preserved.
- **SHA-256 Provenance & JSON Session Manifest (`app.py`, v1.7.8)**: The `💾 Reproducibility & Export` manifest now carries `Source Image SHA-256` (`hashlib.sha256(image_bytes).hexdigest()` of the exact analysed file) and `PhasePix Build Version`, and a new `📥 Download Manifest (.json)` button sits next to `📥 Download Manifest (.txt)` to export the very same dict as pretty-printed JSON (`phasepix_session_manifest_{image}.json`) for a LIMS, notebook or supplementary archive. Together with the runtime profile this pins each reported number to a verifiable image byte-stream and a specific PhasePix build.

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
├── cpu_thread_guard.py               # Platform/hardware-aware CPU thread guard (Windows CPU-only fallback)
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
