# Naming Convention: Render Outputs & Metrics

> Reference for interpreting render output names and metric names across experiments.
> Updated 2026-03-23 — rename from legacy names to standardized scheme.

## Render Outputs

| Output key | Description | When available |
|-----------|-------------|----------------|
| `clean_rgb` | Scene render without water effects (Gaussians + learned background) | Always |
| `medium_rgb` | Scene render through water medium (attenuation + backscatter applied) | When SeaThru is active |
| `observed_rgb` | Full observation including marine snow: `medium_rgb + S_k` | When marine snow modelling is enabled (future) |
| `direct` | Attenuated scene signal only (no backscatter): `clean_rgb * T(z)` | When SeaThru is active |
| `backscatter` | Backscatter component only: `B(z)` | When SeaThru is active |
| `attenuation_map` | Per-pixel attenuation values: `T(z)` | When SeaThru is active |
| `depth` | Processed depth map | Always |
| `accumulation` | Alpha/opacity accumulation | Always |

### Internal output keys (not typically rendered, but exist in `get_outputs()`)

| Output key | Description | Relationship to render outputs |
|-----------|-------------|-------------------------------|
| `rgb` | Raw splatfacto Gaussian rasterization output — **no background compositing** | Empty pixels are black (zero). Differs from `clean_rgb` where alpha < 1. |
| `image` | Gaussians + learned background: `rgb + learned_bg * (1 - alpha)` | **Same tensor as `clean_rgb`**. When `bg_from_backscatter` is True and SeaThru is active, background compositing is skipped so `image = rgb`. |
| `rendered_image` | Alias for raw rasterization output | Same as `rgb` |

### Rendering pipeline

```
Gaussians → rasterize → rgb (raw, black background)
                          │
                          + learned_bg * (1 - alpha)
                          │
                          ▼
                        clean_rgb (= image, with background fill)
                          │
                    medium model (T, B)
                          │
                          ▼
                        medium_rgb (underwater prediction)
                          │
                     snow model (future)
                          │
                          ▼
                        observed_rgb (full observation)
```

Each layer adds one physical effect. At inference, later layers can be dropped for cleaner renders.

**Note**: When `bg_from_backscatter` is True and SeaThru is active, the learned background step is skipped — the backscatter model fills in the water color at infinity instead. In this case `rgb = clean_rgb = image`.

## Metrics

### Primary metrics (nerfstudio compatibility)

| Metric | Computed against | Notes |
|--------|-----------------|-------|
| `psnr` | `medium_rgb` vs GT (when SeaThru active), else `clean_rgb` vs GT | Primary metric for nerfstudio eval pipeline |
| `ssim` | Same as psnr | Includes decomposition: `ssim_luminance`, `ssim_contrast`, `ssim_structure` |
| `lpips` | Same as psnr | Includes per-layer: `lpips_layer1` through `lpips_layer5` |

### Clean (out-of-medium) metrics

| Metric | Computed against | Notes |
|--------|-----------------|-------|
| `clean_psnr` | `clean_rgb` vs GT | Only logged when SeaThru is active (otherwise identical to primary) |
| `clean_ssim` | `clean_rgb` vs GT | Same |
| `clean_lpips` | `clean_rgb` vs GT | Same |

### Observed metrics (future — marine snow)

| Metric | Computed against | Notes |
|--------|-----------------|-------|
| `observed_psnr` | `observed_rgb` vs GT | Only when marine snow modelling is enabled |
| `observed_ssim` | `observed_rgb` vs GT | Diagnostic only — spatial mismatch expected |
| `observed_lpips` | `observed_rgb` vs GT | Diagnostic only |

### Cross-experiment comparison

- **Fair comparison across all experiments**: use `psnr` / `ssim` / `lpips` (the primary metrics)
  - When SeaThru is active, primary = medium (scene + water)
  - When SeaThru is inactive, primary = clean (scene only)
- **Decomposition analysis**: compare `clean_*` vs primary to measure medium model contribution
- **Marine snow analysis**: compare `observed_*` vs primary to measure snow model contribution

## Legacy Mapping

Experiments before 2026-03-23 use the old naming scheme:

| Legacy name | New name | Context |
|------------|----------|---------|
| `rgb` (render output) | `clean_rgb` | In `--rendered-output-names`, `RENDER_OUTPUT_NAMES`, render videos |
| `underwater_rgb` (render output) | `medium_rgb` | Same contexts |
| `rgb` (nerfstudio internal key) | `rgb` (unchanged) | Internal `outputs["rgb"]` from splatfacto base class — NOT renamed |
| `psnr` (metric) | `psnr` (unchanged) | Was always computed against the primary prediction |
| `clean_psnr` (metric) | `clean_psnr` (unchanged) | Was already named this way |
| `ssim`, `lpips` (metrics) | Unchanged | Were always primary metrics |
| NEW: `clean_ssim`, `clean_lpips` | — | Did not exist before this change |

### Important: legacy `rgb` renders ≠ `clean_rgb`

Legacy experiments rendered `outputs["rgb"]` (raw Gaussian output, black background). The new `clean_rgb` renders `outputs["image"]` (with learned background compositing). The difference is only visible where alpha < 1 (scene edges, sparse coverage). In well-reconstructed regions they are identical. When `bg_from_backscatter` is True and SeaThru is active, they are the same tensor.

The **metrics** were always computed against `outputs["image"]` (now `clean_rgb`), so `clean_psnr` values are consistent across legacy and new experiments. Only the visual renders differ slightly.

### Interpreting legacy experiment results

- Legacy `psnr` = current `psnr` (same computation, same name)
- Legacy `clean_psnr` = current `clean_psnr` (same computation, same name)
- Legacy renders named `rgb/` or `rgb.mp4` ≈ current `clean_rgb` (minor background difference, see above)
- Legacy renders named `underwater_rgb/` or `underwater_rgb.mp4` = current `medium_rgb`
- Legacy results do NOT have `clean_ssim` or `clean_lpips` — these are new metrics

## TensorBoard Training Metrics

> Tags logged during training via `get_metrics_dict()`. All prefixed with `Train Metrics Dict/` in TensorBoard.
> Updated 2026-03-30 — added decomposition activity metrics.

### Base metrics (from nerfstudio splatfacto)

| Tag | Source | Meaning |
|-----|--------|---------|
| `psnr` | L1/L2 vs GT | Training PSNR — uses `medium_rgb` vs GT when SeaThru active, else `clean_rgb` |
| `gaussian_count` | Gaussian model | Number of active Gaussians (tracks densification/pruning) |

### Medium model parameters

Logged when SeaThru is active (`seathru_from_iter` reached).

| Tag | Source | Meaning | Physical expectation |
|-----|--------|---------|---------------------|
| `bg_r/g/b` | `sigmoid(learned_bg)` | Learned background color (RGB) | Should match scene background |
| `binf_r/g/b` | `sigmoid(B_inf)` | Backscatter at infinity — open-water color | Should match far-field water color; channels < 0.5 |
| `bs_beta_r/g/b` | `backscatter_conv_params` | Backscatter depth coefficients (model parameter) | Positive; controls depth-falloff of scattering |
| `at_beta_r/g/b` | `attenuation_conv_params` | Attenuation β_D per channel (V3 model) | R > G > B ordering expected in water (red attenuates fastest) |
| `at_beta_eff_r/g/b` | `-log(T(z=1))` at reference depth | Effective attenuation at z=1.0 (V4 model) | Same R > G > B ordering; range 0.1–5.0 plausible |

### Gradient diagnostics

| Tag | Source | Meaning |
|-----|--------|---------|
| `grad_backscatter` | Gradient norm | Gradient magnitude flowing to backscatter model (recorded pre-nulling) |
| `grad_attenuation` | Gradient norm | Gradient magnitude flowing to attenuation model (recorded pre-nulling) |

### Clean render statistics

| Tag | Source | Meaning | Interpretation |
|-----|--------|---------|----------------|
| `clean_mean_r/g/b` | `mean(clean_rgb)` per channel | Channel means of the clean (out-of-medium) render | Low R relative to G = Gaussians memorized underwater color cast |
| `gw_weight_eff` | Anneal schedule | Effective gray world weight during GW annealing | Tracks anneal progress from start→end value |

### Decomposition activity metrics

Continuous proxies for whether the medium model is doing useful work. Logged in **all training phases**: zeros in Phase 1 (baseline), real values in Phase 2/3.

| Tag | Formula | Meaning | Interpretation |
|-----|---------|---------|----------------|
| `medium_contribution` | `mean(\|medium_rgb - clean_rgb\|) / mean(\|medium_rgb\|)` | Fraction of final image explained by medium | → 0: medium collapsed to identity; > 0.1: active decomposition. **The** continuous proxy for decomposition quality. |
| `attenuation_magnitude` | `mean(\|1 - T\|)` where T = attenuation map | Deviation of attenuation from identity | 0: no attenuation learned; higher values indicate depth-dependent structure |
| `backscatter_magnitude` | `mean(\|B\|)` where B = backscatter output | Image-space backscatter contribution | 0: no visible scattering; complementary to `bs_beta` which is just the parameter |

### Optional / conditional metrics

| Tag | Condition | Meaning |
|-----|-----------|---------|
| `snow_magnitude` | `use_marine_snow` enabled | Mean marine snow contribution per pixel |
| `binf_offset_abs_mean` | `use_per_frame_binf` enabled | Mean absolute per-frame B_inf offset |
| `binf_offset_std` | `use_per_frame_binf` enabled | Std dev of per-frame B_inf offsets |
| `exposure_abs_mean` | `use_per_frame_exposure` enabled | Mean absolute per-frame exposure correction |
| `exposure_std` | `use_per_frame_exposure` enabled | Std dev of per-frame exposure corrections |

### Derived metrics (not logged — computed in analysis/plotting)

These are computed from existing tags, not logged to TensorBoard directly.

| Derived | Formula | Meaning |
|---------|---------|---------|
| `clean_rg_ratio` | `clean_mean_r / clean_mean_g` | Color correction proxy. ~1.0 = balanced/restored; < 0.5 = underwater color cast persists |
| `beta_d_channel_order` | R > G > B check on `at_beta_r/g/b` | Whether attenuation follows physically expected ordering in water |
| `psnr_clean_gap` | `psnr - clean_psnr` (eval-only) | Medium decomposition activity signal. Larger gap = more medium contribution |

### Loss tags

Logged under `Train Loss Dict/` prefix.

| Tag | Loss | Drives |
|-----|------|--------|
| `main_loss` | L1/L2 reconstruction | Gaussians (primary), medium (weak indirect) |
| `gray_world` | Gray world color balance | β_D (in reverse_J mode) or Gaussians |
| `dcp` | Dark channel prior | β_B (backscatter upper bound, 1000:1 asymmetry) |
| `rgb_sat` | Saturation ceiling | Gaussians (prevents oversaturation) |
| `rgb_sv` | Contrast preservation | Attenuation (trivially satisfied near identity) |

Total loss is logged as `Train Loss`.
