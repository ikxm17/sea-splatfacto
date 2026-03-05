# SeaSplat Reference Notes

## Architecture Overview

SeaSplat extends 3D Gaussian Splatting with a physically-grounded underwater image formation model. A standard 3DGS produces a clean color render J and depth map Z. Two small learned networks (AttenuateNetV3, BackscatterNetV2) take the depth map as input and produce per-pixel attenuation and backscatter maps. The final underwater image is `I = J * A(Z) + B(Z)`. Training is two-phase: first train vanilla 3DGS with learned background, then jointly optimize 3DGS + medium parameters with additional physics-based losses.

## Image Formation Model (Eq 3)

```
I = J * exp(-beta_D * Z) + B_inf * (1 - exp(-beta_B * Z))
```

- `I`: captured underwater image
- `J`: clean scene color (3DGS render)
- `Z`: depth from camera
- `beta_D`: per-channel attenuation coefficient (learned, AttenuateNetV3)
- `beta_B`: per-channel backscatter coefficient (learned, BackscatterNetV2)
- `B_inf`: backscatter water color at infinity (learned parameter in BackscatterNetV2)
- `D = J * exp(-beta_D * Z)`: direct signal (attenuated true color)
- `B = B_inf * (1 - exp(-beta_B * Z))`: backscatter (veiling light)

Both beta_D and beta_B are implemented as (1,1,3) convolutions on depth, making them per-channel and depth-dependent.

## Code-to-Paper Mapping

| Paper | Concept | Reference Code |
|-------|---------|----------------|
| Eq 1 | Gaussian splatting color | (standard 3DGS, `gaussian_renderer/__init__.py:86-94`) |
| Eq 2 | Base reconstruction loss | `train.py:293` — `(1-lambda)*L1 + lambda*(1-SSIM)` |
| Eq 3 | Underwater image formation | `train.py:251-273` (training), `render_uw.py:32-45` (analytic version) |
| Eq 4 | Backscatter loss (DCP variant) | `deepseecolor/losses.py:146-161` — DarkChannelPriorLossV3 |
| Eq 5 | Gray world prior loss | `deepseecolor/losses.py:185-203` — GrayWorldPriorLoss |
| Eq 6 | Saturation loss | `deepseecolor/losses.py:218-232` — RgbSaturationLoss (T_sat=0.7) |
| Eq 7 | Depth-weighted reconstruction | `utils/loss_utils.py:17-20` — depth_weighted_l1_loss |
| Eq 8 | Depth smoothness | `deepseecolor/depth_losses.py:6-25` — SmoothDepthLoss |
| Eq 9 | Background opacity loss | `deepseecolor/losses.py:234-295` — AlphaBackgroundLoss |
| Eq 10 | Total loss | `train.py:293-331` (accumulated across all loss terms) |
| Sec IV.C | AttenuateNetV3 (beta_D) | `deepseecolor/models.py:203-241` — clamp(conv(depth, params), 0) then exp(-x) |
| Sec IV.C | BackscatterNetV2 (beta_B, B_inf) | `deepseecolor/models.py:39-91` — sigmoid(B_inf)*(1-exp(-clamp(conv(depth,params),0))) |

## Training Flow

### How the Medium is Learned (conceptual)

The core learning signal is **reconstruction loss on the predicted underwater image vs. the
actual captured underwater image**.  The training does NOT have access to clean (in-air)
ground truth — it only has the degraded underwater images I_gt.

Forward path each step:
1. 3DGS renders a "clean" image J and depth map Z.
2. Medium models take Z and produce attenuation A(Z) and backscatter B(Z).
3. Predicted underwater image: `I_pred = J * A(Z) + B(Z)`.
4. Reconstruction loss: `||I_pred - I_gt||` (L1 + SSIM).

Gradients flow back through BOTH the medium models (learning β_D, β_B, B∞) AND
the 3DGS (learning J, the true scene colors).  Without auxiliary losses this is
under-determined (e.g. A=1, B=0, J=I_gt is a trivial solution), so additional losses
are critical:

| Loss | Role in disentanglement |
|------|------------------------|
| DCP (Eq 4) | Prevents backscatter from being too small — pushes `I_gt - B` to look haze-free |
| Gray world (Eq 5) | Prevents J from keeping the underwater blue/green cast |
| Saturation (Eq 6) | Prevents J from overcompensating (clipping bright) |
| Depth-weighted recon (Eq 7) | Upweights distant pixels where medium effects are largest |
| Depth smooth (Eq 8) | Regularizes depth → cleaner medium maps |
| Background opacity (Eq 9) | Prevents floater Gaussians in the water column |

### Phase 1: Vanilla 3DGS (iter 0 → `seathru_from_iter`)
- Standard 3DGS training with L1 + SSIM reconstruction loss on `image` vs `gt`
  (where `image = render + sigmoid(learned_bg) * (1 - alpha)`)
- Learned background: sigmoid(bg_params) * (1 - alpha), initialized with blue-ish tint
- Depth smoothness loss: edge-aware total variation on depth (Eq 8)
- Alpha background loss: push alpha low where render color ≈ bg color (Eq 9)
- Gray world loss on rendered image (from `gw_from_iter`, default 10k)
- DCP loss is computed but NOT added to loss (logged only, `train.py:404-407`)
- Standard densification/pruning

### Phase 2a: Medium Warm-up (1000 steps, medium-only)
- Triggered at `iteration > seathru_from_iter` (strict `>`, NOT `>=`; `train.py:246`)
- At `seathru_from_iter + 1`: GS params frozen except colors (`train.py:185-189`)
- B_inf initialized from learned_bg values (`train.py:208-212`)
- For 1000 steps: only bs_optimizer and at_optimizer are stepped
  (`train.py:446-447`), then `continue` skips GS optimizer, bg_optimizer,
  and densification entirely
- At `seathru_from_iter + 2`: GS params unfrozen (`train.py:190-192`),
  but still in burst so `continue` prevents GS optimizer step anyway
- The freeze/unfreeze at +1/+2 is effectively a no-op since the `continue`
  already prevents GS updates for the entire 1000-step burst

### Phase 2b: GS Color Correction (2000 steps, GS colors only)
- After medium warm-up completes: `adjust_gs_colors_for_cc = True`
- For 2000 steps: only `gaussians.optimizer.step()` is called (`train.py:461`),
  then `continue` skips bs/at optimizer steps AND bg_optimizer step
- Purpose: let GS colors adapt to work with the now-initialized medium models
- Note: learned_bg is NOT updated during this phase (the `continue` at
  `train.py:463` skips `bg_optimizer.step()` at line 553)

### Phase 3: Joint Optimization (ongoing)
- Full image formation: `underwater_image = clamp(J * A(Z) + B(Z), 0, 1)`
- Reconstruction loss now on `underwater_image` vs `gt` (not `image` vs `gt`)
- All optimizers stepped: GS, bs/at, bg
- Periodic bs/at updates: every `update_bs_at_interval` (100) iters, run
  `update_bs_at_count` (50) inner steps on bs/at only (same `continue` mechanism)
- Backscatter loss (Eq 4): applied to `gt - backscatter` (direct image estimate from gt)
- Attenuation loss (DSC): optional, on `(gt - backscatter) / attenuation`
- Densification resumes (with optional `scale_grad_threshold` multiplier after unfreeze)
- All regularization losses continue

### Selective Optimization Mechanism (reference code)

The reference code uses a `continue` statement (`train.py:453,463`) to skip the rest
of the training loop (optimizer steps, densification, logging) during medium-only and
color-correction phases.  This is a simple but effective way to implement alternating
optimization in a single training loop.

What `continue` skips during medium burst (`train.py:453`):
- `gaussians.optimizer.step()` (line 551)
- `bg_optimizer.step()` (line 553)
- Densification (lines 516-546)
- Checkpoint saving

What `continue` skips during color correction (`train.py:463`):
- `bs_optimizer.step()` / `at_optimizer.step()` (never reached)
- `bg_optimizer.step()` (line 553)
- Densification (lines 516-546)

In the nerfstudio port, we cannot use `continue` because the training loop is in
the framework.  Instead, we null out `.grad` in `step_post_backward()` — Adam skips
any parameter whose `.grad` is `None`.  This is functionally equivalent.

## Loss Inventory

| Loss | Config Flag | Lambda | Description |
|------|------------|--------|-------------|
| L1 + SSIM | always on | `lambda_dssim=0.2` | Standard reconstruction (Eq 2) |
| Depth-weighted L1 | `add_recon_depth_l1=True` | `dwr_lambda=1.0` | Recon weighted by depth (Eq 7) |
| Depth smooth | `use_depth_smooth_loss=True` | `depth_smooth_lambda=2.0` | Edge-aware depth TV (Eq 8) |
| Gray world | `use_gw_loss=True` | `gw_loss_lambda=0.1` | Push channel means to 0.5 (Eq 5) |
| RGB saturation | `use_rgb_sat_loss=True` | `sat_loss_lambda=2.0` | Penalize J > 0.7 (Eq 6) |
| Backscatter/DCP | `use_dcp_loss=True` | `dcp_loss_lambda=1.0` | DCP variant on direct image (Eq 4) |
| Background alpha | `learn_background=True` | `bg_lambda=0.01` | Push alpha low near bg color (Eq 9) |
| Opacity prior | `use_opacity_prior=False` | `opacity_prior_lambda=0.0001` | Mixture of Laplacians on opacity |
| DSC attenuation | `use_dsc_at_loss=False` | `dsc_at_lambda=1.0` | DeepSeeColor attenuation loss |
| B_inf loss | `use_binf_loss=False` | `binf_loss_lambda=1.0` | Push B_inf toward atmospheric light |
| RGB spatial var | `use_rgb_sv_loss=False` | — | Match spatial variation of J to direct |
| Alpha smooth | `use_alpha_smooth_loss=False` | `alpha_smooth_lambda=1.0` | TV on alpha mask |
| Depth L1 | `use_depth_l1_loss=False` | 0.1 | L1 vs pseudo-GT depth (needs GT) |

## Key Config Defaults

From `arguments/__init__.py`:

```
seathru_from_iter = 9_000_000   # effectively disabled; must set explicitly (e.g. 10000)
sh_degree = 0                   # zero-order SH only (no view-dependent color)
use_at_v3 = True                # simplest attenuation model
bs_scale = 5.0, at_scale = 5.0  # conv param range
learn_background = True
bg_from_bs = True               # after seathru starts, drop learned bg, rely on backscatter
filter_depth = True             # depth = depth/alpha, nan->max, then normalize
normalize_depth = 1.0
norm_depth_max = True           # min-max normalize depth to [0,1]
update_bs_at_interval = 100     # every 100 GS iters, do a bs/at update burst
update_bs_at_count = 50         # 50 inner steps per burst
gw_from_iter = 10_000           # gray world loss kicks in at 10k
```
