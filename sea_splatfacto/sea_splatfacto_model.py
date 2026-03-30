"""
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""

import math
import torch

from torch.nn import Parameter
from gsplat.strategy import DefaultStrategy, MCMCStrategy

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

from pytorch_msssim import SSIM
from pytorch_msssim.ssim import _fspecial_gauss_1d, gaussian_filter

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

from nerfstudio.models.splatfacto import (
    SplatfactoModel,
    SplatfactoModelConfig,
)  # for subclassing Splatfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.model_components.lib_bilagrid import (
    BilateralGrid,
    color_correct,
    slice,
    total_variation_loss,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.math import k_nearest_sklearn, random_quat_tensor
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases

from sea_splatfacto.deepseecolor.models import BackscatterNetV2, AttenuateNetV3, AttenuateNetV4, ColorMLP
from sea_splatfacto.deepseecolor.losses import (
    AttenuateLoss,
    DarkChannelPriorLossV3,
    GrayWorldPriorLoss,
    SmoothDepthLoss,
    RGBSpatialVariationLoss,
    RGBSaturationLoss,
    AlphaBackgroundLoss,
    mixture_of_laplacians_loss,
)

from sea_splatfacto.utils.general_utils import inverse_sigmoid
from sea_splatfacto.utils.loss_utils import (
    depth_weighted_l1_loss,
    depth_weighted_l2_loss,
)


def _compute_ssim_components(
    X: torch.Tensor,
    Y: torch.Tensor,
    data_range: float = 1.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
) -> Tuple[float, float, float, float]:
    """Compute SSIM with full 3-way decomposition (Wang et al. 2004).

    Returns (ssim, luminance, contrast, structure) as scalar floats.
    Uses the same Gaussian window as pytorch_msssim for numerical consistency.
    """
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    C3 = C2 / 2

    C = X.shape[1]  # channels
    win = _fspecial_gauss_1d(win_size, win_sigma).repeat([C, 1, 1, 1]).to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(X * X, win) - mu1_sq
    sigma2_sq = gaussian_filter(Y * Y, win) - mu2_sq
    sigma12 = gaussian_filter(X * Y, win) - mu1_mu2

    sigma1 = torch.sqrt(sigma1_sq.clamp(min=0))
    sigma2 = torch.sqrt(sigma2_sq.clamp(min=0))

    # 3-way decomposition
    l_map = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    c_map = (2 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)
    s_map = (sigma12 + C3) / (sigma1 * sigma2 + C3)

    # Average across spatial dims, then across channels
    l_val = float(torch.flatten(l_map, 2).mean(-1).mean().item())
    c_val = float(torch.flatten(c_map, 2).mean(-1).mean().item())
    s_val = float(torch.flatten(s_map, 2).mean(-1).mean().item())

    ssim_map = l_map * c_map * s_map
    ssim_val = float(torch.flatten(ssim_map, 2).mean(-1).mean().item())

    return ssim_val, l_val, c_val, s_val


class MarineSnowCNN(torch.nn.Module):
    """Lightweight CNN that predicts marine snow from the clean rendered image.

    Conditions on I_rendered (clean scene, no medium model) so that the
    network cannot absorb medium model errors — addressing B0's core failure.
    The small receptive field constrains output to local patterns (particle-
    sized blobs), preventing scene-level corrections.
    """

    def __init__(self, channels: int = 32, depth: int = 4, kernel_size: int = 9):
        super().__init__()
        layers: list = []
        in_ch = 3
        for _ in range(depth - 1):
            layers.append(torch.nn.Conv2d(in_ch, channels, kernel_size, padding=kernel_size // 2))
            layers.append(torch.nn.ReLU(inplace=True))
            in_ch = channels
        layers.append(torch.nn.Conv2d(channels, 3, kernel_size, padding=kernel_size // 2))
        self.net = torch.nn.Sequential(*layers)

        # Near-zero init: softplus(-5) ≈ 0.007, so output is ~0 at training start
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.constant_(self.net[-1].bias, -5.0)

    def forward(self, rendered_image: torch.Tensor) -> torch.Tensor:
        """Predict non-negative marine snow map from the clean rendered image.

        Args:
            rendered_image: (1, 3, H, W) clean scene render (before medium model)
        Returns:
            snow_map: (1, 3, H, W) non-negative additive snow prediction
        """
        return torch.nn.functional.softplus(self.net(rendered_image))


@dataclass
class SeaSplatfactoModelConfig(SplatfactoModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: SeaSplatfactoModel)

    # Override Splatfacto defaults
    output_depth_during_training: bool = (
        True  # rendered depth is needed at every step for the medium models
    )
    background_color: Literal["black", "white", "random", "learned"] = (
        "black"  # background compositing is handled ourselves via learned_background, so the base renderer shoudl composite against black (i.e. contribute nothing)
    )
    sh_degree: int = 0
    """SeaSplat uses SH degree 0 (no view-dependent color)."""
    color_activation: Literal["sigmoid", "linear"] = "linear"
    """Use linear color representation to match original 3DGS. Sigmoid causes gradient vanishing
    for underwater scenes where red channel saturates near zero, preventing medium decomposition."""
    densify_grad_thresh: float = 0.0002
    """SeaSplat uses the original 3DGS densification gradient threshold."""
    use_absgrad: bool = False
    """SeaSplat uses original 3DGS-style gradients (not absgrad)."""
    cull_alpha_thresh: float = 0.005
    """Lower cull threshold; splatfacto's 0.1 is too aggressive for underwater scenes."""
    num_downscales: int = 0
    """No resolution downscaling; train at full resolution from step 1."""

    # Gaussian Splatting behavior
    do_isotropic: bool = False
    """Force isotropic (uniform-scale) Gaussians."""
    freeze_gs_from_iter: int = 9_000_000
    """Freeze all GS parameters (except colors) from this iteration."""
    unfreeze_gs_from_iter: int = 9_000_000
    """Unfreeze GS parameters from this iteration."""
    shuffle: bool = True
    """Shuffle camera order each epoch."""
    use_depth_weighted_l1: bool = False
    """Use depth-weighted L1 for the main reconstruction loss."""
    use_depth_weighted_l2: bool = False
    """Use depth-weighted L2 for the main reconstruction loss."""

    # Learned background
    learn_background: bool = True
    """Learn a background color composited via alpha: image = render + sigmoid(background) * (1 - alpha)."""
    bg_init_r: float = 0.05
    """Initial red channel value for learned background (before inverse-sigmoid)."""
    bg_init_g: float = 0.25
    """Initial green channel value for learned background."""
    bg_init_b: float = 0.80
    """Initial blue channel value for learned background."""
    bg_lambda: float = 0.01
    """Weight for the alpha-background loss."""
    add_bg_binf: bool = False
    """When using alpha_binf_* variants, also keep the base learned_background loss term (additive rather than replacement)"""
    use_lab: bool = False
    """Use L*a*b color difference in AlphaBackgroundLoss instead of RGB."""
    bg_lr: float = 1e-2
    """Learning rate for the learned background parameter."""
    bg_from_backscatter: bool = True
    """Once SeaThru activates, stop using learned_bg and let backscatter handle the water color.  B_inf is initialized from learned_bg at that point."""
    alpha_bg_opacities: bool = False
    """Compute alpha-background loss on per-Gaussian opacities (using SH to RGB colors) in addition to the rendered image."""
    alpha_binf_uw: bool = True
    """Compute alpha-background loss using B_inf against the underwater image (instead of learned_background) once SeaThru is active."""
    alpha_binf_render: bool = False
    """Compute alpha-background loss using B_inf against the underwater image once SeaThru is active."""
    alpha_bg_uw: bool = False
    """Compute alpha-background loss using B_inf against the underwater image (duplicate variant of alpha_binf_uw)."""
    turn_off_bg_loss: bool = False
    """Completely disable the alpha-background loss after SeaThru activates (when bg_from backscatter is True)."""

    # Depth processing
    use_gt_depth: bool = False
    """Swap out rendered depth with pseudo ground-truth depth maps."""
    use_depth_l1_loss: bool = False
    """Use L1 loss between rendered depth and GT depth."""
    depth_l1_lambda: float = 0.1
    """Weight for the depth L1 loss (rendered vs GT depth)."""
    use_depth_smooth_loss: bool = True
    """Edge-aware depth smootheness loss weighted by RGB gradients."""
    depth_smooth_lambda: float = 2.0
    """Weight for depth smoothness loss."""
    filter_depth: bool = True
    """Divide rendered depth by alpha and clean up NaN/Inf values"""
    normalize_depth: float = 1.0
    """Divide depth by this constant before further normalization."""
    norm_depth_max: bool = True
    """Min-max normalize depth to [0, 1]."""
    depth_alpha_threshold: float = 0.5
    """Alpha threshold used when masking depth for visualization / evaluation."""

    # Alpha smoothness loss
    use_alpha_smooth_loss: bool = False
    """Apply the same edge-aware smoothness loss to the alpha/accumulation map."""
    alpha_smooth_lambda: float = 1.0
    """Weight for the alpha smoothness loss."""

    # Opacity prior loss
    use_opacity_prior: bool = False
    """Mixture-of-Laplacians prior pushing opacities toward 0 or 1."""
    opacity_prior_lambda: float = 0.0001
    """Weight for the opacity prior loss."""

    # Mean opacity regularizer (idea 006)
    opacity_reg_lambda: float = 0.0
    """[idea-006] Penalizes mean opacity across all Gaussians, pushing unnecessary
    Gaussians toward zero. Useful for reducing floaters in narrow-view datasets.
    Set > 0 to enable (typical range 1e-3 to 1e-2). 0.0 = disabled."""
    opacity_reg_from_iter: int = 15000
    """[idea-006] First step to enable opacity regularization. Must be after
    densification completes to avoid destroying the Gaussian population during
    Phase 1. Default matches seathru_from_iter / stop_split_at."""

    # Depth-weighted reconstruction loss
    add_recon_depth_l1: bool = True
    """Add a depth-weighted L1 reconstruction loss."""
    dwr_lambda: float = 1.0
    """Weight for the depth-weighted reconstruction loss."""

    # Dark channel prior loss
    use_dcp_loss: bool = True
    """Dark channel prior loss — encourages haze-free direct signal."""
    dcp_loss_lambda: float = 1.0
    """Weight for DCP loss."""
    dcp_cost_ratio: float = 1000.0
    """Ratio of negative-to-positive penalty in DarkChannelPriorLossV3."""
    dcp_smooth_l1_beta: float = 0.2
    """Beta (transition point) for the SmoothL1Loss in DarkChannelPriorLossV3."""

    # RGB saturation loss
    use_rgb_sat_loss: bool = True
    """Penalize rendered pixel values outside [0, saturation_val]."""
    sat_loss_lambda: float = 2.0
    """Weight for the RGB saturation loss."""
    saturation_threshold: float = 0.7
    """Saturation value for RGBSaturationLoss — pixels beyond this are penalized."""

    # Gray world prior loss
    use_gw_loss: bool = True
    """Push mean channel intensities toward 0.5 (gray world assumption)."""
    gw_loss_lambda: float = 0.1
    """Weight for the gray world loss."""
    gw_reverse_J: bool = False
    """Apply gray world loss on J = direct / attenuation (the fully restored image) instead of the rendered image."""
    use_render_for_gw: bool = False
    """Apply gray world loss on the raw rendered image (before learned background compositing)."""
    gw_detach_alpha_bg: bool = False
    """Detach the learned_bg when compositing for the gray world loss input, so gradients only flow through the Gaussians"""
    gw_from_iter: int = 10_000
    """Only activate gray world loss after this iteration."""
    gw_filter_by_alpha: float = 0.0
    """If > 0, only include pixel with alpha above this threshold in the gray world loss computation."""

    # RGB spatial variation loss
    use_rgb_sv_loss: bool = False
    """Ensure the restored image has similar spatial variation to the input (from DeepSeeColor)."""
    rgb_sv_lambda: float = 0.01
    """Weight for the RGB spatial variation loss."""

    # B_inf loss
    use_binf_loss: bool = False
    """Push B_inf toward the estimated atmospheric light from dark channel prior estimation."""
    binf_loss_lambda: float = 1.0
    """Weight for the B_inf loss."""

    # DeepSeeColor attenuation loss
    use_dsc_attenuation_loss: bool = False
    """Regularize the resoted image J = (GT - backscatter) / attenuation to have reasonable intensity and spatial statistics."""
    dsc_attenuation_lambda: float = 1.0
    """Weight for the DSC attenuation loss."""

    # SeaThru medium models
    do_seathru: bool = True
    """Master toggle for the SeaThru underwater medium modelling. When False, the model behaves as standard Splatfacto with learned background and regularization losses only."""
    seathru_from_iter: int = 10_000
    """Iteration at which to activate the SeaThru medium models.  Set to a value larger than max_num_iterations to effectively disable."""
    backscatter_attenuation_lr: float = 1e-2
    """Learning rate for both the backscatter and attenuation models."""
    backscatter_scale: float = 5.0
    """Scale factor for BackscatterNetV2 depth-dependent coefficients."""
    attenuation_scale: float = 5.0
    """Scale factor for AttenuateNetV3 depth-dependent coefficients."""
    backscatter_do_sigmoid: bool = False
    """Use sigmoid (instead of clamp) on backscatter conv parameters."""
    attenuation_do_sigmoid: bool = False
    """Use sigmoid (instead of clamp) on attenuation conv parameters."""
    backscatter_use_residual: bool = False
    """Include the residual J_prime * exp(-β_d * z) term in the backscatter
    model (equation 10 from SeaThru)."""
    use_attenuation_v2: bool = False
    """Use AttenuateNetV2 (drops some terms) instead of the default."""
    use_attenuation_v3: bool = True
    """Use AttenuateNetV3 (simplest) — the default attenuation model."""
    disable_attenuation: bool = False
    """Simplified model that only accounts for backscatter (no attenuation)."""
    use_depth_dependent_beta_d: bool = False
    """[idea-012] Use AttenuateNetV4 with depth-varying attenuation rate
    (double exponential: beta_d(z) = w*exp(-v*z) + y*exp(-x*z), 12 params).
    Structurally prevents spatially uniform beta_D that defeated ideas 010/011."""
    medium_update_interval: int = 100
    """Interleaved medium step frequency: every this many iterations during
    Phase 3 joint training, one step updates medium models while GS is
    frozen.  Reference: train.py:434 — medium optimizers step once every
    update_bs_at_interval iterations, not in consecutive bursts."""
    medium_warmup_steps: int = 1000
    """Number of consecutive medium-only steps in the warm-up burst (Phase 1)."""
    cc_phase_steps: int = 2000
    """Number of GS color-correction steps in Phase 2 (excludes interleaved medium steps)."""
    scale_grad_threshold: float = 1.0
    """Multiplier on the densification gradient threshold after GS parameters are unfrozen (post-SeaThru activation)."""
    do_z_score: bool = False
    """Z-score filter the direct signal to ±3 standard deviations (clamps extreme values)."""

    # Model-dev idea 008: Early medium conditioning
    use_early_medium: bool = False
    """[idea-008] Put the medium model in the rendering path during Phase 1
    with FROZEN parameters. Forces Gaussians to learn colors that, when
    transformed by the medium, reproduce the underwater GT image. Prevents
    Gaussian entrenchment (memorizing underwater colors directly)."""
    early_medium_warmup_steps: int = 200
    """[idea-008] Shortened Phase 2a warm-up when early medium is active.
    The medium doesn't need to 'catch up' since it was present from step 0."""

    # Model-dev idea 009: Dataset-informed medium initialization
    beta_d_init_r: float = 1.1
    """[idea-009] Attenuation β_D red channel initialization. Higher = more red
    absorption. Default 1.1 (reference init). Set from dataset color statistics."""
    beta_d_init_g: float = 0.95
    """[idea-009] Attenuation β_D green channel initialization."""
    beta_d_init_b: float = 0.95
    """[idea-009] Attenuation β_D blue channel initialization."""
    beta_b_init_r: float = -1.0
    """[idea-009] Backscatter β_B red channel initialization. -1 = random (default)."""
    beta_b_init_g: float = -1.0
    """[idea-009] Backscatter β_B green channel initialization. -1 = random."""
    beta_b_init_b: float = -1.0
    """[idea-009] Backscatter β_B blue channel initialization. -1 = random."""

    # Model-dev idea 011: Clean render output constraints (from DeepSeeColor analysis)
    use_amplification_clamp: bool = False
    """[idea-011-A] Hard-clamp exp(β_D·z) to a maximum, limiting how different
    clean_rgb can be from the direct signal. From DeepSeeColor's proven design."""
    max_amplification: float = 3.0
    """[idea-011-A] Maximum amplification factor 1/T(z). 3.0 means clean can be
    at most 3× brighter than direct. DeepSeeColor uses 3.0."""
    use_clean_saturation_loss: bool = False
    """[idea-011-B] Penalize clean_rgb values outside [0, 1]. Prevents blown-out
    clean renders from Gaussian color compensation."""
    clean_sat_lambda: float = 1.0
    """[idea-011-B] Weight for clean render saturation loss."""
    use_variance_preservation: bool = False
    """[idea-011-C] Penalize mismatch between clean and medium render spatial
    variance. Detects Gaussian co-adaptation via spatial statistics divergence."""
    var_preservation_lambda: float = 1.0
    """[idea-011-C] Weight for variance preservation loss."""
    use_beta_d_ordering: bool = False
    """[idea-011-D] Enforce physical channel ordering β_D_R > β_D_G > β_D_B
    (red attenuates fastest underwater)."""
    beta_d_ordering_lambda: float = 0.1
    """[idea-011-D] Weight for channel ordering loss."""

    # Model-dev idea 012v01: Attenuation magnitude regularization
    # Prevents Mode 4 bypass where V4 learns near-identity attenuation at 50K
    use_attn_magnitude_loss: bool = False
    """[idea-012v01] Penalize the attenuation map for being too close to identity
    (all ones). Unlike beta_d_min (idea 010) which constrains parameter magnitude,
    this constrains the *functional* attenuation effect: mean(1 - T(z)) >= floor.
    This targets the integrated effect over actual scene depths, preventing
    sophisticated depth-correlated near-identity bypass."""
    attn_magnitude_lambda: float = 0.5
    """[idea-012v01] Weight for attenuation magnitude loss."""
    attn_magnitude_floor: float = 0.1
    """[idea-012v01] Minimum mean attenuation effect. 0.1 means: on average across
    the image, at least 10% of light should be attenuated. If mean(1-T(z)) < floor,
    the loss activates. Physical reference: saltpond depth range ~2-9m with moderate
    attenuation should produce 10-40% mean effect."""

    # Model-dev idea 013: GS color stop-gradient during medium steps
    use_gs_color_stop_gradient: bool = False
    """[idea-013] Detach clean_rgb during medium update steps so Gaussians cannot
    co-adapt while the medium is updating. Inspired by DeepSeeColor's .detach()
    between stages."""
    medium_window_size: int = 1
    """[idea-013] Number of consecutive medium steps before switching to GS steps.
    1 = current alternating behavior. Higher values give the medium time to converge
    before Gaussians react (DeepSeeColor uses 500)."""

    # Model-dev idea 015: Staged freeze training (DeepSeeColor-style)
    staged_medium_only_steps: int = 0
    """[idea-015] Extended medium-only training phase. When > 0, overrides the
    normal warmup duration (medium_warmup_steps / early_medium_warmup_steps) with
    this value. During this phase, ALL Gaussian gradients are nulled and
    densification is skipped — only medium models train. This forces medium
    activation because Gaussians cannot compete or bypass. Inspired by
    DeepSeeColor's staged training approach. Typical values: 3000-5000."""

    # Model-dev idea 010: β_D minimum regularization
    use_beta_d_min_reg: bool = False
    """[idea-010] Penalize β_D values below per-channel minimums using a smooth
    softplus penalty. Prevents the optimizer from collapsing attenuation to
    near-identity, which is the root bypass mechanism for decomposition."""
    beta_d_min_reg_lambda: float = 0.1
    """[idea-010] Weight for the β_D minimum regularization loss."""
    beta_d_min_r: float = 0.2
    """[idea-010] Minimum β_D for red channel. Red attenuates fastest underwater."""
    beta_d_min_g: float = 0.1
    """[idea-010] Minimum β_D for green channel."""
    beta_d_min_b: float = 0.05
    """[idea-010] Minimum β_D for blue channel. Blue attenuates least."""

    # Model-dev idea 002: Phase 3 medium LR decay
    use_medium_lr_decay: bool = False
    """[idea-002] Decay medium model LR during Phase 3 joint training to prevent
    medium parameter drift. When enabled, exponentially decays the backscatter and
    attenuation optimizer learning rates from their initial value to
    initial_lr * medium_lr_decay_factor over medium_lr_decay_steps steps,
    starting at Phase 3 onset."""
    medium_lr_decay_factor: float = 0.01
    """[idea-002] Final LR as a fraction of initial LR.
    E.g., 0.01 decays from 1e-2 to 1e-4."""
    medium_lr_decay_steps: int = 10000
    """[idea-002] Number of training steps over which to decay (from Phase 3 onset)."""

    # Model-dev idea 001: Phase 1→2 transition smoothing (fade-in blending)
    medium_fade_in_steps: int = 0
    """[idea-001] Number of steps to linearly blend from raw Gaussian RGB to medium
    output after SeaThru activation. 0 = disabled (abrupt switch, baseline behavior).
    When enabled, the rendered output gradually transitions:
      output = (1 - alpha) * raw_image + alpha * medium_image
    where alpha ramps from 0 to 1 over medium_fade_in_steps steps."""

    # Model-dev idea 005: Marine snow modelling
    use_marine_snow: bool = False
    """[idea-005] Enable marine snow model. The observation model becomes
    I_observed = I_underwater + S_k, where S_k captures view-inconsistent
    marine snow particles via sparsity and smoothness priors."""
    marine_snow_method: Literal["tensor", "cnn"] = "cnn"
    """[idea-005] Marine snow model architecture:
    'tensor' (B0, abandoned): unconstrained per-frame learnable tensor.
    'cnn' (B1): lightweight CNN conditioned on clean rendered image, with
    limited receptive field (~33px) preventing absorption of scene errors."""
    marine_snow_l1_lambda: float = 0.01
    """[idea-005] L1 sparsity prior weight. Pushes S_k toward zero (most
    pixels should have no snow). Higher values = sparser snow maps."""
    marine_snow_tv_lambda: float = 0.001
    """[idea-005] Total-variation smoothness prior weight. Encourages spatially
    smooth snow particles (Gaussian blobs, not pixel noise)."""
    marine_snow_from_iter: int = 15000
    """[idea-005] First step to enable marine snow learning. Must be after
    densification stabilizes so that Gaussians learn scene geometry first,
    not marine snow artifacts."""
    marine_snow_resolution: int = 64
    """[idea-005-B0] Spatial resolution of the per-frame snow map (tensor method only).
    The raw tensor is (N, 3, res, res) and bilinearly upsampled to image resolution."""
    marine_snow_cnn_channels: int = 32
    """[idea-005-B1] Hidden channels in the CNN. Controls model capacity."""
    marine_snow_cnn_depth: int = 4
    """[idea-005-B1] Number of convolutional layers. Receptive field = 1 + depth * (kernel_size - 1)."""
    marine_snow_cnn_kernel_size: int = 9
    """[idea-005-B1] Kernel size for each conv layer. With depth=4, kernel=9 gives RF=33px,
    appropriate for marine snow particle sizes (multi-pixel blobs, not scene-level patterns)."""

    # Model-dev idea 005-A1: Robust mask for marine snow
    use_robust_mask: bool = False
    """[idea-005-A1] RobustNeRF-style trimmed least-squares mask that excludes
    high-error pixels (likely marine snow or other transient occluders) from the
    reconstruction loss. The mask is recomputed each iteration based on per-pixel
    L1 errors, with a dynamic threshold interpolated from tracked loss statistics."""
    robust_mask_percentage: Tuple[float, float] = (0.0, 0.40)
    """[idea-005-A1] (min, max) fraction of pixels to mask out. The actual percentage
    is dynamically interpolated based on current loss relative to tracked min/max."""
    robust_mask_reset_interval: int = 6000
    """[idea-005-A1] Steps between resetting the max loss tracker. Allows the
    dynamic range to adapt as training progresses."""
    never_mask_upper: float = 0.0
    """[idea-005-A1] Fraction of image (top rows) to never mask. Set to 0.0 for
    underwater scenes (no sky). SplatFactoW uses 0.4 for outdoor scenes."""
    start_robust_mask_at: int = 6000
    """[idea-005-A1] First training step to enable robust masking. Must be after
    the medium model has had time to converge, so early-training residuals reflect
    genuine model error rather than initialization artifacts."""

    # Model-dev idea 003: GW loss annealing
    use_gw_anneal: bool = False
    """[idea-003] Anneal gw_loss_lambda from gw_anneal_start to gw_anneal_end over
    gw_anneal_steps, starting at gw_from_iter. Strong early GW establishes color
    correction; late relaxation prevents PSNR regression from GW-reconstruction
    conflict in Phase 3."""

    gw_anneal_start: float = 0.30
    """[idea-003] Initial GW weight at gw_from_iter (color-establishing phase)."""

    gw_anneal_end: float = 0.05
    """[idea-003] Final GW weight after annealing completes (fine-tuning phase)."""

    gw_anneal_steps: int = 15000
    """[idea-003] Number of steps over which to linearly anneal GW weight."""

    # Model-dev idea 007: Per-frame appearance correction
    use_per_frame_binf: bool = False
    """[idea-007-A] Per-frame B_inf offset: adds a learned [N_frames, 3] offset
    to the backscatter B_inf parameter in logit space (before sigmoid), allowing
    per-frame water color variation. 3 learnable parameters per training frame.
    At eval time, offset is zero (frame-independent baseline)."""

    use_per_frame_exposure: bool = False
    """[idea-007-B] Per-frame exposure/color scale: multiplicative correction
    applied to the combined medium model output. Parameterized in log-space
    so exp(0)=1 is the identity. 3 learnable parameters per training frame.
    At eval time, scale is 1.0 (no correction)."""

    per_frame_appearance_from_iter: int = 15000
    """[idea-007] Iteration at which per-frame appearance parameters begin
    learning (gradients are nulled before this). Should be well into Phase 3
    so the frame-independent medium model converges first."""

    # Model-dev idea 014: Color MLP bottleneck (SplatFacto-W inspired)
    use_color_mlp: bool = False
    """[idea-014] Replace direct Gaussian RGB with MLP-mediated colors.
    Per-Gaussian features are mapped through a shared MLP to produce
    pre-sigmoid RGB. The MLP has no depth input, structurally preventing
    Gaussians from encoding depth-dependent water effects — those must
    come from the medium model. Inspired by SplatFacto-W."""
    color_mlp_feature_dim: int = 16
    """[idea-014] Dimensionality of per-Gaussian appearance features."""
    color_mlp_hidden_dim: int = 64
    """[idea-014] Hidden layer width of the shared color MLP."""
    color_mlp_num_layers: int = 2
    """[idea-014] Number of hidden layers in the shared color MLP."""


class SeaSplatfactoModel(SplatfactoModel):
    """_summary_

    Args:
        SplatfactoModel (_type_): _description_
    """

    config: SeaSplatfactoModelConfig

    @property
    def features_dc(self):
        """Override base property to route through color MLP when enabled.

        When MLP is active, gauss_params["features_dc"] holds learned feature vectors
        (N x feature_dim) instead of raw RGB. The MLP maps these to RGB (N x 3).
        """
        if self.config.use_color_mlp and self.color_mlp_module is not None:
            return self.color_mlp_module(self.gauss_params["features_dc"])
        return self.gauss_params["features_dc"]

    def populate_modules(self):
        super().populate_modules()

        # [idea-014] Color MLP bottleneck
        # Replace per-Gaussian RGB with learned features routed through a shared MLP.
        # The MLP has no depth input, so Gaussians cannot encode depth-dependent color,
        # forcing the medium model to handle water effects.
        # IMPORTANT: We replace gauss_params["features_dc"] in-place (not add a new key)
        # because gsplat densification requires a 1:1 mapping between gauss_params keys
        # and optimizer entries.
        self.color_mlp_module: Optional[ColorMLP] = None
        if self.config.use_color_mlp:
            num_points = self.gauss_params["means"].shape[0]
            self.gauss_params["features_dc"] = torch.nn.Parameter(
                torch.zeros(
                    num_points,
                    self.config.color_mlp_feature_dim,
                    device=self.gauss_params["means"].device,
                )
            )
            self.color_mlp_module = ColorMLP(
                feature_dim=self.config.color_mlp_feature_dim,
                hidden_dim=self.config.color_mlp_hidden_dim,
                num_layers=self.config.color_mlp_num_layers,
            )

        self.backscatter_model: Optional[BackscatterNetV2] = None
        self.attenuation_model: Optional[nn.Module] = None

        if self.config.do_seathru:
            # [idea-009] Dataset-informed initialization for medium models
            beta_d_init = [
                self.config.beta_d_init_r,
                self.config.beta_d_init_g,
                self.config.beta_d_init_b,
            ]
            # Use explicit init if any value differs from reference defaults
            use_beta_d_init = beta_d_init != [1.1, 0.95, 0.95]

            beta_b_vals = [
                self.config.beta_b_init_r,
                self.config.beta_b_init_g,
                self.config.beta_b_init_b,
            ]
            # -1 sentinel means "use default behavior" (random init)
            beta_b_init = beta_b_vals if all(v >= 0 for v in beta_b_vals) else None

            # Backscatter and attenuation models
            self.backscatter_model = BackscatterNetV2(
                use_residual=self.config.backscatter_use_residual,
                scale=self.config.backscatter_scale,
                do_sigmoid=self.config.backscatter_do_sigmoid,
                beta_b_init=beta_b_init,
            )
            max_amp = self.config.max_amplification if self.config.use_amplification_clamp else None
            if self.config.use_depth_dependent_beta_d:
                # [idea-012] Depth-dependent beta_D (double exponential, 12 params)
                self.attenuation_model = AttenuateNetV4(
                    scale=self.config.attenuation_scale,
                    do_sigmoid=self.config.attenuation_do_sigmoid,
                    max_amplification=max_amp,
                )
            else:
                self.attenuation_model = AttenuateNetV3(
                    scale=self.config.attenuation_scale,
                    do_sigmoid=self.config.attenuation_do_sigmoid,
                    init_vals=not self.config.attenuation_do_sigmoid and not use_beta_d_init,
                    beta_d_init=beta_d_init if use_beta_d_init else None,
                    max_amplification=max_amp,
                )

            # [idea-009] Log medium initialization
            from nerfstudio.utils.rich_utils import CONSOLE as _C
            if use_beta_d_init:
                _C.log(
                    f"[INFO] [idea-009] β_D initialized from config: {beta_d_init}"
                )
            if beta_b_init is not None:
                _C.log(
                    f"[INFO] [idea-009] β_B initialized from config: {beta_b_init}"
                )

            # [idea-008] Initialize B_inf from bg_init values (not random) when
            # early medium is enabled. The frozen medium needs physically
            # plausible parameters — random B_inf injects garbage during Phase 1.
            if self.config.use_early_medium and self.config.learn_background:
                bg_init_logit = torch.log(
                    torch.tensor([
                        self.config.bg_init_r,
                        self.config.bg_init_g,
                        self.config.bg_init_b,
                    ]).clamp(1e-6, 1 - 1e-6)
                    / (1 - torch.tensor([
                        self.config.bg_init_r,
                        self.config.bg_init_g,
                        self.config.bg_init_b,
                    ]).clamp(1e-6, 1 - 1e-6))
                )
                with torch.no_grad():
                    self.backscatter_model.B_inf.data.copy_(
                        bg_init_logit.reshape(3, 1, 1)
                    )
                _C.log(
                    f"[INFO] [idea-008] B_inf initialized from bg_init = "
                    f"[{self.config.bg_init_r}, {self.config.bg_init_g}, {self.config.bg_init_b}] "
                    f"(logit: {bg_init_logit.tolist()})"
                )

        # Learn background
        if self.config.learn_background:
            bg_init = torch.tensor([
                self.config.bg_init_r,
                self.config.bg_init_g,
                self.config.bg_init_b,
            ])
            self.learned_bg = torch.nn.Parameter(inverse_sigmoid(bg_init))
        else:
            self.register_buffer("learned_bg", torch.zeros(3))

        # Loss criteria
        self.depth_smooth_criterion = SmoothDepthLoss()
        self.gw_criterion = GrayWorldPriorLoss()
        self.rgb_sv_criterion = RGBSpatialVariationLoss()
        self.rgb_sat_criterion = RGBSaturationLoss(saturation_val=self.config.saturation_threshold)
        self.alpha_bg_criterion = AlphaBackgroundLoss(use_kornia=self.config.use_lab)
        self.dsc_attenuation_criterion = AttenuateLoss()
        self.dcp_criterion = DarkChannelPriorLossV3(
            cost_ratio=self.config.dcp_cost_ratio,
            beta=self.config.dcp_smooth_l1_beta,
        )

        # State variables for tracking
        self.seathru_active: bool = False
        self.medium_inited = False
        self.warmup_counter = 0
        self.medium_steps_total = 0
        self.done_binf_init_with_bg = False
        self.adjust_gs_colors_for_color_correction = False
        self.gs_color_correction_counter = 0
        self._in_medium_burst: bool = False
        self._gs_frozen: bool = False
        self._phase3_onset_step: int = -1
        self._phase2_onset_step: int = -1

        # Marine snow model (idea 005)
        if self.config.use_marine_snow:
            if self.config.marine_snow_method == "cnn":
                # B1: CNN conditioned on clean rendered image
                self.marine_snow_cnn = MarineSnowCNN(
                    channels=self.config.marine_snow_cnn_channels,
                    depth=self.config.marine_snow_cnn_depth,
                    kernel_size=self.config.marine_snow_cnn_kernel_size,
                )
                n_params = sum(p.numel() for p in self.marine_snow_cnn.parameters())
                rf = 1 + self.config.marine_snow_cnn_depth * (self.config.marine_snow_cnn_kernel_size - 1)
                CONSOLE.log(
                    f"[INFO] Marine snow CNN: {n_params:,} params, RF={rf}px, "
                    f"channels={self.config.marine_snow_cnn_channels}, "
                    f"depth={self.config.marine_snow_cnn_depth}, "
                    f"l1={self.config.marine_snow_l1_lambda}, "
                    f"from_iter={self.config.marine_snow_from_iter}"
                )
            else:
                # B0: Per-frame learnable tensor (abandoned, kept for reference)
                res = self.config.marine_snow_resolution
                self.marine_snow_raw = torch.nn.Parameter(
                    torch.full(
                        (self.num_train_data, 3, res, res),
                        -5.0,  # softplus(-5) ≈ 0.007, near-zero init
                    )
                )
                mem_mb = self.marine_snow_raw.numel() * 4 / 1e6
                CONSOLE.log(
                    f"[INFO] Marine snow tensor: ({self.num_train_data}, 3, {res}, {res}), "
                    f"{mem_mb:.1f} MB, l1={self.config.marine_snow_l1_lambda}, "
                    f"tv={self.config.marine_snow_tv_lambda}, from_iter={self.config.marine_snow_from_iter}"
                )

        # Per-frame appearance correction (idea 007)
        if self.config.use_per_frame_binf:
            self.per_frame_binf_offsets = torch.nn.Parameter(
                torch.zeros(self.num_train_data, 3)
            )
            CONSOLE.log(
                f"[INFO] Per-frame B_inf offsets: ({self.num_train_data}, 3), "
                f"from_iter={self.config.per_frame_appearance_from_iter}"
            )
        if self.config.use_per_frame_exposure:
            self.per_frame_exposure = torch.nn.Parameter(
                torch.zeros(self.num_train_data, 3)
            )
            CONSOLE.log(
                f"[INFO] Per-frame exposure scale: ({self.num_train_data}, 3), "
                f"from_iter={self.config.per_frame_appearance_from_iter}"
            )

        # Robust mask loss tracking (idea 005-A1)
        self._robust_loss_min: float = float("inf")
        self._robust_loss_max: float = float("-inf")

        # Gradient magnitude tracking (recorded before nulling in step_post_backward)
        self._last_grad_bs: float = 0.0
        self._last_grad_at: float = 0.0

        CONSOLE.log(
            f"[INFO] do_seathru: {self.config.do_seathru}, "
            f"seathru_from_iter: {self.config.seathru_from_iter}, "
            f"disable_attenuation: {self.config.disable_attenuation}"
        )
        if self.config.use_early_medium:
            CONSOLE.log(
                f"[INFO] [idea-008] Early medium conditioning ENABLED. "
                f"Medium in rendering path from step 0 (frozen). "
                f"Phase 2a warmup: {self.config.early_medium_warmup_steps} steps."
            )

    def step_post_backward(self, step: int) -> None:
        """After backward: null out gradients for groups that should NOT be
        updated this step, then conditionally run Splatfacto's densification.

        This is the SOLE place where selective optimization is enforced.
        We cannot toggle ``requires_grad`` because gsplat's
        ``DefaultStrategy.step_pre_backward()`` (called during the forward
        pass) needs all GS params to have ``requires_grad=True`` so it can
        call ``.retain_grad()``.  Instead we null out ``.grad`` after
        backward -- Adam skips any parameter whose ``.grad`` is ``None``.

        Phase 2 (CC) interleaves medium optimizer steps every
        ``medium_update_interval`` iterations to keep medium models warm,
        matching the reference behavior (train.py:434-463).

        Source: train.py lines 430-463 (alternating optimization),
                train.py lines 516-557 (densification -- skipped when frozen).
        """
        # Record gradient magnitudes BEFORE nulling — for TensorBoard diagnostics.
        # Must happen here because get_metrics_dict() runs after nulling.
        if self.seathru_active and self.backscatter_model is not None:
            self._last_grad_bs = sum(
                p.grad.abs().mean().item() for p in self.backscatter_model.parameters()
                if p.grad is not None
            )
        if self.seathru_active and self.attenuation_model is not None:
            self._last_grad_at = sum(
                p.grad.abs().mean().item() for p in self.attenuation_model.parameters()
                if p.grad is not None
            )

        # [idea-008] Early medium phase: medium is in the rendering path but
        # FROZEN. Null medium gradients so only Gaussians update. Gaussians
        # receive gradients through the frozen medium (differentiable transform).
        if self.config.use_early_medium and not self.seathru_active:
            if self.backscatter_model is not None:
                for p in self.backscatter_model.parameters():
                    p.grad = None
            if self.attenuation_model is not None:
                for p in self.attenuation_model.parameters():
                    p.grad = None
            # Continue to densification (Gaussians still update normally)
            super().step_post_backward(step)
            return

        if self._in_medium_burst:
            # Medium-only: null out GS and learned_bg gradients
            for param in self.gauss_params.values():
                param.grad = None
            if self.config.learn_background and isinstance(self.learned_bg, Parameter):
                self.learned_bg.grad = None
            # Skip densification during medium bursts
            return

        if self.adjust_gs_colors_for_color_correction:
            # CC phase: primarily GS color adjustment, but medium models get
            # an interleaved update every medium_update_interval steps to stay
            # warm (reference: train.py:434-463 — the `iteration %
            # update_bs_at_interval == 0` check runs during CC too).
            if self._is_medium_step(step):
                # Interleaved medium step during CC: null GS + bg grads
                for param in self.gauss_params.values():
                    param.grad = None
                if self.config.learn_background and isinstance(self.learned_bg, Parameter):
                    self.learned_bg.grad = None
                self.medium_steps_total += 1
            else:
                # Normal CC step: GS updates, medium + bg frozen
                if self.backscatter_model is not None:
                    for p in self.backscatter_model.parameters():
                        p.grad = None
                if self.attenuation_model is not None:
                    for p in self.attenuation_model.parameters():
                        p.grad = None
                if self.config.learn_background and isinstance(self.learned_bg, Parameter):
                    self.learned_bg.grad = None
            # Skip densification during color correction
            return

        if self._gs_frozen:
            # GS freeze: null out all GS grads except colors
            # Source: train.py lines 174-192 (config-driven + seathru activation)
            for name, param in self.gauss_params.items():
                if name not in ("features_dc", "features_rest"):
                    param.grad = None
            # Still skip densification when GS is frozen
            return

        # Phase 3 joint training: interleaved GS and medium steps.
        # Reference (train.py:434): medium optimizers step ONCE every
        # update_bs_at_interval iterations (with `continue` to skip GS).
        # On all other iterations, GS optimizer steps normally.
        # [idea-013] medium_window_size > 1 runs consecutive medium steps per cycle.
        if self.seathru_active and self.medium_inited:
            if self._is_medium_step(step):
                # Interleaved medium step: null GS + bg grads, medium updates
                for param in self.gauss_params.values():
                    param.grad = None
                if self.config.learn_background and isinstance(
                    self.learned_bg, Parameter
                ):
                    self.learned_bg.grad = None
                self.medium_steps_total += 1
                # Skip densification (reference: `continue` skips everything)
                return
            else:
                # Normal GS step: null medium grads
                if self.backscatter_model is not None:
                    for p in self.backscatter_model.parameters():
                        p.grad = None
                if self.attenuation_model is not None:
                    for p in self.attenuation_model.parameters():
                        p.grad = None

        # [idea-005] Null marine snow gradients before activation
        if self.config.use_marine_snow and step < self.config.marine_snow_from_iter:
            if hasattr(self, "marine_snow_cnn"):
                for p in self.marine_snow_cnn.parameters():
                    p.grad = None
            elif hasattr(self, "marine_snow_raw"):
                self.marine_snow_raw.grad = None

        # [idea-007] Null per-frame appearance gradients before activation
        if step < self.config.per_frame_appearance_from_iter:
            if self.config.use_per_frame_binf and hasattr(self, "per_frame_binf_offsets"):
                self.per_frame_binf_offsets.grad = None
            if self.config.use_per_frame_exposure and hasattr(self, "per_frame_exposure"):
                self.per_frame_exposure.grad = None

        # Normal operation -- run Splatfacto's strategy (densification/pruning)
        super().step_post_backward(step)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """_summary_

        Args:
            training_callback_attributes (TrainingCallbackAttributes): _description_

        Returns:
            List[TrainingCallback]: _description_
        """
        cbs = super().get_training_callbacks(training_callback_attributes)

        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self._seasplat_before_iteration,
            )
        )
        return cbs

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """_summary_

        Returns:
            Dict[str, List[Parameter]]: _description_
        """
        param_groups = super().get_param_groups()  # get base parameter groups

        # [idea-014] Color MLP parameters
        if self.config.use_color_mlp and self.color_mlp_module is not None:
            param_groups["color_mlp"] = list(self.color_mlp_module.parameters())

        if self.config.do_seathru and self.backscatter_model is not None:
            param_groups["backscatter_model"] = list(
                self.backscatter_model.parameters()
            )
        if self.config.do_seathru and self.attenuation_model is not None:
            param_groups["attenuation_model"] = list(
                self.attenuation_model.parameters()
            )
        if self.config.learn_background:
            param_groups["learned_background"] = [self.learned_bg]

        # Per-frame appearance parameters (idea 007)
        per_frame_params = []
        if self.config.use_per_frame_binf and hasattr(self, "per_frame_binf_offsets"):
            per_frame_params.append(self.per_frame_binf_offsets)
        if self.config.use_per_frame_exposure and hasattr(self, "per_frame_exposure"):
            per_frame_params.append(self.per_frame_exposure)
        if per_frame_params:
            param_groups["per_frame_appearance"] = per_frame_params

        return param_groups

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """_summary_

        Args:
            camera (Cameras): _description_

        Returns:
            Dict[str, Union[torch.Tensor, List]]: _description_
        """
        outputs = super().get_outputs(camera)

        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a Cameras instance")
            return {}

        # Get base outputs (includes rgb, depth, accumulation, background)
        rendered_image = outputs["rgb"]  # [H, W, 3]
        alpha = outputs["accumulation"]  # [H, W, 1]
        depth_raw = outputs["depth"]  # [H, W, 1] or None

        # Learned background compositing
        # [idea-008] Early medium: put medium in the rendering path during Phase 1
        # with frozen parameters. Gaussians learn through the medium transform.
        early_medium_phase = (
            self.config.use_early_medium
            and self.config.do_seathru
            and self.backscatter_model is not None
            and self.attenuation_model is not None
            and self.training
            and not self.seathru_active
        )
        seathru_forward = (
            self.config.do_seathru
            and self.backscatter_model is not None
            and self.attenuation_model is not None
            and (self.seathru_active or not self.training or early_medium_phase)
        )

        if self.config.learn_background:
            if self.config.bg_from_backscatter and seathru_forward and not early_medium_phase:
                clean_rgb = rendered_image  # after SeaThru activates, stop adding learned_bg, the backscatter model fills in the water color
            else:
                bg_color = torch.sigmoid(self.learned_bg)  # [3]
                bg_image = bg_color.reshape(1, 1, 3) * (1 - alpha)  # [H, W, 3]
                clean_rgb = rendered_image + bg_image
        else:
            clean_rgb = rendered_image

        outputs["image"] = clean_rgb
        outputs["clean_rgb"] = clean_rgb
        outputs["rendered_image"] = rendered_image

        # Clean render variant with detached background (no GW gradients into learned_bg)
        if self.config.learn_background and self.config.gw_detach_alpha_bg:
            bg_detached = torch.sigmoid(self.learned_bg.detach()).reshape(1, 1, 3) * (
                1 - alpha
            )
            outputs["clean_rgb_bg_detached"] = rendered_image + bg_detached

        # Depth processing
        if depth_raw is not None:
            depth_processed = self._process_depth(depth_raw, alpha)
        else:
            # fallback, create a dummy depth tensor
            H, W = rendered_image.shape[:2]
            depth_processed = torch.ones(H, W, 1, device=self.device)
        outputs["depth_processed"] = depth_processed

        # SeaThru forward
        if seathru_forward:
            # convert to BCHW for medium models
            clean_bchw = self._to_bchw(clean_rgb)  # [1, 3, H, W]
            depth_bchw = self._to_bchw(depth_processed)  # [1, 1, H, W]

            # [idea-013] Stop-gradient on Gaussian colors during medium steps
            # Detach clean_bchw so medium receives full gradients but they don't
            # flow back into Gaussian colors, preventing co-adaptation
            if (
                self.config.use_gs_color_stop_gradient
                and self.training
                and self.seathru_active
                and self.medium_inited
                and self._is_medium_step(self.step)
            ):
                clean_bchw = clean_bchw.detach()

            # attenuation
            if self.config.disable_attenuation:
                direct_bchw = clean_bchw
                attenuation_map_bchw = torch.ones_like(
                    clean_bchw
                )  # all 1s, no attenuation
                attenuation_map_detach_bchw = attenuation_map_bchw
            else:
                attenuation_map_bchw = self.attenuation_model(
                    depth_bchw
                )  # [1, 3, H, W]
                attenuation_map_detach_bchw = self.attenuation_model(
                    depth_bchw.detach()
                )
                direct_bchw = clean_bchw * attenuation_map_bchw

            # z-score filter
            if self.config.do_z_score:
                direct_mean = direct_bchw.mean(dim=[2, 3], keepdim=True)
                direct_std = direct_bchw.std(dim=[2, 3], keepdim=True).clamp(min=1e-8)
                direct_zscore = (direct_bchw - direct_mean) / direct_std
                direct_zscore_clamped = torch.clamp(direct_zscore, -3, 3)
                direct_bchw = torch.clamp(
                    (direct_zscore_clamped * direct_std)
                    + torch.maximum(
                        direct_mean, torch.tensor(1.0 / 255, device=self.device)
                    ),
                    0,
                    1,
                )

            # [idea-007-A] Per-frame B_inf offset for water color correction
            # Apply during both training AND eval when cam_idx is available.
            # Novel views (no cam_idx) fall back to identity (offset=0).
            binf_offset = None
            if self.config.use_per_frame_binf and "cam_idx" in camera.metadata:
                cam_idx = camera.metadata["cam_idx"]
                binf_offset = self.per_frame_binf_offsets[cam_idx].reshape(3, 1, 1)

            # backscatter
            backscatter_bchw = self.backscatter_model(
                depth_bchw, binf_offset=binf_offset
            )  # [1, 3, H, W]
            backscatter_detach_bchw = self.backscatter_model(
                depth_bchw.detach(), binf_offset=binf_offset
            )

            # combined medium image (scene through water)
            medium_bchw = torch.clamp(direct_bchw + backscatter_bchw, 0.0, 1.0)

            # [idea-007-B] Per-frame exposure/color correction
            # Apply during both training AND eval when cam_idx is available.
            # Novel views (no cam_idx) fall back to identity (scale=1).
            if self.config.use_per_frame_exposure and "cam_idx" in camera.metadata:
                cam_idx = camera.metadata["cam_idx"]
                exposure_scale = torch.exp(
                    self.per_frame_exposure[cam_idx]
                ).reshape(1, 3, 1, 1)
                medium_bchw = torch.clamp(medium_bchw * exposure_scale, 0.0, 1.0)

            # [idea-001] Fade-in blending: gradual transition from raw image to medium
            if (
                self.config.medium_fade_in_steps > 0
                and self.training
                and self._phase2_onset_step > 0
            ):
                fade_progress = min(
                    (self.step - self._phase2_onset_step)
                    / self.config.medium_fade_in_steps,
                    1.0,
                )
                if fade_progress < 1.0:
                    medium_bchw = (
                        (1.0 - fade_progress) * clean_bchw
                        + fade_progress * medium_bchw
                    )

            # Store in outputs (HWC format)
            outputs["medium_rgb"] = self._to_hwc(medium_bchw)
            outputs["direct"] = self._to_hwc(direct_bchw)
            outputs["backscatter"] = self._to_hwc(backscatter_bchw)
            outputs["attenuation_map"] = self._to_hwc(attenuation_map_bchw)
            outputs["backscatter_depth_detached"] = self._to_hwc(
                backscatter_detach_bchw
            )
            outputs["attenuation_depth_detached"] = self._to_hwc(
                attenuation_map_detach_bchw
            )
            # [idea-008] Flag so loss dict knows to skip medium-specific losses
            # during early medium phase (medium is frozen, can't respond to losses)
            if early_medium_phase:
                outputs["early_medium_phase"] = True

        # [idea-005] Marine snow: I_observed = I_underwater + S_k
        if self.config.use_marine_snow and self.training:
            H, W = rendered_image.shape[:2]
            if self.config.marine_snow_method == "cnn":
                # B1: CNN predicts snow from clean rendered image (no medium model info)
                clean_bchw = self._to_bchw(outputs["clean_rgb"])  # [1, 3, H, W]
                snow_bchw = self.marine_snow_cnn(clean_bchw)  # [1, 3, H, W]
            else:
                # B0: Per-frame tensor lookup
                cam_idx = camera.metadata["cam_idx"]
                snow_raw = self.marine_snow_raw[cam_idx].unsqueeze(0)  # [1, 3, res, res]
                snow_lowres = torch.nn.functional.softplus(snow_raw)
                snow_bchw = torch.nn.functional.interpolate(
                    snow_lowres, size=(H, W), mode="bilinear", align_corners=False
                )  # [1, 3, H, W]
            snow_hwc = self._to_hwc(snow_bchw)  # [H, W, 3]
            outputs["marine_snow"] = snow_hwc

        return outputs

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        # Use medium image for PSNR when seathru is active
        modified_outputs = dict(outputs)
        if "medium_rgb" in outputs:
            modified_outputs["rgb"] = outputs["medium_rgb"]
        else:
            modified_outputs["rgb"] = outputs["clean_rgb"]

        metrics_dict = super().get_metrics_dict(modified_outputs, batch)

        # Log medium model parameters
        # Source: train.py lines 636-668
        if self.config.learn_background:
            bg_rgb = torch.sigmoid(self.learned_bg)
            metrics_dict["bg_r"] = bg_rgb[0]
            metrics_dict["bg_g"] = bg_rgb[1]
            metrics_dict["bg_b"] = bg_rgb[2]

        if self.backscatter_model is not None and self.seathru_active:
            binf = torch.sigmoid(self.backscatter_model.B_inf.detach()).squeeze()
            metrics_dict["binf_r"] = binf[0]
            metrics_dict["binf_g"] = binf[1]
            metrics_dict["binf_b"] = binf[2]

            # Backscatter beta (conv params) — effective depth coefficients
            bs_beta = self.backscatter_model.backscatter_conv_params.detach().squeeze()
            metrics_dict["bs_beta_r"] = bs_beta[0]
            metrics_dict["bs_beta_g"] = bs_beta[1]
            metrics_dict["bs_beta_b"] = bs_beta[2]

        if self.attenuation_model is not None and self.seathru_active:
            if isinstance(self.attenuation_model, AttenuateNetV3):
                # V3: scalar beta_D per channel
                at_beta = self.attenuation_model.attenuation_conv_params.detach().squeeze()
                metrics_dict["at_beta_r"] = at_beta[0]
                metrics_dict["at_beta_g"] = at_beta[1]
                metrics_dict["at_beta_b"] = at_beta[2]
            elif isinstance(self.attenuation_model, AttenuateNetV4):
                # V4: log effective beta_d at reference depth z=1.0
                with torch.no_grad():
                    z_ref = torch.ones(1, 1, 1, 1, device=self.device)
                    t_ref = self.attenuation_model(z_ref).squeeze()  # [3]
                    beta_eff = -torch.log(t_ref.clamp(min=1e-8))  # effective β_D*z at z=1
                    metrics_dict["at_beta_eff_r"] = beta_eff[0]
                    metrics_dict["at_beta_eff_g"] = beta_eff[1]
                    metrics_dict["at_beta_eff_b"] = beta_eff[2]

        # Gradient diagnostics: use pre-recorded values from step_post_backward
        # (recorded BEFORE gradient nulling, so they reflect actual gradient flow)
        if self.seathru_active and self.training:
            metrics_dict["grad_backscatter"] = torch.tensor(self._last_grad_bs)
            metrics_dict["grad_attenuation"] = torch.tensor(self._last_grad_at)

        # Clean RGB channel means — detect entrenchment (low red = memorized underwater)
        clean_rgb = outputs.get("clean_rgb")
        if clean_rgb is not None:
            clean_means = clean_rgb.detach().mean(dim=(0, 1))  # [3]
            metrics_dict["clean_mean_r"] = clean_means[0]
            metrics_dict["clean_mean_g"] = clean_means[1]
            metrics_dict["clean_mean_b"] = clean_means[2]

        # [idea-003] Log effective GW weight when annealing
        if self.config.use_gw_anneal and self.step > self.config.gw_from_iter:
            anneal_progress = min(
                (self.step - self.config.gw_from_iter) / max(self.config.gw_anneal_steps, 1),
                1.0,
            )
            metrics_dict["gw_weight_eff"] = (
                self.config.gw_anneal_start
                + (self.config.gw_anneal_end - self.config.gw_anneal_start) * anneal_progress
            )

        # [idea-005-B0] Log snow magnitude
        if "marine_snow" in outputs:
            metrics_dict["snow_magnitude"] = outputs["marine_snow"].detach().mean()

        # [idea-007] Log per-frame appearance parameter statistics
        if self.config.use_per_frame_binf and hasattr(self, "per_frame_binf_offsets"):
            offsets = self.per_frame_binf_offsets.detach()
            metrics_dict["binf_offset_abs_mean"] = offsets.abs().mean()
            metrics_dict["binf_offset_std"] = offsets.std()
        if self.config.use_per_frame_exposure and hasattr(self, "per_frame_exposure"):
            exposures = self.per_frame_exposure.detach()
            metrics_dict["exposure_abs_mean"] = exposures.abs().mean()
            metrics_dict["exposure_std"] = exposures.std()

        # Decomposition activity metrics — continuous proxies for medium health
        # Log in all phases: zeros in Phase 1 (baseline), real values in Phase 2/3
        medium_rgb = outputs.get("medium_rgb")
        clean_rgb = outputs.get("clean_rgb")
        attenuation_map = outputs.get("attenuation_map")
        backscatter = outputs.get("backscatter")

        if self.seathru_active and medium_rgb is not None and clean_rgb is not None:
            medium_mean = medium_rgb.detach().abs().mean().clamp(min=1e-8)
            metrics_dict["medium_contribution"] = (
                (medium_rgb.detach() - clean_rgb.detach()).abs().mean() / medium_mean
            )
        else:
            metrics_dict["medium_contribution"] = torch.tensor(0.0)

        if self.seathru_active and attenuation_map is not None:
            metrics_dict["attenuation_magnitude"] = (
                (1.0 - attenuation_map.detach()).abs().mean()
            )
        else:
            metrics_dict["attenuation_magnitude"] = torch.tensor(0.0)

        if self.seathru_active and backscatter is not None:
            metrics_dict["backscatter_magnitude"] = backscatter.detach().abs().mean()
        else:
            metrics_dict["backscatter_magnitude"] = torch.tensor(0.0)

        return metrics_dict

    @torch.no_grad()
    def _compute_robust_mask(self, errors: torch.Tensor) -> torch.Tensor:
        """Compute a binary inlier mask using trimmed least-squares (RobustNeRF/SplatFactoW).

        Pixels with high per-pixel error (likely marine snow or other transient
        occluders) are masked out. The masking threshold is dynamically determined
        from a quantile of the error distribution, with the quantile percentage
        interpolated between a min/max range based on tracked loss statistics.

        Args:
            errors: Per-pixel absolute errors, shape [H, W, C].

        Returns:
            Binary mask [H, W, 1] where 1 = inlier (keep), 0 = outlier (mask).
        """
        H, W, C = errors.shape

        # Zero out the top portion of the image (never mask sky region).
        # For underwater scenes, never_mask_upper should be 0.0.
        if self.config.never_mask_upper > 0.0:
            errors = errors.clone()
            errors[: int(H * self.config.never_mask_upper), :, :] = 0.0

        # Track min/max loss for dynamic masking percentage
        mean_loss = errors.mean().item()
        if (
            mean_loss > self._robust_loss_max
            or self.step % self.config.robust_mask_reset_interval == 0
        ):
            self._robust_loss_max = mean_loss
        if mean_loss < self._robust_loss_min:
            self._robust_loss_min = mean_loss

        # Interpolate masking percentage from loss statistics
        loss_range = self._robust_loss_max - self._robust_loss_min + 1e-6
        pct_min, pct_max = self.config.robust_mask_percentage
        mask_percentage = (
            (mean_loss - self._robust_loss_min) / loss_range
        ) * (pct_max - pct_min) + pct_min

        # Per-pixel error (mean over channels)
        error_per_pixel = errors.mean(dim=-1, keepdim=True)  # [H, W, 1]

        # Inlier threshold: pixels below this quantile are kept
        inlier_threshold = torch.quantile(
            error_per_pixel.reshape(-1), 1.0 - mask_percentage
        )
        is_inlier = (error_per_pixel <= inlier_threshold).float()  # [H, W, 1]

        # Spatial smoothing: 5x5 box filter to ensure spatial coherence
        # A pixel with enough inlier neighbors is also kept
        f = 5
        window = torch.ones(1, 1, f, f, device=errors.device) / (f * f)
        is_inlier_bchw = is_inlier.permute(2, 0, 1).unsqueeze(0)  # [1, 1, H, W]
        has_inlier_neighbors = torch.nn.functional.conv2d(
            is_inlier_bchw, window, padding="same"
        )
        has_inlier_neighbors = (
            has_inlier_neighbors.squeeze(0).permute(1, 2, 0) > 0.4
        ).float()  # [H, W, 1]

        # Union: pixel is inlier if it passes EITHER criterion
        mask = ((is_inlier + has_inlier_neighbors) > 0.0).float()  # [H, W, 1]

        return mask

    def get_loss_dict(
        self, outputs, batch, metrics_dict=None
    ) -> Dict[str, torch.Tensor]:
        """_summary_

        Args:
            outputs (_type_): _description_
            batch (_type_): _description_
            metrics_dict (_type_, optional): _description_. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: _description_
        """
        seathru_forward = "medium_rgb" in outputs
        early_medium = outputs.get("early_medium_phase", False)

        # [idea-005-B0] When marine snow is active, the reconstruction target
        # becomes observed_rgb = medium_rgb + snow_map, so the per-frame snow
        # tensor absorbs marine snow particles instead of the Gaussians.
        snow_active = (
            "marine_snow" in outputs
            and self.config.use_marine_snow
            and self.step >= self.config.marine_snow_from_iter
        )

        modified_outputs = dict(outputs)
        if seathru_forward:
            pred_rgb = outputs["medium_rgb"]
        else:
            pred_rgb = outputs["clean_rgb"]

        if snow_active:
            observed_rgb = torch.clamp(pred_rgb + outputs["marine_snow"], 0.0, 1.0)
            modified_outputs["rgb"] = observed_rgb
            outputs["observed_rgb"] = observed_rgb
        else:
            modified_outputs["rgb"] = pred_rgb

        loss_dict = super().get_loss_dict(modified_outputs, batch, metrics_dict)

        # Get commonly used tensors
        gt_image = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )  # [H, W, 3]
        clean_rgb = outputs["clean_rgb"]  # clean render + learned_bg
        rendered_image = outputs["rendered_image"]  # raw render
        alpha = outputs["accumulation"]  # [H,W,1]
        depth = outputs["depth_processed"]  # [H,W,1]

        # tensors in BCHW for loss criteria that expect batch format
        gt_bchw = self._to_bchw(gt_image)
        clean_bchw = self._to_bchw(clean_rgb)
        render_bchw = self._to_bchw(rendered_image)
        alpha_bchw = self._to_bchw(alpha)
        depth_bchw = self._to_bchw(depth)

        # Final prediction for reconstruction losses: includes snow when active
        pred_rgb = observed_rgb if snow_active else pred_rgb

        step = self.step

        # Robust mask: recompute main_loss with outlier pixels masked out (idea 005-A1)
        if self.config.use_robust_mask and step >= self.config.start_robust_mask_at:
            per_pixel_errors = torch.abs(pred_rgb - gt_image)  # [H, W, 3]
            robust_mask = self._compute_robust_mask(per_pixel_errors)  # [H, W, 1]

            # Apply mask to both images (zeroing outlier pixels)
            gt_masked = gt_image * robust_mask
            pred_masked = pred_rgb * robust_mask

            # Recompute L1 and SSIM with masked images
            Ll1_masked = torch.abs(gt_masked - pred_masked).mean()
            simloss_masked = 1.0 - self.ssim(
                gt_masked.permute(2, 0, 1)[None, ...],
                pred_masked.permute(2, 0, 1)[None, ...],
            )
            loss_dict["main_loss"] = (
                (1 - self.config.ssim_lambda) * Ll1_masked
                + self.config.ssim_lambda * simloss_masked
            )
        else:
            robust_mask = None

        # Depth-weighted reconstruction L1
        if self.config.add_recon_depth_l1:
            if robust_mask is not None:
                depth_weighted_l1 = depth_weighted_l1_loss(
                    pred_rgb * robust_mask, gt_image * robust_mask, depth.detach()
                )
            else:
                depth_weighted_l1 = depth_weighted_l1_loss(
                    pred_rgb, gt_image, depth.detach()
                )
            loss_dict["recon_depth_l1"] = self.config.dwr_lambda * depth_weighted_l1

        # Opacity prior (mixture-of-laplacians)
        if self.config.use_opacity_prior:
            loss_dict["opacity_prior"] = (
                self.config.opacity_prior_lambda
                * mixture_of_laplacians_loss(torch.sigmoid(self.opacities))
            )

        # Mean opacity regularizer (idea 006) — gated to start after densification
        if self.config.opacity_reg_lambda > 0.0 and step >= self.config.opacity_reg_from_iter:
            loss_dict["opacity_reg"] = self.config.opacity_reg_lambda * torch.sigmoid(self.opacities).mean()

        # Per-frame marine snow regularization (idea 005-B0)
        if snow_active:
            snow_map = outputs["marine_snow"]  # [H, W, 3]
            snow_bchw = self._to_bchw(snow_map)  # [1, 3, H, W]
            # L1 sparsity: most pixels should have zero snow
            loss_dict["snow_l1"] = self.config.marine_snow_l1_lambda * snow_map.mean()
            # TV smoothness: snow particles are spatially smooth (Gaussian blobs)
            loss_dict["snow_tv"] = self.config.marine_snow_tv_lambda * total_variation_loss(snow_bchw)

        # Alpha-background loss
        if self.config.learn_background:
            alpha_chw = alpha.permute(2, 0, 1)
            alpha_bg_loss = self.alpha_bg_criterion(
                rendered_image.permute(2, 0, 1).detach(),
                torch.sigmoid(self.learned_bg.detach()),
                alpha_chw,
            )

            if seathru_forward and not early_medium:
                medium_chw = outputs["medium_rgb"].permute(2, 0, 1)
                b_inf_sigmoid = torch.sigmoid(self.backscatter_model.B_inf.detach())

                if (
                    self.config.alpha_binf_uw
                    or self.config.alpha_binf_render
                    or self.config.alpha_bg_opacities
                ):
                    if not self.config.add_bg_binf:
                        alpha_bg_loss = torch.tensor(0.0, device=self.device)
                    if self.config.alpha_binf_uw:
                        alpha_bg_loss = alpha_bg_loss + self.alpha_bg_criterion(
                            medium_chw.detach(),
                            b_inf_sigmoid.squeeze(),
                            alpha_chw,
                        )
                    if self.config.alpha_bg_uw:
                        alpha_bg_loss = alpha_bg_loss + self.alpha_bg_criterion(
                            medium_chw.detach(),
                            b_inf_sigmoid.squeeze(),
                            alpha_chw,
                        )
                    if self.config.alpha_binf_render:
                        alpha_bg_loss = alpha_bg_loss + self.alpha_bg_criterion(
                            rendered_image.permute(2, 0, 1).detach(),
                            b_inf_sigmoid.squeeze(),
                            alpha_chw,
                        )
                    if self.config.alpha_bg_opacities:
                        alpha_bg_loss = alpha_bg_loss + self.alpha_bg_criterion(
                            self.colors.detach(),  # [N,3]
                            b_inf_sigmoid.squeeze(),  # [3]
                            torch.sigmoid(self.opacities).squeeze(),  # [N]
                        )

                elif self.config.bg_from_backscatter:
                    if self.config.turn_off_bg_loss:
                        alpha_bg_loss = torch.tensor(0.0, device=self.device)
                    else:
                        alpha_bg_loss = self.alpha_bg_criterion(
                            medium_chw.detach(),
                            torch.sigmoid(self.learned_bg.detach()),
                            alpha_chw,
                        )

            loss_dict["alpha_bg"] = self.config.bg_lambda * alpha_bg_loss

        # Depth L1 loss vs GT depth
        if self.config.use_depth_l1_loss and "depth_image" in batch:
            gt_depth = batch["depth_image"].to(self.device)  # [H,W,1]
            loss_dict["depth_l1"] = self.config.depth_l1_lambda * torch.abs(depth - gt_depth).mean()

        # Depth smoothness loss (edge-aware)
        if self.config.use_depth_smooth_loss:
            loss_dict["depth_smooth"] = (
                self.config.depth_smooth_lambda
                * self.depth_smooth_criterion(gt_bchw, depth_bchw)
            )

        # Alpha smoothness loss
        if self.config.use_alpha_smooth_loss:
            loss_dict["alpha_smooth"] = (
                self.config.alpha_smooth_lambda
                * self.depth_smooth_criterion(gt_bchw, alpha_bchw)
            )

        # Gray world prior
        if self.config.use_gw_loss and step > self.config.gw_from_iter:
            if self.config.use_render_for_gw:
                gray_world_input = render_bchw
            elif self.config.gw_detach_alpha_bg and "clean_rgb_bg_detached" in outputs:
                gray_world_input = self._to_bchw(outputs["clean_rgb_bg_detached"])
            elif self.config.gw_reverse_J and seathru_forward:
                # J = direct.detach() / attenuation
                direct_bchw = self._to_bchw(outputs["direct"])
                attenuation_bchw = self._to_bchw(outputs["attenuation_map"])
                gray_world_input = direct_bchw.detach() / attenuation_bchw.clamp(min=1e-8)
            else:
                gray_world_input = clean_bchw

            if self.config.gw_filter_by_alpha > 0.0:
                mask = (
                    alpha.detach().squeeze(-1) > self.config.gw_filter_by_alpha
                )  # [H,W]
                # Flatten to [1, C, N] for the criterion
                gray_world_input = gray_world_input[:, :, mask]

            # [idea-003] Compute effective GW weight (constant or annealed)
            if self.config.use_gw_anneal:
                anneal_progress = min(
                    (step - self.config.gw_from_iter) / max(self.config.gw_anneal_steps, 1),
                    1.0,
                )
                gw_weight = (
                    self.config.gw_anneal_start
                    + (self.config.gw_anneal_end - self.config.gw_anneal_start) * anneal_progress
                )
            else:
                gw_weight = self.config.gw_loss_lambda

            loss_dict["gray_world"] = gw_weight * self.gw_criterion(
                gray_world_input
            )

        # RGB saturation loss — operates on clean_rgb (Gaussians), physically
        # meaningful even during early medium Phase 1 where it constrains the
        # feasible region for Gaussian colors alongside GW.
        if seathru_forward and self.config.use_rgb_sat_loss:
            loss_dict["rgb_sat"] = (
                self.config.sat_loss_lambda * self.rgb_sat_criterion(clean_bchw)
            )

        # SeaThru losses requiring active (unfrozen) medium — gradients flow
        # to medium model parameters (backscatter, attenuation, B_inf).
        # Gated off during early medium Phase 1 because medium is frozen.
        if seathru_forward and not early_medium:
            backscatter_detach_bchw = self._to_bchw(
                outputs["backscatter_depth_detached"]
            )
            attenuation_detach_bchw = self._to_bchw(
                outputs["attenuation_depth_detached"]
            )
            direct_bchw = self._to_bchw(outputs["direct"])
            backscatter_bchw = self._to_bchw(outputs["backscatter"])

            # DCP loss (dark channel prior on estimated direct signal)
            if self.config.use_dcp_loss:
                reverse_direct = gt_bchw.detach() - backscatter_detach_bchw
                dcp_loss = self.dcp_criterion(reverse_direct, depth_bchw.detach())
                loss_dict["dcp"] = self.config.dcp_loss_lambda * dcp_loss

            # RGB spatial variation loss
            if self.config.use_rgb_sv_loss:
                loss_dict["rgb_sv"] = self.config.rgb_sv_lambda * self.rgb_sv_criterion(
                    clean_bchw.detach(), direct_bchw
                )

            # B_inf loss
            if self.config.use_binf_loss:
                loss_dict["binf"] = (
                    self.config.binf_loss_lambda
                    * self.backscatter_model.compute_binf_loss(clean_bchw.detach())
                )

            # β_D minimum regularization — prevent attenuation collapse to identity
            # Only applies to V3 (scalar β_D); V4 uses depth-dependent params
            if self.config.use_beta_d_min_reg and isinstance(self.attenuation_model, AttenuateNetV3):
                beta_d = self.attenuation_model.attenuation_conv_params.squeeze()
                beta_d_min = torch.tensor(
                    [self.config.beta_d_min_r, self.config.beta_d_min_g, self.config.beta_d_min_b],
                    device=beta_d.device, dtype=beta_d.dtype,
                )
                # softplus(min - val): smooth penalty when val < min, ~0 when val > min
                loss_dict["beta_d_min"] = (
                    self.config.beta_d_min_reg_lambda
                    * torch.nn.functional.softplus(beta_d_min - beta_d).mean()
                )

            # [idea-012v01] Attenuation magnitude — prevent near-identity T(z)
            if self.config.use_attn_magnitude_loss and "attenuation_map" in outputs:
                attn_map = self._to_bchw(outputs["attenuation_map"])
                # Mean attenuation effect: how much light is actually attenuated
                # T(z) near 1.0 means no attenuation; (1 - T(z)) measures the effect
                attn_effect = (1.0 - attn_map).mean()
                # Penalize when the mean effect is below the floor
                loss_dict["attn_magnitude"] = (
                    self.config.attn_magnitude_lambda
                    * torch.relu(self.config.attn_magnitude_floor - attn_effect)
                )

            # [idea-011-B] Clean render saturation — penalize clean_rgb outside [0, 1]
            if self.config.use_clean_saturation_loss:
                loss_dict["clean_sat"] = (
                    self.config.clean_sat_lambda
                    * (torch.relu(-clean_bchw) + torch.relu(clean_bchw - 1.0)).square().mean()
                )

            # [idea-011-C] Variance preservation — clean and medium spatial stats should match
            if self.config.use_variance_preservation and "medium_rgb" in outputs:
                medium_bchw = self._to_bchw(outputs["medium_rgb"])
                clean_std = torch.std(clean_bchw, dim=[2, 3])
                medium_std = torch.std(medium_bchw, dim=[2, 3])
                loss_dict["var_preservation"] = (
                    self.config.var_preservation_lambda
                    * torch.nn.functional.mse_loss(clean_std, medium_std)
                )

            # [idea-011-D] Channel ratio ordering — enforce β_D_R > β_D_G > β_D_B
            # Only applies to V3 (scalar β_D); V4 ordering is implicit in double exponential
            if self.config.use_beta_d_ordering and isinstance(self.attenuation_model, AttenuateNetV3):
                beta_d = self.attenuation_model.attenuation_conv_params.squeeze()
                # Penalize when green >= red or blue >= green
                loss_dict["beta_d_ordering"] = (
                    self.config.beta_d_ordering_lambda
                    * (torch.relu(beta_d[1] / beta_d[0].clamp(min=1e-8) - 1.0)
                       + torch.relu(beta_d[2] / beta_d[1].clamp(min=1e-8) - 1.0))
                )

            # DSC attenuation loss
            if self.config.use_dsc_attenuation_loss:
                reverse_direct_detached = (gt_bchw - backscatter_bchw).detach()
                if self.config.disable_attenuation:
                    J = torch.zeros_like(reverse_direct_detached)
                else:
                    J = reverse_direct_detached / attenuation_detach_bchw.clamp(
                        min=1e-8
                    )
                loss_dict["dsc_attenuation"] = (
                    self.config.dsc_attenuation_lambda
                    * self.dsc_attenuation_criterion(reverse_direct_detached, J)
                )
        else:
            # TODO: Compute Pre-SeaThru DCP but NOT added to loss in original SeaSplat (for logging if needed)
            pass

        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor) and v.dim() != 0:
                CONSOLE.log(
                    f"[WARNING] loss '{k}' is non-scalar "
                    f"(shape={v.shape}), reducing with .mean()"
                )
                loss_dict[k] = v.mean()

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """_summary_

        Args:
            outputs (Dict[str, torch.Tensor]): _description_
            batch (Dict[str, torch.Tensor]): _description_

        Returns:
            Tuple[Dict[str, float], Dict[str, torch.Tensor]]: _description_
        """
        gt_rgb = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )  # [H,W,3]

        # Select the predicted image -- medium if available, else clean
        if "medium_rgb" in outputs:
            pred_rgb = outputs["medium_rgb"]
        else:
            pred_rgb = outputs["clean_rgb"]
        seathru_forward = "medium_rgb" in outputs

        # Clamp for safety
        pred_rgb = torch.clamp(pred_rgb, 0.0, 1.0)
        gt_rgb = torch.clamp(gt_rgb, 0.0, 1.0)

        combined_rgb = torch.cat([gt_rgb, pred_rgb], dim=1)

        # Metrics: PSNR, SSIM, LPIPS -- in [1,C,H,W] format
        gt_bchw = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        pred_bchw = torch.moveaxis(pred_rgb, -1, 0)[None, ...]

        # SSIM decomposition (luminance, contrast, structure)
        ssim_val, ssim_l, ssim_c, ssim_s = _compute_ssim_components(gt_bchw, pred_bchw)

        # LPIPS per-layer decomposition (AlexNet, 5 layers)
        lpips_total, lpips_layers = self.lpips.net(
            gt_bchw, pred_bchw, retperlayer=True, normalize=True
        )

        metrics_dict = {
            # Primary metrics (nerfstudio compatibility)
            # When seathru active: these are medium metrics
            # When seathru inactive: these are clean metrics
            "psnr": float(self.psnr(gt_bchw, pred_bchw).item()),
            "ssim": ssim_val,
            "ssim_luminance": ssim_l,
            "ssim_contrast": ssim_c,
            "ssim_structure": ssim_s,
            "lpips": float(lpips_total.item()),
            "lpips_layer1": float(lpips_layers[0].item()),
            "lpips_layer2": float(lpips_layers[1].item()),
            "lpips_layer3": float(lpips_layers[2].item()),
            "lpips_layer4": float(lpips_layers[3].item()),
            "lpips_layer5": float(lpips_layers[4].item()),
        }

        # Clean (out-of-medium) metrics — scene without water effects
        # Only meaningful when seathru is active (otherwise clean == primary)
        if seathru_forward:
            clean_rgb = torch.clamp(outputs["clean_rgb"], 0.0, 1.0)
            clean_bchw = torch.moveaxis(clean_rgb, -1, 0)[None, ...]
            clean_ssim, _, _, _ = _compute_ssim_components(gt_bchw, clean_bchw)
            clean_lpips_total, _ = self.lpips.net(
                gt_bchw, clean_bchw, retperlayer=True, normalize=True
            )
            metrics_dict["clean_psnr"] = float(self.psnr(gt_bchw, clean_bchw).item())
            metrics_dict["clean_ssim"] = clean_ssim
            metrics_dict["clean_lpips"] = float(clean_lpips_total.item())

        images_dict: Dict[str, torch.Tensor] = {"img": combined_rgb}

        # Depth visualization
        depth = outputs.get("depth_processed")
        if depth is not None:
            d_vis = depth / depth.max().clamp(min=1e-8)
            images_dict["depth"] = d_vis.repeat(1, 1, 3)  # grayscale -> RGB

        # Alpha visualization
        alpha = outputs.get("accumulation")
        if alpha is not None:
            images_dict["alpha"] = alpha.repeat(1, 1, 3)

        # Medium model visualizations
        if "backscatter" in outputs:
            images_dict["backscatter"] = torch.clamp(outputs["backscatter"], 0.0, 1.0)
        if "attenuation_map" in outputs:
            images_dict["attenuation_map"] = torch.clamp(
                outputs["attenuation_map"], 0.0, 1.0
            )
        if "direct" in outputs:
            images_dict["direct"] = torch.clamp(outputs["direct"], 0.0, 1.0)
        if "medium_rgb" in outputs:
            images_dict["medium"] = torch.clamp(outputs["medium_rgb"], 0.0, 1.0)
        if "rendered_image" in outputs:
            images_dict["clean_render"] = torch.clamp(
                outputs["rendered_image"], 0.0, 1.0
            )
        if "marine_snow" in outputs:
            # Scale up for visibility (snow is typically very sparse/dim)
            images_dict["marine_snow"] = torch.clamp(
                outputs["marine_snow"] * 5.0, 0.0, 1.0
            )

        return metrics_dict, images_dict

    # Helpers
    def _is_medium_step(self, step: int) -> bool:
        """Check if the current step is a medium update step.

        With medium_window_size=1 (default), this matches the existing behavior:
        one medium step every medium_update_interval steps.

        With medium_window_size=W, runs W consecutive medium steps in every
        (medium_update_interval + W) cycle.
        """
        window = self.config.medium_window_size
        interval = self.config.medium_update_interval
        cycle = interval + window - 1  # total cycle length
        return (step % cycle) < window

    @staticmethod
    def _to_bchw(hwc: torch.Tensor) -> torch.Tensor:
        """Convert [H, W, C] to [1, C, H, W]"""
        return hwc.permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def _to_hwc(bchw: torch.Tensor) -> torch.Tensor:
        """Convert [1, C, H, W] to [H, W, C]"""
        return bchw.squeeze(0).permute(1, 2, 0)

    def _process_depth(self, depth: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Process raw expected depth from the rasterizer.

        Source: train.py lines 222-237.

        Steps:
          1. Divide by alpha (convert from accumulated to per-surface depth).
          2. Replace NaN / Inf with the maximum valid depth value.
          3. Divide by ``normalize_depth``.
          4. Min-max normalize to [0, 1] if ``norm_depth_max`` is set.

        Args:
            depth: [H, W, 1] raw expected depth from rasterization.
            alpha: [H, W, 1] accumulated alpha / opacity.

        Returns:
            [H, W, 1] processed depth.
        """
        if not self.config.filter_depth:
            return depth

        d = depth / alpha.clamp(min=1e-8)

        # Replace NaN / Inf
        valid = torch.isfinite(d)
        if not valid.all():
            valid_vals = d[valid]
            fill = valid_vals.max().item() if valid_vals.numel() > 0 else 100.0
            d = torch.where(
                valid, d, torch.tensor(fill, device=d.device, dtype=d.dtype)
            )

        d = d / self.config.normalize_depth

        if self.config.norm_depth_max:
            d_min, d_max = d.min(), d.max()
            if d_min != d_max:
                d = (d - d_min) / (d_max - d_min)
            else:
                d = d / d_max.clamp(min=1e-8)

        return d

    def _freeze_gs_params(self, freeze: bool, colors_only: bool = False) -> None:
        """Toggle ``requires_grad`` on Gaussian parameter groups.

        Args:
            freeze: If True, set requires_grad=False; else True.
            colors_only: If True and freeze=True, only freeze non-color
                params (keep features_dc/features_rest unfrozen).
                Source: train.py line 189 -- freeze everything except colors.
        """
        # Determine which params are "color" params (kept unfrozen when colors_only=True)
        color_params = {"features_dc", "features_rest"}
        for name, param in self.gauss_params.items():
            if colors_only and name in color_params:
                param.requires_grad_(True)
            else:
                param.requires_grad_(not freeze)

    def _freeze_medium_params(self, freeze: bool) -> None:
        """Toggle ``requires_grad`` on medium model parameters."""
        if self.backscatter_model is not None:
            for p in self.backscatter_model.parameters():
                p.requires_grad_(not freeze)
        if self.attenuation_model is not None:
            for p in self.attenuation_model.parameters():
                p.requires_grad_(not freeze)

    # Callbacks
    def _seasplat_before_iteration(self, step: int) -> None:
        """BEFORE_TRAIN_ITERATION callback -- manage training phase state.

        IMPORTANT: We do NOT toggle ``requires_grad`` here because gsplat's
        ``DefaultStrategy.step_pre_backward()`` (called inside
        ``super().get_outputs()``) needs ``requires_grad=True`` on GS params
        to call ``.retain_grad()``.  Instead, we only update phase-tracking
        flags; the actual selective optimization is handled in
        ``step_post_backward()`` by nulling out gradients for groups that
        should not be updated.

        Phases (source: train.py lines 174-192, 430-463):
          (a) GS freeze / unfreeze at configurable iterations.
          (b) SeaThru activation at seathru_from_iter:
              - Freeze GS params (except colors) for 1 iteration.
              - Initialize B_inf from learned_bg.
              - Start 1000-step medium-only warm-up burst.
          (c) After warm-up: ~2000-step GS color adjustment with interleaved
              medium steps every medium_update_interval iterations.
          (d) Joint training with interleaved medium steps every
              medium_update_interval iterations (handled in
              step_post_backward, not here).
        """
        # --- (a) GS freeze / unfreeze ---
        # Tracked via self._gs_frozen; enforced in step_post_backward.
        if step == self.config.freeze_gs_from_iter:
            CONSOLE.log(f"[INFO] [Step {step}] Freezing GS params (except colors)")
            self._gs_frozen = True
        if step == self.config.unfreeze_gs_from_iter:
            CONSOLE.log(f"[INFO] [Step {step}] Unfreezing GS params")
            self._gs_frozen = False

        if not self.config.do_seathru:
            return

        # --- (b) SeaThru activation ---
        if step > self.config.seathru_from_iter and not self.seathru_active:
            self.seathru_active = True
            self._gs_frozen = True
            self._phase2_onset_step = step
            CONSOLE.log(
                f"[INFO] [Step {step}] SeaThru activated; "
                "GS frozen (except colors) for 1 iter"
            )

            # Initialize B_inf from learned_bg
            # Source: train.py lines 208-212
            if (
                self.config.learn_background
                and self.config.bg_from_backscatter
                and not self.done_binf_init_with_bg
                and self.backscatter_model is not None
            ):
                with torch.no_grad():
                    self.backscatter_model.B_inf.data.copy_(
                        self.learned_bg.data.reshape(3, 1, 1)
                    )
                self.done_binf_init_with_bg = True
                CONSOLE.log(
                    f"[INFO] [Step {step}] B_inf initialized from learned_bg = "
                    f"{torch.sigmoid(self.learned_bg).tolist()}"
                )

            # Start initial 1000-step medium-only warm-up burst
            self._in_medium_burst = True
            self.warmup_counter = 0

        # Unfreeze GS 1 step after seathru activation
        # Source: train.py lines 190-192
        if step == self.config.seathru_from_iter + 2 and self._gs_frozen:
            self._gs_frozen = False
            CONSOLE.log(f"[INFO] [Step {step}] Unfreezing GS params (seathru +2)")

        # --- Alternating optimization state machine ---
        # Source: train.py lines 433-463
        if not self.seathru_active:
            return

        if self._in_medium_burst:
            # Initial warm-up burst only (1000 consecutive medium-only steps).
            # Periodic Phase 3 updates are interleaved, not bursted — handled
            # in step_post_backward().
            # [idea-008] Use shorter warmup when early medium was active
            # [idea-015] Extended medium-only phase overrides normal warmup
            if self.config.staged_medium_only_steps > 0:
                warmup_target = self.config.staged_medium_only_steps
            elif self.config.use_early_medium:
                warmup_target = self.config.early_medium_warmup_steps
            else:
                warmup_target = self.config.medium_warmup_steps
            if self.warmup_counter >= warmup_target:
                self.warmup_counter = 0
                self._in_medium_burst = False
                CONSOLE.log(
                    f"[INFO] [Step {step}] Medium warm-up complete ({warmup_target} steps)"
                )
                self.medium_inited = True
                self.adjust_gs_colors_for_color_correction = True
            else:
                self.warmup_counter += 1
                self.medium_steps_total += 1

        elif self.adjust_gs_colors_for_color_correction:
            if self.gs_color_correction_counter >= self.config.cc_phase_steps:
                CONSOLE.log(f"[INFO] [Step {step}] GS color adjustment complete")
                self.adjust_gs_colors_for_color_correction = False
            elif step % self.config.medium_update_interval == 0:
                # Interleaved medium step — don't count toward CC progress
                pass
            else:
                self.gs_color_correction_counter += 1

        else:
            # Phase 3: joint training — interleaved medium steps handled
            # in step_post_backward() (1 medium step every
            # medium_update_interval iterations, matching reference
            # train.py:434).

            # Record Phase 3 onset for LR decay scheduling
            if self._phase3_onset_step < 0:
                self._phase3_onset_step = step
                CONSOLE.log(f"[INFO] [Step {step}] Phase 3 onset (joint training)")
                if self.config.use_medium_lr_decay:
                    initial_lr = self.config.backscatter_attenuation_lr
                    final_lr = initial_lr * self.config.medium_lr_decay_factor
                    CONSOLE.log(
                        f"[INFO] [Step {step}] Medium LR decay enabled: "
                        f"{initial_lr:.1e} → {final_lr:.1e} over "
                        f"{self.config.medium_lr_decay_steps} steps"
                    )

            # [idea-002] Apply medium LR decay during Phase 3
            if self.config.use_medium_lr_decay and self._phase3_onset_step >= 0:
                progress = min(
                    (step - self._phase3_onset_step) / self.config.medium_lr_decay_steps,
                    1.0,
                )
                # Exponential interpolation: initial_lr at progress=0,
                # initial_lr * decay_factor at progress=1
                new_lr = self.config.backscatter_attenuation_lr * (
                    self.config.medium_lr_decay_factor ** progress
                )
                for name in ("backscatter_model", "attenuation_model"):
                    if hasattr(self, "optimizers") and name in self.optimizers:
                        for pg in self.optimizers[name].param_groups:
                            pg["lr"] = new_lr
