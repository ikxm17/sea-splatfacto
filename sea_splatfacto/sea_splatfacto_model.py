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
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.math import k_nearest_sklearn, random_quat_tensor
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases

from sea_splatfacto.deepseecolor.models import BackscatterNetV2, AttenuateNetV3, AttenuateNetV4
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



class SeaSplatfactoModel(SplatfactoModel):
    """_summary_

    Args:
        SplatfactoModel (_type_): _description_
    """

    config: SeaSplatfactoModelConfig

    def populate_modules(self):
        super().populate_modules()

        self.backscatter_model: Optional[BackscatterNetV2] = None
        self.attenuation_model: Optional[nn.Module] = None

        if self.config.do_seathru:
            # Backscatter and attenuation models
            self.backscatter_model = BackscatterNetV2(
                use_residual=self.config.backscatter_use_residual,
                scale=self.config.backscatter_scale,
                do_sigmoid=self.config.backscatter_do_sigmoid,
            )
            if self.config.use_depth_dependent_beta_d:
                # [idea-012] Depth-dependent beta_D (double exponential, 12 params)
                self.attenuation_model = AttenuateNetV4(
                    scale=self.config.attenuation_scale,
                    do_sigmoid=self.config.attenuation_do_sigmoid,
                )
            else:
                self.attenuation_model = AttenuateNetV3(
                    scale=self.config.attenuation_scale,
                    do_sigmoid=self.config.attenuation_do_sigmoid,
                    init_vals=not self.config.attenuation_do_sigmoid,
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

        # Gradient magnitude tracking (recorded before nulling in step_post_backward)
        self._last_grad_bs: float = 0.0
        self._last_grad_at: float = 0.0

        CONSOLE.log(
            f"[INFO] do_seathru: {self.config.do_seathru}, "
            f"seathru_from_iter: {self.config.seathru_from_iter}, "
            f"disable_attenuation: {self.config.disable_attenuation}"
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
        seathru_forward = (
            self.config.do_seathru
            and self.backscatter_model is not None
            and self.attenuation_model is not None
            and (self.seathru_active or not self.training)
        )

        if self.config.learn_background:
            if self.config.bg_from_backscatter and seathru_forward:
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

            # backscatter
            backscatter_bchw = self.backscatter_model(depth_bchw)  # [1, 3, H, W]
            backscatter_detach_bchw = self.backscatter_model(depth_bchw.detach())

            # combined medium image (scene through water)
            medium_bchw = torch.clamp(direct_bchw + backscatter_bchw, 0.0, 1.0)

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

        modified_outputs = dict(outputs)
        if seathru_forward:
            pred_rgb = outputs["medium_rgb"]
        else:
            pred_rgb = outputs["clean_rgb"]
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

        step = self.step

        # Depth-weighted reconstruction L1
        if self.config.add_recon_depth_l1:
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

        # Alpha-background loss
        if self.config.learn_background:
            alpha_chw = alpha.permute(2, 0, 1)
            alpha_bg_loss = self.alpha_bg_criterion(
                rendered_image.permute(2, 0, 1).detach(),
                torch.sigmoid(self.learned_bg.detach()),
                alpha_chw,
            )

            if seathru_forward:
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

            loss_dict["gray_world"] = self.config.gw_loss_lambda * self.gw_criterion(
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
        if seathru_forward:
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

        return metrics_dict, images_dict

    # Helpers
    def _is_medium_step(self, step: int) -> bool:
        """Check whether the current step is a medium update step.

        One medium step every medium_update_interval steps; GS steps on
        all other iterations.
        """
        return (step % self.config.medium_update_interval) == 0

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
            if self.warmup_counter >= self.config.medium_warmup_steps:
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

