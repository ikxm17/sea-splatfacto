"""
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""

import torch

from torch.nn import Parameter
from gsplat.strategy import DefaultStrategy, MCMCStrategy

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

from pytorch_msssim import SSIM

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

from deepseecolor.models import BackscatterNetV2, AttenuateNetV3
from deepseecolor.losses import (
    AttenuateLoss,
    DarkChannelPriorLossV3,
    GrayWorldPriorLoss,
    SmoothDepthLoss,
    RGBSpatialVariationLoss,
    RGBSaturationLoss,
    AlphaBackgroundLoss,
    mixture_of_laplacians_loss,
)

from utils.general_utils import inverse_sigmoid
from utils.loss_utils import depth_weighted_l1_loss, depth_weighted_l2_loss

@dataclass
class SeaSplatfactoModelConfig(SplatfactoModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: SeaSplatfactoModel)

    # Override Splatfacto defaults
    output_depth_during_training: bool = True # rendered depth is needed at every step for the medium models
    background_color: Literal["black", "white", "learned"] = "black" # background compositing is handled ourselves via learned_background, so the base renderer shoudl composite against black (i.e. contribute nothing)
    
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
    use_alpha_smooth_loss: bool = True
    """Apply the same edge-aware smoothness loss to the alpha/accumulation map."""
    alpha_smooth_lambda: float = 1.0
    """Weight for the alpha smoothness loss."""
    
    # Opacity prior loss
    use_opacity_prior: bool = False
    """Mixture-of-Laplacians prior pushing opacities toward 0 or 1."""
    opacity_prior_lambda: float = 0.0001
    """Weight for the opacity prior loss."""
    
    # Depth smoothness loss
    use_depth_smooth_loss: bool = True
    """Edge-aware depth smoothness loss weighted by RGB gradients."""
    depth_smooth_lambda: float = 2.0
    """Weight for depth smoothness loss."""
    
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
    
    # RGB saturation loss
    use_rgb_sat_loss: bool = True
    """Penalize rendered pixel values outside [0, saturation_val]."""
    sat_loss_lambda: float = 2.0
    """Weight for the RGB saturation loss."""
    
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
    seathru_from_iter: int = 15_000
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
    use_at_v2: bool = False
    """Use AttenuateNetV2 (drops some terms) instead of the default."""
    use_at_v3: bool = True
    """Use AttenuateNetV3 (simplest) — the default attenuation model."""
    disable_attenuation: bool = False
    """Simplified model that only accounts for backscatter (no attenuation)."""
    update_bs_at_interval: int = 100
    """Every this many GS training steps, perform a burst of medium-only
    updates."""
    update_bs_at_count: int = 50
    """Number of consecutive medium-only optimizer steps per burst."""
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

    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.

        # === Initialize Underwater Models (train.py lines 63-72) ===
        if self.config.do_seathru:
            # BackscatterNetV2 initialization (train.py line 64)
            self.backscatter_model = BackscatterNetV2(
                use_residual=self.config.use_backscatter_residual,
                scale=self.config.backscatter_scale,
                do_sigmoid=self.config.backscatter_do_sigmoid,
            ).to(self.device)
            # AttenuateNetV3 initialization (train.py lines 65-70)
            self.attenuation_model = AttenuateNetV3(
                scale=self.config.attenuation_scale,
                do_sigmoid=self.config.attenuation_do_sigmoid,
                init_vals=not self.config.attenuation_do_sigmoid,
            )
        else:
            self.backscatter_model = None
            self.attenuation_model = None

        # === Initialize Loss Criteria (train.py lines 74-84) ===
        self.depth_smooth_critetion = SmoothDepthLoss().to(self.device)
        self.gw_criterion = GrayWorldPriorLoss().to(self.device)
        self.rgb_sv_criterion = RGBSpatialVariationLoss().to(self.device)
        self.rgb_01_criterion = RGBSaturationLoss(saturation_limit=1.0).to(self.device)
        self.rgb_sat_criterion = RGBSaturationLoss(saturation_limit=0.7).to(self.device)
        self.alpha_bg_criterion = AlphaBackgroundLoss(use_kornia=False).to(self.device)
        self.dsc_attenuation_criterion = AttenuateLoss().to(self.device)
        self.dcp_criterion = DarkChannelPriorLossV3().to(self.device)

        # === To learn the background ===
        if self.config.learn_background:
            bg_init = self.rand(3, device=self.device)
            bg_init[2] = 0.8
            bg_init[1] = 0.24
            bg_init[0] = 0.05
            self.learned_bg = torch.nn.Parameter(
                inverse_sigmoid(bg_init.requires_grad_(True))
            )
        else:
            self.learned_bg = None

        # === State variables for training ===
        self.backscatter_inited = False
        self.attenuation_inited = False
        self.backscatter_update_counter = 0
        self.attenuation_update_counter = 0
        self.done_binf_init_with_bg = False
        self.adjust_gs_colors_for_cc = False
        self.update_gs_color_counter = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """_summary_

        Returns:
            Dict[str, List[Parameter]]: _description_
        """
        # Get base parameter groups
        gps = super().get_param_groups()

        # Add learned background parameters
        if self.config.learn_background and self.learned_bg is not None:
            gps["learned_bg"] = list(self.learned_bg) # type: ignore

        # add underwater model parameters
        if (
            self.config.do_seathru
            and self.backscatter_model is not None
            and self.attenuation_model is not None
        ):
            gps["backscatter_model"] = list(self.backscatter_model.parameters())
            gps["attenuation_model"] = list(self.attenuation_model.parameters())

        return gps

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """_summary_

        Args:
            camera (Cameras): _description_

        Returns:
            Dict[str, Union[torch.Tensor, List]]: _description_
        """
        # get base outputs (includes rgb, depth, accumulation, background)
        outputs = super().get_outputs(camera)

        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a Cameras instance")
            return {}
        
        # [H, W, 3]
        raw_rgb = outputs["rgb"].clone() # type: ignore
        # [H, W, 1]
        alpha_image = outputs["alpha"].clone() # type: ignore
        
        # TODO: should assign a more meaningful name than "image"
        if self.config.learn_background:
            if self.config.bg_from_backscatter and self.config.do_seathru and self.step > self.config.seathru_from_iter:
                image = raw_rgb
                if not self.done_binf_init_with_bg:
                    print(f"{self.step}: Updated backscatter model's B_inf with learned background parameters")
                    self.backscatter_model.B_inf = torch.nn.Parameter(torch.clone(self.learned_bg.data).reshape(3, 1, 1).to(self.device)) # type: ignore
            else:
                bg_image = torch.sigmoid(self.learned_bg).reshape(3, 1, 1) * (1 - alpha_image) # type: ignore
                bg_detached_image = torch.sigmoid(self.learned_bg).reshape(3, 1, 1) * (1 - alpha_image) # type: ignore
                image = raw_rgb + bg_image
        else:
            image = raw_rgb
        
        # [H, W, 1] or None
        depth_image = outputs["depth"].clone() # type: ignore
        # ? What is going on here?
        if self.config.filter_depth:
            depth_image = depth_image = depth_image / alpha_image
            if torch.any(torch.isnan(torch.logical_or(torch.isnan(depth_image), torch.isinf(depth_image)))):
                valid_depth_vals = depth_image[torch.logical_not(torch.logical_or(torch.isnan(depth_image), torch.isinf(depth_image)))]
                if len(valid_depth_vals) == 0:
                    print(f"[Training] everything is NaN)")
                else:
                    not_nan_max = torch.max(valid_depth_vals).item()
                depth_image = torch.nan_to_num(depth_image, not_nan_max, not_nan_max)
            depth_image = depth_image / self.config.normalize_depth
            if self.config.norm_depth_max:
                if depth_image.min() != depth_image.max():
                    depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
                else:
                    depth_image = depth_image / depth_image.max()
        
        # TODO: if ground truth depth image is available, replace `depth_image` with it here   
        outputs["processed_depth"] = depth_image          
        
        if self.config.do_seathru and self.step > self.config.seathru_from_iter:
            # reshape from nerfstudio tensor format [H, W, C] to PyTorch tensor format [1, C, H, W]
            image_batch = image.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
            depth_image_batch = depth_image.permute(2, 0, 1).unsqueeze(0) # [1, 1, H, W]
            
            # estimate attenuation
            if self.config.disable_attenuation:
                direct_image = image_batch
            else:
                attenuation = self.attenuation_model(depth_image_batch) # type: ignore
                direct_image = image_batch * attenuation
            outputs["attenuation"] = attenuation.squeeze(0).permute(1, 2, 0) # convert to nerfstudio tensor format [H, W, 3]
            outputs["direct_image"] = direct_image.squeeze(0).permute(1, 2, 0) # convert to nerfstudio tensor format [H, W, 3]
            
            # TODO: add z-score
            
            # estimate backscatter
            backscatter = self.backscatter_model(depth_image_batch) # type: ignore
            outputs["backscatter"] = backscatter.squeeze(0).permute(1, 2, 0) # convert to nerfstudio tensor format [H, W, 3]
            
            # underwater image formation
            underwater_image = torch.clamp(direct_image + backscatter, 0.0, 1.0)
            outputs["underwater_image"] = underwater_image.squeeze(0).permute(1, 2, 0) # convert to nerfstudio tensor format [H, W, 3]
        return outputs
        
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
        loss_dict = dict()
        
        # get ground truth image
        gt_image = self.composite_with_background(
            self.get_gt_image(batch["image"]), outputs["background"]
        )
        
        # get predicted image
        if self.config.do_seathru and self.step > self.config.seathru_from_iter:
            pred_image = outputs["underwater_image"]
        else:
            pred_image = outputs.get("image", outputs["rgb"]) 
        
        # get depth image
        depth_image = outputs["depth"]
        
        if self.config.use_depth_weighted_l1:
            Ll1 = depth_weighted_l1_loss(pred_image, gt_image, depth_image.detach())
        elif self.config.use_depth_weighted_l2:
            Ll1 = depth_weighted_l2_loss(pred_image, gt_image, depth_image.detach())
        else:
            Ll1 = torch.abs(gt_image - pred_image).mean()
            
        simloss = 1 - self.ssim(gt_image.permute(2, 0, 1)[None, ...], pred_image.permute(2, 0, 1)[None, ...])
        
        # main loss implemented by splatfacto
        main_loss = (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss # main_loss from SplatfactoModel's implementation
        loss_dict["main_loss"] = main_loss
        
        # depth weighted l1
        if self.config.add_recon_depth_l1 and "processed_depth" in outputs:
            dl1 = depth_weighted_l1_loss(pred_image, gt_image, depth_image.detach())
            depth_weighted_l1 = self.config.dwr_lambda * dl1
            loss_dict["depth_weighted_l1_loss"] = depth_weighted_l1
        
        # binary accumulation loss
        if self.config.use_opacity_prior:
            opacity_prior_loss = mixture_of_laplacians_loss(self.get_gaussian_param_groups()["opacities"]) # TODO: check whether this corresponds to scene.gaussians.get_opacity
            loss_dict["opacity_prior_loss"] = self.config.opacity_prior_lambda * opacity_prior_loss        
        
        # TODO: add the other losses for alpha_bg_loss
        # alpha background loss
        if self.config.learn_background:
            rendered_image = outputs["rgb"]
            underwater_image = outputs["underwater_image"]
            alpha_image = outputs["alpha"]
            alpha_bg_loss = self.alpha_bg_criterion(rendered_image.detach(), torch.sigmoid(self.learned_bg.detach()), alpha_image) # type: ignore
            if self.config.do_seathru and self.step > self.config.seathru_from_iter:
                if self.config.alpha_binf_uw:
                    alpha_bg_loss = self.alpha_bg_criterion(underwater_image.squeeze().detach(), torch.sigmoid(self.backscatter_model.B_inf).squeeze().detach(), alpha_image) # type: ignore
            loss_dict["alpha_bg_loss"] = self.config.bg_lambda * alpha_bg_loss
        
        # depth smooth loss
        if self.config.use_depth_smooth_loss:
            gt_image_batch = gt_image.permute(2, 0, 1).unsqueeze(0)
            depth_image_batch = outputs["processed_depth"].permute(2, 0, 1).unsqueeze(0)
            depth_smooth_loss = self.depth_smooth_critetion(gt_image_batch, depth_image_batch)
            loss_dict["depth_smooth_loss"] = self.config.depth_smooth_lambda * depth_smooth_loss
        
        # gray world loss  
        if self.config.use_gw_loss and self.step > self.config.gw_from_iter:
            image_batch = outputs["image"].permute(2, 0, 1).unsqueeze(0)
            gw_loss = self.gw_criterion(image_batch)
            loss_dict["gray_world_loss"] = self.config.gw_loss_lambda * gw_loss
        
        # dark channel prior loss
        if self.config.use_dcp_loss:
            gt_image_batch = gt_image.permute(2, 0, 1).unsqueeze(0)
            depth_image_batch = outputs["processed_depth"].permute(2, 0, 1).unsqueeze(0)
            if self.config.do_seathru and self.step > self.config.seathru_from_iter:
                backscatter_depth_detached = self.backscatter_model(depth_image_batch.detach()) # type: ignore
                direct_reversed = gt_image_batch.detach() - backscatter_depth_detached # direct reverse from ground truth through backscatter model
                dcp_loss, _ = self.dcp_criterion(direct_reversed, depth_image_batch.detach())
            else:
                dcp_loss, _ = self.dcp_criterion(gt_image_batch.detach(), depth_image_batch.detach())
            loss_dict["dark_channel_prior_loss"] = self.config.dcp_loss_lambda * dcp_loss
                
        
        # losses that only apply when seathru
        if self.config.do_seathru and self.step > self.config.seathru_from_iter:
            # TODO: rgb spatial variation loss
            # rgb saturation loss
            if self.config.use_rgb_sat_loss:
                image_batch = outputs["image"].permute(2, 0, 1).unsqueeze(0)
                rgb_sat_loss = self.rgb_sat_criterion(image_batch)
                loss_dict["rgb_saturation_loss"] = self.config.sat_loss_lambda * rgb_sat_loss
            # TODO: B_inf loss
            # dsc attenuation loss
            if self.config.use_dsc_attenuation_loss:
                gt_image_batch = gt_image.permute(2, 0, 1).unsqueeze(0)
                depth_image_batch = outputs["processed_depth"].permute(2, 0, 1).unsqueeze(0)
                attenuation = outputs["attenuation"].permute(2, 0, 1).unsqueeze(0)
                attenuation_depth_detached = attenuation_model(depth_image_batch.detach()) # type: ignore
                direct_reversed = (gt_image_batch - outputs["backscatter"].permute(2, 0, 1).unsqueeze(0))
                if self.config.disable_attenuation:
                    J_through_attenuation = torch.zeros_like(direct_reversed)
                else:
                    J_through_attenuation = direct_reversed / attenuation_depth_detached
                dsc_attenuation_loss = self.dsc_attenuation_criterion(direct_reversed, J_through_attenuation)
                loss_dict["dsc_attenuation_loss"] = self.config.dsc_attenuation_lambda * dsc_attenuation_loss
        
        # from parent: scale regulaization
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            loss_dict["scale_reg"] = 0.1 * scale_reg.mean()
        else:
            loss_dict["scale_reg"] = torch.tensor(0.0, device=self.device)
            
        # from parent: add camera optimizer loss if training
        if self.training:
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid:
                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)
        
        return loss_dict
    # Helpers
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
            d = torch.where(valid, d, torch.tensor(fill, device=d.device, dtype=d.dtype))

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
        for name, param in self.gauss_params.items():
            if colors_only and name in ("features_dc", "features_rest"):
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
