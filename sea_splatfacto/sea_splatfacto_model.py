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

    _target: Type = field(default_factory=lambda: SplatfactoModel)

    # === SeaThru Core Parameters ===
    do_seathru: bool = True
    """Enable underwater image formation model"""
    seathru_from_iter: int = 5000
    """Iteration to start seathru modeling (original default: 9_000_000, set lower to enable)"""
    backscatter_attenuation_lr: float = 1e-2
    """Learning rate for backscatter/attenuation models"""
    backscatter_scale: float = 5.0
    """Scale for backscatter parameters"""
    attenuation_scale: float = 5.0
    """Scale for attenuation parameters"""
    backscatter_do_sigmoid: bool = False
    """Use sigmoid activation for backscatter params"""
    attenuation_do_sigmoid: bool = False
    """Use sigmoid activation for attenuation params"""
    use_backscatter_residual: bool = False
    """Use residual term in backscatter model"""
    use_at_v3: bool = True
    """Use AttenuateNetV3 (simplified)"""
    disable_attenuation: bool = False
    """Disable attenuation (backscatter only model)"""

    # === Update Schedule ===
    update_backscatter_attenuation_interval: int = 100
    """Every N GS updates, update backscatter/at models"""
    update_backscatter_attenuation_count: int = 50
    """Update backscatter/at models this many times per interval"""

    # === Background Parameters ===
    learn_background: bool = True
    """Learn background color composited with splat render"""
    bg_lambda: float = 0.01
    """Lambda for background alpha loss"""
    bg_from_backscatter: bool = True
    """Once seathru enabled, use B_inf instead of learned_bg"""
    bg_lr: float = 1e-2
    """Learning rate for background color"""
    alpha_binf_uw: bool = True
    """Use B_inf for alpha loss on underwater image"""

    # === Depth Parameters ===
    use_depth_smooth_loss: bool = True
    """Enable depth smoothness loss"""
    depth_smooth_lambda: float = 2.0
    """Lambda for depth smooth loss"""
    filter_depth: bool = True
    """Filter depth by alpha and normalize"""
    normalize_depth: float = 1.0
    """Normalize depth by this value"""
    norm_depth_max: bool = True
    """Normalize depth to [0, 1] range"""
    depth_alpha_threshold: float = 0.5
    """Alpha threshold for depth masking"""

    # === Gray World Loss ===
    use_gw_loss: bool = True
    """Enable gray world prior loss"""
    gw_loss_lambda: float = 0.1
    """Lambda for gray world loss"""
    gw_from_iter: int = 10000
    """Start gray world loss from this iteration"""

    # === Dark Channel Prior Loss ===
    use_dcp_loss: bool = True
    """Enable dark channel prior loss"""
    dcp_loss_lambda: float = 1.0
    """Lambda for DCP loss"""

    # === RGB Saturation Loss ===
    use_rgb_sat_loss: bool = True
    """Enable RGB saturation loss"""
    sat_loss_lambda: float = 2.0
    """Lambda for saturation loss"""

    # === Attenuation Loss ===
    use_dsc_attenuation_loss: bool = False
    """Enable DeepSeeColor attenuation loss"""
    dsc_attenuation_lambda: float = 1.0
    """Lambda for DSC attenuation loss"""

    # === Additional Losses ===
    use_depth_weighted_l1: bool = False
    use_depth_weighted_l2: bool = False
    
    use_alpha_smooth_loss: bool = False
    """Enable alpha smoothness loss (arguments.py line 97)"""
    alpha_smooth_lambda: float = 1.0

    use_opacity_prior: bool = False
    """Enable bimodal opacity prior (arguments.py line 99)"""
    opacity_prior_lambda: float = 0.0001

    add_recon_depth_l1: bool = True
    """Add depth-weighted L1 reconstruction loss (arguments.py line 100)"""
    dwr_lambda: float = 1.0

    # === Parameter Freezing ===
    freeze_gs_from_iter: int = 9_000_000
    """Freeze gaussian splat parameters from this iteration"""
    unfreeze_gs_from_iter: int = 9_000_000
    """Unfreeze gaussian splat parameters from this iteration"""


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
