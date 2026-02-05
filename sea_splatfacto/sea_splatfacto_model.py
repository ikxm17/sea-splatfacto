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
class TemplateModelConfig(NerfactoModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: TemplateModel)


class TemplateModel(NerfactoModel):
    """Template Model."""

    config: TemplateModelConfig

    def populate_modules(self):
        super().populate_modules()

    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.
