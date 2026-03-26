import math

import torch
import torch.nn as nn

from kornia.color import rgb_to_lab

class AttenuateLoss(nn.Module):
    """Regularize the restored image J = direct / attenuation.

    NOT from paper — extra loss from DeepSeeColor reference code.
    Combines three sub-losses on J:
    - Saturation: penalizes pixel values outside [0, 1] (squared ReLU)
    - Spatial variation: J should have similar per-channel std as the direct signal
    - Intensity: channel means should be near target_intensity (gray-world on J)

    Config: use_dsc_attenuation_loss (default False), dsc_attenuation_lambda (default 1.0)

    Input shapes:
        direct: [B, C, H, W] — attenuated clean signal (J * T(z))
        J: [B, C, H, W] — restored image (direct / T(z))
    """

    def __init__(self, target_intensity: float = 0.5):
        """Args:
            target_intensity: Target for channel mean intensity. Default: 0.5.
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.target_intensity = target_intensity

    def forward(self, direct: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        # Saturation: penalize J values outside [0, 1]
        saturation_loss = (torch.relu(-J) + torch.relu(J - 1)).square().mean()

        # Spatial variation: J should match direct's per-channel std
        init_spatial = torch.std(direct, dim=[2, 3])
        channel_spatial = torch.std(J, dim=[2, 3])
        spatial_variation_loss = self.mse(channel_spatial, init_spatial)

        # Intensity: push channel means toward target (gray-world on J)
        channel_intensities = torch.mean(J, dim=[2, 3], keepdim=True)
        intensity_loss = (
            (channel_intensities - self.target_intensity).square().mean()
        )

        if torch.any(torch.isnan(saturation_loss)):
            print("NaN saturation loss!")
        if torch.any(torch.isnan(spatial_variation_loss)):
            print("NaN spatial variation loss!")
        if torch.any(torch.isnan(intensity_loss)):
            print("NaN intensity loss!")

        return saturation_loss + spatial_variation_loss + intensity_loss


class DarkChannelPriorLossV3(nn.Module):
    """Dark channel prior — penalize negative direct signal (Eq 4).

    The direct signal D = J * T(z) should be non-negative (it represents
    attenuated scene radiance). Applies an asymmetric penalty:
    - Positive values: L1 toward zero (mild regularization)
    - Negative values: SmoothL1 * cost_ratio (heavy penalty, 1000x default)

    Paper: sum(max(D,0) + k*min(D,0)). Code uses SmoothL1 for smoother
    gradients near zero (L2 in [-beta, beta], L1 outside).

    Config: use_dcp_loss (default True), dcp_loss_lambda (default 1.0)

    Input shapes:
        direct: [B, C, H, W] — estimated direct signal (GT - backscatter)
    """

    def __init__(self, cost_ratio: float = 1000.0, beta: float = 0.2):
        """Args:
            cost_ratio: Penalty multiplier for negative values. Default: 1000.
            beta: SmoothL1 L2→L1 transition point. Default: 0.2.
        """
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.smooth_l1_loss = nn.SmoothL1Loss(beta=beta)
        self.cost_ratio = cost_ratio

    def forward(self, direct, depth=None):
        pos = self.l1_loss(torch.relu(direct), torch.zeros_like(direct))
        neg = self.smooth_l1_loss(torch.relu(-direct), torch.zeros_like(direct))
        return self.cost_ratio * neg + pos


class GrayWorldPriorLoss(nn.Module):
    """Gray world prior — push channel means toward neutral gray (Eq 5).

    Assumes a properly white-balanced natural image has approximately equal
    mean intensity across R, G, B. Applied to the clean/restored image J
    to drive the medium model toward correct color correction.

    Loss: (1/C) * sum_c (mean(J_c) - target)^2

    Config: use_gw_loss (default True), gw_loss_lambda (default 0.1)

    Input shapes:
        J: [B, C, H, W] or [B, C, N] (flattened spatial dims when alpha-filtered)
    """

    def __init__(self, target_intensity: float = 0.5):
        """Args:
            target_intensity: Target channel mean. Default: 0.5.
        """
        super().__init__()
        self.target_intensity = target_intensity

    def forward(self, J: torch.Tensor) -> torch.Tensor:
        if len(J.size()) == 4:  # B x C x H x W
            channel_mean_intensities = torch.mean(J, dim=[-2, -1], keepdim=True)
        elif len(J.size()) == 3:  # B x C x (H*W)
            channel_mean_intensities = torch.mean(J, dim=[-1])
        else:
            assert False
        intensity_loss = (
            (channel_mean_intensities - self.target_intensity).square().mean()
        )
        return intensity_loss


class SmoothDepthLoss(nn.Module):
    """Edge-aware depth smoothness loss (Eq 8).

    Penalizes depth gradients, weighted by image gradients:
        L = mean|d_dx * exp(-rgb_dx)| + mean|d_dy * exp(-rgb_dy)|

    The exp(-grad) weighting suppresses depth penalty at image edges where
    depth discontinuities are expected. Note: both reference and port omit
    abs() on image gradients vs paper — negative gradients amplify the penalty.

    Config: use_depth_smooth_loss (default True), depth_smooth_lambda (default 1.0)
    Also used for alpha smoothness (use_alpha_smooth_loss).

    Input shapes:
        rgb: [B, C, H, W] — ground truth image (provides edge guidance)
        depth: [B, 1, H, W] — predicted depth (or alpha) map
    """

    def __init__(self):
        super().__init__()
        super().__init__()

    def forward(self, rgb, depth):
        depth_dx = depth.diff(
            dim=-1
        )  # calulate differences across horizontal dimension; shape: B x 1 x H x (W-1)
        depth_dy = depth.diff(
            dim=-2
        )  # calculate differences across vertical dimension; shape: B x 1 x (H-1) x W

        rgb_dx = torch.mean(
            rgb.diff(dim=-1), dim=-3, keepdim=True
        )  # calculate mean of differences across horizontal dimension (W) over all color channels; shape: B x 1 x H x (W-1)
        rgb_dy = torch.mean(
            rgb.diff(dim=-2), dim=-3, keepdim=True
        )  # calculate mean of difference across vertical dimension (H) over all color channels; shape: B x 1 x (H-1) x W

        depth_dx *= torch.exp(-rgb_dx)
        depth_dy *= torch.exp(-rgb_dy)

        return torch.abs(depth_dx).mean() + torch.abs(depth_dy).mean()


class RGBSpatialVariationLoss(nn.Module):
    """Match spatial variation between clean image and direct signal.

    NOT from paper — extra loss from DeepSeeColor reference code.
    Penalizes difference in per-channel std: MSE(std(J), std(direct)).
    Encourages J to maintain similar spatial statistics after attenuation reversal.

    Config: use_rgb_sv_loss (default False), rgb_sv_lambda (default 0.01)

    Input shapes:
        J: [B, C, H, W] — clean/restored image (detached at call site)
        direct: [B, C, H, W] — attenuated clean signal
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, J, direct):
        init_spatial = torch.std(direct, dim=[2, 3])
        channel_spatial = torch.std(J, dim=[2, 3])
        spatial_variation_loss = self.mse(channel_spatial, init_spatial)
        if torch.any(torch.isnan(spatial_variation_loss)):
            print("NaN RGB Spatial Variation loss!")
        return spatial_variation_loss


class RGBSaturationLoss(nn.Module):
    """Penalize clean image pixel values outside [0, saturation_val] (Eq 6, extended).

    Paper Eq 6 only penalizes over-saturation: max(J - T_sat, 0).
    Code extends with below-zero penalty: (relu(-J) + relu(J - T_sat))^2.
    Both reference and port implement this extension.

    Config: use_rgb_sat_loss (default True), sat_loss_lambda (default 1.0)

    Input shapes:
        rgb: [B, C, H, W] — clean/restored image J
    """

    def __init__(self, saturation_val: float = 1.0):
        super().__init__()
        self.relu = nn.ReLU()
        self.saturation_val = saturation_val

    def forward(self, rgb):
        saturation_loss = (
            (self.relu(-rgb) + self.relu(rgb - self.saturation_val)).square().mean()
        )
        if torch.any(torch.isnan(saturation_loss)):
            print("NaN RGB Saturation loss!")
        return saturation_loss


class AlphaBackgroundLoss(nn.Module):
    """Push alpha toward zero where pixel color matches background (Eq 9).

    For pixels whose color is within a threshold of the background color,
    the opacity should be low — these regions are empty space. Prevents
    Gaussians from explaining background color with opaque splats.

    Two color distance modes:
    - RGB (default): L2 distance in unit RGB cube.
      Max distance = sqrt(3) ≈ 1.732, threshold = 0.2 * sqrt(3) ≈ 0.346
    - LAB (use_kornia=True): L2 distance in CIELAB space.
      Max distance = sqrt(100^2 + 255^2 + 255^2) ≈ 375, threshold = 50

    Note: the rgb parameter is overloaded — may receive rendered images
    [C,H,W], medium images [C,H,W], or per-Gaussian colors [N,3] depending
    on the alpha_bg config variant.

    Config: learn_background (enables), bg_lambda (default 1.0)
    """

    # Eq (9) color distance constants
    _RGB_MAX_DIST = math.sqrt(3)               # unit RGB cube diagonal
    _RGB_THRESHOLD = 0.2 * math.sqrt(3)        # 20% of max RGB distance
    _LAB_MAX_DIST = math.sqrt(100**2 + 255**2 + 255**2)  # CIELAB max distance
    _LAB_THRESHOLD = 50                         # CIELAB similarity threshold

    def __init__(self, use_kornia: bool = False):
        super().__init__()
        self.use_kornia = use_kornia
        self.l1 = nn.L1Loss()
        if use_kornia:
            self.range = self._LAB_MAX_DIST
            self.threshold = self._LAB_THRESHOLD
        else:
            self.range = self._RGB_MAX_DIST
            self.threshold = self._RGB_THRESHOLD

    def forward(self, rgb, background, alpha):
        # Compute per-pixel (or per-Gaussian) color distance to background
        if self.use_kornia:
            lab_image = rgb_to_lab(rgb)
            lab_background = rgb_to_lab(background.reshape(3, 1, 1))
            diff = lab_image - lab_background
            dist = torch.linalg.vector_norm(diff, dim=0)
        else:
            if len(rgb.size()) == 2:
                diff = rgb - background
                dist = torch.linalg.vector_norm(diff, dim=1)
            else:
                diff = rgb - background.reshape(3, 1, 1)
                dist = torch.linalg.vector_norm(diff, dim=0)

        # Mask pixels similar to background, push their alpha toward zero
        mask = dist < self.threshold
        if len(alpha.size()) == 1:
            masked_alpha = alpha[mask]
        else:
            masked_alpha = alpha[:, mask]
        if torch.sum(mask) == 0:
            return torch.tensor(0.0, device=rgb.device)
        return self.l1(masked_alpha, torch.zeros_like(masked_alpha))


def mixture_of_laplacians_loss(x):
    """Bimodal opacity prior — push values toward 0 or 1.

    NOT from paper — extra loss from reference code.
    Mixture of two Laplacians centered at 0 and 1:
        L = -mean(log(Lap(x; 0, b) + Lap(x; 1, b)))
    where b = 0.1 controls peak sharpness.

    Config: use_opacity_prior (default False), opacity_prior_lambda (default 0.0001)

    Args:
        x: Sigmoid-activated opacities in (0, 1). Shape: [N] or [N, 1].
    """
    LAPLACIAN_SCALE = 0.1  # smaller = sharper peaks at 0 and 1
    lp1 = torch.exp(-torch.abs(x) / LAPLACIAN_SCALE)
    lp2 = torch.exp(-torch.abs(1 - x) / LAPLACIAN_SCALE)
    return -torch.mean(torch.log(lp1 + lp2))
