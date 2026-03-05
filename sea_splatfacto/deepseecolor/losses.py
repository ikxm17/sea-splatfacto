import math

import torch
import torch.nn as nn

from kornia.color import rgb_to_lab

class AttenuateLoss(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, target_intensity: float = 0.5):
        """_summary_

        Args:
            target_intensity (float, optional): _description_. Defaults to 0.5.
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.target_intensity = target_intensity

    def forward(self, direct: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            direct (torch.Tensor): _description_
            J (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        # ? saturation loss
        # * saturation loss that penalizes values of J that fall outside the range [0, 1]
        # * torch.relu(-J) acitvates when J < 0
        # * torch.relu(J - 1) activates when J > 1
        saturation_loss = (torch.relu(-J) + torch.relu(J - 1)).square().mean()

        # ? spatial variation loss
        init_spatial = torch.std(direct, dim=[2, 3])
        channel_spatial = torch.std(J, dim=[2, 3])
        spatial_variation_loss = self.mse(channel_spatial, init_spatial)

        # ? intensity loss
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
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, cost_ratio: float = 1000.0):
        """_summary_

        Args:
            cost_ratio (float, optional): _description_. Defaults to 1000.0.
        """
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.smooth_l1_loss = nn.SmoothL1Loss(beta=0.1)  # TODO: make beta a parameter
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.cost_ratio = cost_ratio

    def forward(self, direct, depth=None):
        """_summary_

        Args:
            direct (_type_): _description_
            depth (_type_, optional): _description_. Defaults to None.
        """
        pos = self.l1_loss(self.relu(direct), torch.zeros_like(direct))
        neg = self.smooth_l1_loss(self.relu(-direct), torch.zeros_like(direct))
        backscatter_loss = self.cost_ratio * neg + pos
        return backscatter_loss, torch.zeros_like(direct)


class GrayWorldPriorLoss(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, target_intensity: float = 0.5):
        """_summary_

        Args:
            target_intensity (float, optional): _description_. Defaults to 0.5.
        """
        super().__init__()
        self.target_intensity = target_intensity

    def forward(self, J: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            J (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
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
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        """_summary_"""
        super().__init__()

    def forward(self, rgb, depth):
        """_summary_

        Args:
            rgb (_type_): _description_
            depth (_type_): _description_
        """
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

        return torch.abs(depth_dx).mean() + torch.abs(depth_dy).mean()


class RGBSpatialVariationLoss(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
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
    """_summary_

    Args:
        nn (_type_): _description_
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
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, use_kornia: bool = False):
        super().__init__()
        self.use_kornia = use_kornia
        self.l1_loss = nn.L1Loss()
        if use_kornia:
            self.range = math.sqrt(
                100 * 100 + 255 * 255 + 255 * 255
            )  # ? what are these numbers
            self.threshold = 50  # ? what are these numbers
        else:
            self.range = math.sqrt(3)  # ? what are these numbers
            self.threshold = 0.2 * math.sqrt(3)  # ? what are these numbers

    def forward(self, rgb, background, alpha):
        if self.use_kornia:
            lab_image = rgb_to_lab(rgb)
            lab_background = rgb_to_lab(background.reshape(3, 1, 1))  # ? what is lab
            diff = lab_image - lab_background
            dist = torch.linalg.vector_norm(diff, dim=0)
        else:
            if len(rgb.size()) == 2:
                diff = rgb - background
                dist = torch.linalg.vector_norm(diff, dim=1)
            else:
                diff = rgb - background.reshape(3, 1, 1)
                dist = torch.linalg.vector_norm(diff, dim=0)

        # ? other approach from seasplat
        other_approach = False
        if other_approach:
            clamped_diff = torch.max(dist - self.threshold, torch.Tensor([0.0]).cuda())
            if self.use_kornia:
                mask = torch.exp(-clamped_diff / 10)  # ? what is 10
            else:
                mask = torch.exp(-clamped_diff / 0.05)  # what is 0.05
            masked_alpha = alpha * mask
            if torch.sum(mask) == 0:
                loss = torch.Tensor([0.0]).squeeze().cuda()
            else:
                loss = self.mse(masked_alpha, torch.zeros_like(masked_alpha))

        else:
            mask = dist < self.threshold
            if len(alpha.size()) == 1:
                masked_alpha = alpha[mask]
            else:
                masked_alpha = alpha[:, mask]
            if torch.sum(mask) == 0:
                loss = torch.Tensor([0.0]).squeeze().cuda()
            else:
                loss = self.l1_loss(masked_alpha, torch.zeros_like(masked_alpha))
        return loss


def mixture_of_laplacians_loss(x):
    """_summary_

    Args:
        x (_type_): _description_
    """
    lp1 = torch.exp(-torch.abs(x) / 0.1)
    lp2 = torch.exp(-torch.abs(1 - x) / 0.1)
    return -torch.mean(torch.log(lp1 + lp2))
