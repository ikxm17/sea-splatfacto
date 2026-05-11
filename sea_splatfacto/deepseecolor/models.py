import math
from typing import Optional

import torch
import torch.nn as nn

class BackscatterNetV2(nn.Module):
    """Backscatter model — Eq (3) backscatter term.

    Models depth-dependent water backscatter:
        B(z) = sigmoid(B_inf) * (1 - exp(-beta_B * z))

    Where:
    - B_inf [3,1,1]: water color at infinite depth (sigmoid-constrained to (0,1))
    - beta_B: per-channel backscatter coefficients (encoded as conv2d weights)
    - conv2d(depth, params) computes beta_B * z (1x1 conv = per-channel scalar multiply)

    Optional residual term (use_residual=True):
        B(z) += sigmoid(J_prime) * exp(-beta_D' * z)

    Args:
        use_residual: Add secondary exponential decay term. Default: False.
        scale: Multiplier for conv params when do_sigmoid=True. Default: 1.0.
        do_sigmoid: Apply sigmoid to conv params before multiplication. Default: False.
        init_vals: Use reference initialization [0.95, 0.8, 0.8]. Default: False.
    """

    def __init__(
        self,
        use_residual: bool = False,
        scale: float = 1.0,
        do_sigmoid: bool = False,
        init_vals: bool = False,
    ):
        super().__init__()
        self.scale = scale
        self.do_sigmoid = do_sigmoid
        self.use_residual = use_residual

        self.relu = nn.ReLU()

        if init_vals:
            self.backscatter_conv_params = nn.Parameter(
                torch.Tensor([0.95, 0.8, 0.8]).reshape(3, 1, 1, 1)
            )
        else:
            self.backscatter_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))

        self.B_inf = nn.Parameter(torch.rand(3, 1, 1))
        self.l2 = nn.MSELoss()

        if use_residual:
            self.residual_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
            self.J_prime = nn.Parameter(torch.rand(3, 1, 1))

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """Compute backscatter B(z) from depth map.

        Args:
            depth: [1, 1, H, W] normalized depth map.
        Returns:
            [1, 3, H, W] per-channel backscatter contribution.
        """
        if self.do_sigmoid:
            beta_b_conv = self.relu(
                nn.functional.conv2d(
                    depth, self.scale * torch.sigmoid(self.backscatter_conv_params)
                )
            )
        else:
            beta_b_conv = torch.clamp(
                nn.functional.conv2d(depth, self.backscatter_conv_params), 0.0
            )

        # B(z) = sigmoid(B_inf) * (1 - exp(-beta_B * z))
        # beta_b_conv = conv2d(depth, params) already equals β·z; do NOT multiply by depth again
        backscatter = torch.sigmoid(self.B_inf) * (1 - torch.exp(-beta_b_conv))

        # Residual term: adds secondary depth-dependent contribution
        if self.use_residual:
            if self.do_sigmoid:
                beta_d_conv = self.relu(
                    nn.functional.conv2d(
                        depth, self.scale * torch.sigmoid(self.residual_conv_params)
                    )
                )
            else:
                beta_d_conv = torch.clamp(
                    nn.functional.conv2d(depth, self.residual_conv_params), 0.0
                )

            backscatter += torch.sigmoid(self.J_prime) * torch.exp(-beta_d_conv)

        return backscatter

    def compute_binf_loss(self, clean_bchw):
        """B_inf loss: push B_inf toward estimated atmospheric light.

        NOT from paper — extra loss from reference code.
        Estimates atmospheric light from the clean render using the dark channel
        prior, then penalizes MSE distance to B_inf.

        Config: use_binf_loss (default False), binf_loss_lambda (default 1.0)

        Args:
            clean_bchw: [B, C, H, W] clean rendered image (detached at call site).
        """
        from sea_splatfacto.utils.uw_utils import estimate_atmospheric_light

        atmospheric_colors = list()
        for rgb_image in clean_bchw:
            atmospheric_colors.append(estimate_atmospheric_light(rgb_image.detach()))

        atmospheric_color = torch.mean(torch.stack(atmospheric_colors), dim=0)

        return self.l2(atmospheric_color.squeeze(), self.B_inf.squeeze())
        

class AttenuateNetV3(nn.Module):
    """Attenuation model — Eq (3) transmission term.

    Models depth-dependent light attenuation (transmission):
        T(z) = exp(-beta_D * z)

    Where beta_D are per-channel attenuation coefficients. Red light attenuates
    fastest underwater, so beta_D_R > beta_D_G > beta_D_B at convergence.

    conv2d(depth, params) computes beta_D * z (1x1 conv = per-channel scalar multiply).
    The direct signal is then: D = J * T(z) = clean_rgb * exp(-beta_D * z).

    Args:
        scale: Multiplier for conv params when do_sigmoid=True.
        do_sigmoid: Apply sigmoid to conv params before multiplication. Default: False.
        init_vals: Use reference initialization [1.1, 0.95, 0.95]. Default: False.
    """

    def __init__(self, scale, do_sigmoid: bool = False, init_vals: bool = False, max_amplification: Optional[float] = None):
        super().__init__()
        self.scale = scale
        self.do_sigmoid = do_sigmoid
        # [idea-011-A] Hard clamp on attenuation to limit clean/medium divergence
        # max_amplification=3.0 means 1/T(z) ≤ 3, so beta_d*z ≤ log(3)
        self.max_attenuation_log = math.log(max_amplification) if max_amplification is not None else None

        # beta_d: attenuation coefficients (per-channel, rgb)
        if init_vals:
            self.attenuation_conv_params = nn.Parameter(
                torch.Tensor([1.1, 0.95, 0.95]).reshape(3, 1, 1, 1)
            )
        else:
            self.attenuation_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """Compute transmission T(z) from depth map.

        Args:
            depth: [1, 1, H, W] normalized depth map.
        Returns:
            [1, 3, H, W] per-channel transmission values in (0, 1].
        """
        if self.do_sigmoid:
            beta_d_conv = torch.relu(
                nn.functional.conv2d(
                    depth, self.scale * torch.sigmoid(self.attenuation_conv_params)
                )
            )
        else:
            beta_d_conv = torch.clamp(
                nn.functional.conv2d(depth, self.attenuation_conv_params), 0.0
            )

        # attenuation model: exp(-beta_d * depth)
        # beta_d_conv = conv2d(depth, params) already equals β·z; do NOT multiply by depth again
        # [idea-011-A] Clamp beta_d*z to limit amplification factor 1/T(z)
        if self.max_attenuation_log is not None:
            beta_d_conv = torch.clamp(beta_d_conv, max=self.max_attenuation_log)
        attenuation = torch.exp(-beta_d_conv)

        return attenuation


class AttenuateNetV4(nn.Module):
    """Depth-dependent attenuation model — idea 012.

    Restores the original SeaSplat/DeepSeeColor double-exponential
    parameterization where the attenuation *rate* varies with depth:

        a_c(z) = w_c * exp(-v_c * z) + y_c * exp(-x_c * z)
        T_c(z) = exp(-a_c(z) * z)

    With 4 learned parameters per channel (12 total), beta_D is a
    function of depth rather than a scalar.  A constant per-Gaussian
    color cannot trivially cancel a depth-varying attenuation rate,
    breaking the spatially-uniform bypass that defeated ideas 010/011.

    The implementation uses a Conv2d(1→6,1) whose weights are the
    exponential decay rates {v,x} and a separate coefficient tensor
    {w,y}.  Pairs of features are combined per channel (R,G,B).

    Args:
        scale: Multiplier for conv params when do_sigmoid=True.
        do_sigmoid: Apply sigmoid to conv params. Default: False.
        max_amplification: If set, clamp beta_d*z so 1/T(z) ≤ this value.
    """

    def __init__(
        self,
        scale: float = 1.0,
        do_sigmoid: bool = False,
        max_amplification: Optional[float] = None,
    ):
        super().__init__()
        self.scale = scale
        self.do_sigmoid = do_sigmoid
        self.max_attenuation_log = (
            math.log(max_amplification) if max_amplification is not None else None
        )

        # 6 exponential decay rates: {v_r, x_r, v_g, x_g, v_b, x_b}
        # Conv2d(1,6,1,bias=False) maps depth → 6 feature maps
        self.attenuation_conv_params = nn.Parameter(torch.rand(6, 1, 1, 1))
        # 6 coefficients: {w_r, y_r, w_g, y_g, w_b, y_b}
        self.attenuation_coef = nn.Parameter(torch.rand(6, 1, 1))

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """Compute depth-dependent transmission T(z).

        Args:
            depth: [1, 1, H, W] normalized depth map.
        Returns:
            [1, 3, H, W] per-channel transmission values in (0, 1].
        """
        # Compute exp(-v_i * z) for each of 6 decay terms
        # conv2d with shape [6,1,1,1] on [1,1,H,W] → [1,6,H,W]
        if self.do_sigmoid:
            attn_conv = torch.exp(
                -torch.clamp(
                    nn.functional.conv2d(
                        depth,
                        self.scale * torch.sigmoid(self.attenuation_conv_params),
                    ),
                    0.0,
                )
            )
        else:
            attn_conv = torch.exp(
                -torch.clamp(
                    nn.functional.conv2d(depth, self.attenuation_conv_params),
                    0.0,
                )
            )

        # Combine pairs into per-channel beta_d:
        # beta_d_c(z) = w_c * exp(-v_c * z) + y_c * exp(-x_c * z)
        if self.do_sigmoid:
            coef = torch.sigmoid(self.attenuation_coef)
        else:
            coef = torch.clamp(self.attenuation_coef, 0.0)

        beta_d = torch.stack(
            [
                torch.sum(attn_conv[:, i : i + 2, :, :] * coef[i : i + 2], dim=1)
                for i in range(0, 6, 2)
            ],
            dim=1,
        )  # [1, 3, H, W]

        # T(z) = exp(-beta_d(z) * z)
        beta_d_z = torch.clamp(beta_d, 0.0) * depth

        if self.max_attenuation_log is not None:
            beta_d_z = torch.clamp(beta_d_z, max=self.max_attenuation_log)

        attenuation = torch.exp(-beta_d_z)
        return attenuation