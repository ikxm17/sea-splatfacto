import torch
import torch.nn as nn

class BackscatterNetV2(nn.Module):
    """_summary_


    Args:
        use_residual (bool, optional): _description_. Defaults to False.
        scale (float, optional): _description_. Defaults to 1.0.
        do_sigmoid (bool, optional): _description_. Defaults to False.
        init_vals (bool, optional): _description_. Defaults to False.
    """

    def __init__(
        self,
        use_residual: bool = False,
        scale: float = 1.0,
        do_sigmoid: bool = False,
        init_vals: bool = False,
    ):
        """_summary_

        Args:
            use_residual (bool, optional): _description_. Defaults to False.
            scale (float, optional): _description_. Defaults to 1.0.
            do_sigmoid (bool, optional): _description_. Defaults to False.
            init_vals (bool, optional): _description_. Defaults to False.
        """
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
        """_summary_

        Args:
            depth (_type_): _description_

        Returns:
            _type_: _description_
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

        # backscatter model: B(depth) = B_inf * (1 - exp(-beta_b * depth))
        # beta_b_conv = conv2d(depth, params) already equals β·z; do NOT multiply by depth again
        backscatter = torch.sigmoid(self.B_inf) * (1 - torch.exp(-beta_b_conv))

        # ? What is this part doing?
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

    def forward_rgb(self, rgb):
        from sea_splatfacto.utils.uw_utils import estimate_atmospheric_light
        
        atmospheric_colors = list()
        for rgb_image in rgb:
            atmospheric_colors.append(estimate_atmospheric_light(rgb_image.detach()))
        
        atmospheric_color = torch.mean(torch.stack(atmospheric_colors), dim=0)
        
        return self.l2(atmospheric_color.squeeze(), self.B_inf.squeeze())
        

class AttenuateNetV3(nn.Module):
    """_summary_


    Args:
        nn (_type_): _description_
    """

    def __init__(self, scale, do_sigmoid: bool = False, init_vals: bool = False):
        """_summary_

        Args:
            scale (_type_): _description_
            do_sigmoid (bool, optional): _description_. Defaults to False.
            init_vals (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.scale = scale
        self.do_sigmoid = do_sigmoid

        # beta_d: attenuation coefficients (per-channel, rgb)
        if init_vals:
            self.attenuation_conv_params = nn.Parameter(
                torch.Tensor([1.1, 0.95, 0.95]).reshape(3, 1, 1, 1)
            )
        else:
            self.attenuation_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            depth (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
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
        attenuation = torch.exp(-beta_d_conv)

        return attenuation