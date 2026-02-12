import torch
import math

def dark_channel_estimate(rgb):
    patch_size = 41
    padding =  patch_size // 2
    rgb_max = torch.nn.MaxPool3d(
        kernel_size=(3, patch_size, patch_size),
        stride=1,
        padding=(0, padding, padding)
    )
    if len(rgb.size()) == 3:
        dcp = torch.abs(rgb_max(-rgb.unsqueeze(0)))
    else:
        dcp = torch.abs(rgb_max(-rgb))
    return dcp.squeeze()

def estimate_atmospheric_light(rgb):
    dcp = dark_channel_estimate(rgb)

    flat_dcp = torch.flatten(dcp)
    flat_r = torch.flatten(rgb[0])
    flat_g = torch.flatten(rgb[1])
    flat_b = torch.flatten(rgb[2])

    k = math.ceil(0.001 * len(flat_dcp))
    vals, idxs = torch.topk(flat_dcp, k)

    median_val = torch.median(vals)
    median_idxs = torch.where(vals == median_val)
    color_idxs = idxs[median_idxs]

    atmospheric_light = torch.stack([torch.mean(flat_r[color_idxs]), torch.mean(flat_g[color_idxs]), torch.mean(flat_b[color_idxs])])

    return atmospheric_light