from math import floor
import torch
import torch.nn as nn
import numpy as np

from torch.nn import init


def blend_func(opacity, acc_transmission):
    return opacity * acc_transmission


def render_func(ray_feature):
    return ray_feature[..., 1:4]


def tone_map(color, gamma=2.2, exposure=1):
    return color


def ray_march(ray_dist, ray_valid, ray_features, render_func, blend_func, bg_color=None):
    # ray_dist: N x Rays x Samples
    # ray_valid: N x Rays x Samples
    # ray_features: N x Rays x Samples x Features
    # Output
    # ray_color: N x Rays x 3
    # point_color: N x Rays x Samples x 3
    # opacity: N x Rays x Samples
    # acc_transmission: N x Rays x Samples
    # blend_weight: N x Rays x Samples x 1
    # background_transmission: N x Rays x 1
    point_color = render_func(ray_features)

    # we are essentially predicting predict 1 - e^-sigma
    sigma = ray_features[..., 0] * ray_valid.float()
    opacity = 1 - torch.exp(-sigma * ray_dist)

    # cumprod exclusive
    acc_transmission = torch.cumprod(1.0 - opacity + 1e-10, dim=-1)
    temp = torch.ones(opacity.shape[0:2] + (1,)).to(opacity.device).float()  # N x R x 1

    # background_transmission = acc_transmission[:, :, [-1]]
    acc_transmission = torch.cat([temp, acc_transmission[:, :, :-1]], dim=-1)
    blend_weight = blend_func(opacity, acc_transmission)[..., None]

    ray_color = torch.sum(point_color * blend_weight, dim=-2, keepdim=False)
    # if bg_color is not None:
    #     ray_color += bg_color.to(opacity.device).float().view(
    #         background_transmission.shape[0], 1, 3) * background_transmission
    # #
    # if point_color.shape[1] > 0 and (torch.any(torch.isinf(point_color)) or torch.any(torch.isnan(point_color))):
    #     print("ray_color", torch.min(ray_color),torch.max(ray_color))

    # print("background_transmission", torch.min(background_transmission), torch.max(background_transmission))
    # background_blend_weight = blend_func(1, background_transmission)
    # print("ray_color", torch.max(torch.abs(ray_color)), torch.max(torch.abs(sigma)), torch.max(torch.abs(opacity)),torch.max(torch.abs(acc_transmission)), torch.max(torch.abs(background_transmission)), torch.max(torch.abs(acc_transmission)), torch.max(torch.abs(background_blend_weight)))
    return (
        ray_color,
        point_color,
        opacity,
        acc_transmission,
        blend_weight,
    )  # background_transmission, background_blend_weight


def alpha_ray_march(ray_dist, ray_valid, ray_features, blend_func):
    sigma = ray_features[..., 0] * ray_valid.float()
    opacity = 1 - torch.exp(-sigma * ray_dist)

    acc_transmission = torch.cumprod(1.0 - opacity + 1e-10, dim=-1)
    temp = torch.ones(opacity.shape[0:2] + (1,)).to(opacity.device).float()  # N x R x 1
    background_transmission = acc_transmission[:, :, [-1]]
    acc_transmission = torch.cat([temp, acc_transmission[:, :, :-1]], dim=-1)

    blend_weight = blend_func(opacity, acc_transmission)[..., None]
    background_blend_weight = blend_func(1, background_transmission)

    return opacity, acc_transmission, blend_weight, background_transmission, background_blend_weight


def get_xavier_multiplier(m, gain):
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // m.stride[0] // m.stride[1] // m.stride[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * np.sqrt(2.0 / (n1 + n2))
    else:
        return None

    return std


def xavier_uniform_(m, gain):
    std = get_xavier_multiplier(m, gain)
    m.weight.data.uniform_(-std * np.sqrt(3.0), std * np.sqrt(3.0))


def init_weights(net, init_type="xavier_uniform", gain=1):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "xavier_uniform":
                xavier_uniform_(m, gain)
            elif init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError("initialization method [{}] is not implemented".format(init_type))
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_seq(s, init_type="xavier_uniform"):
    """initialize sequential model"""
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            init_weights(a, init_type, nn.init.calculate_gain("relu"))
        elif isinstance(b, nn.LeakyReLU):
            init_weights(a, init_type, nn.init.calculate_gain("leaky_relu", b.negative_slope))
        else:
            init_weights(a, init_type)
    init_weights(s[-1])


def positional_encoding(positions, freqs, ori=False):
    """encode positions with positional encoding
        positions: :math:`(...,D)`
        freqs: int
    Return:
        pts: :math:`(..., 2DF)`
    """
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (F,)
    ori_c = positions.shape[-1]
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],)
    )  # (..., DF)

    if ori:
        pts = torch.cat([positions, torch.sin(pts), torch.cos(pts)], dim=-1).reshape(
            pts.shape[:-1] + (pts.shape[-1] * 2 + ori_c,)
        )
    else:
        pts = torch.stack([torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1] + (pts.shape[-1] * 2,))
    return pts


def around_surface_ray_generation(campos, raydir, point_count, center, sample_radius, jitter=0.0):
    """sample points around the surface

    Args:
        campos (_type_): _description_
        raydir (_type_): _description_
        point_count (_type_): _description_
        center (float): 采样的中心深度，通过 mesh rasterization 获得
        sample_radius (float): 采样半径，即在采样中心前后采样的范围
        jitter (_type_, optional): _description_. Defaults to 0..

    Returns:
        _type_: _description_
    """
    near = center - sample_radius  # near depth
    far = center + sample_radius

    return near_far_linear_ray_generation(campos, raydir, point_count, near, far, jitter)


def gen_middle_points(raydir, point_count, near, far, jitter=0.0):
    assert type(near) == type(far)
    if isinstance(near, torch.Tensor):
        near, far = near[..., None], far[..., None]

    tvals = torch.linspace(0, 1, point_count + 1, device=raydir.device).view(1, 1, -1)
    tvals = near * (1 - tvals) + far * tvals  # N x Rays x Sammples
    segment_length = (tvals[..., 1:] - tvals[..., :-1]) * (
        1 + jitter * (torch.rand((raydir.shape[0], raydir.shape[1], point_count), device=raydir.device) - 0.5)
    )

    end_point_ts = torch.cumsum(segment_length, dim=2)
    end_point_ts = torch.cat(
        [torch.zeros((end_point_ts.shape[0], end_point_ts.shape[1], 1), device=end_point_ts.device), end_point_ts],
        dim=2,
    )
    end_point_ts = near + end_point_ts
    middle_points = (end_point_ts[:, :, :-1] + end_point_ts[:, :, 1:]) / 2

    return middle_points


def dcenter_ray_generation(campos, raydir, point_count, center, sample_radius, dthres, jitter=0.0):
    mask = (center[..., 0] - center[..., 1]) > dthres  # greater than dthres, using two depths
    raypos_placeholder = torch.zeros((raydir.shape[0], raydir.shape[1], point_count, 3)).cuda()

    # sampling position for those single depth rays
    raypos_placeholder[~mask] = around_surface_ray_generation(
        campos, raydir[~mask][None], point_count, center[~mask][None].mean(dim=-1), sample_radius, jitter
    )

    # sampling position for double depth rays
    if mask.any():
        draydir = raydir[mask][None]

        for_num = point_count // 2
        dcenter_for = center[mask][None, :, 1]
        ddepth_raypos_for = gen_middle_points(
            draydir, for_num, dcenter_for - sample_radius, dcenter_for + sample_radius, jitter
        )

        back_num = point_count - for_num
        dcenter_back = center[mask][None, :, 0]
        ddepth_raypos_back = gen_middle_points(
            draydir, back_num, dcenter_back - sample_radius, dcenter_back + sample_radius, jitter
        )

        middle_point_ts = torch.cat([ddepth_raypos_for, ddepth_raypos_back], dim=-1).sort(dim=-1)[0]
        raypos_placeholder[mask] = campos[:, None, None, :] + draydir[:, :, None, :] * middle_point_ts[:, :, :, None]

    # cat two raypos
    return raypos_placeholder


def mix_ray_generation(campos, raydir, point_count, linear_ratio, near, far, center, sample_radius, jitter=0.0):
    # linear
    linear_num = floor(point_count * linear_ratio)
    tvals = torch.linspace(0, 1, linear_num + 1, device=campos.device).view(1, 1, -1)
    tvals = near * (1 - tvals) + far * tvals  # N x Rays x Sammples
    segment_length = (tvals[..., 1:] - tvals[..., :-1]) * (
        1 + jitter * (torch.rand((raydir.shape[0], raydir.shape[1], linear_num), device=campos.device) - 0.5)
    )

    end_point_ts = torch.cumsum(segment_length, dim=2)
    end_point_ts = torch.cat(
        [torch.zeros((end_point_ts.shape[0], end_point_ts.shape[1], 1), device=end_point_ts.device), end_point_ts],
        dim=2,
    )
    end_point_ts = near + end_point_ts
    middle_point_linear = (end_point_ts[:, :, :-1] + end_point_ts[:, :, 1:]) / 2

    # surface
    surface_num = point_count - linear_num
    near = (center - sample_radius)[..., None]
    far = (center + sample_radius)[..., None]
    tvals = torch.linspace(0, 1, surface_num + 1, device=campos.device).view(1, 1, -1)
    tvals = near * (1 - tvals) + far * tvals  # N x Rays x Sammples
    segment_length = (tvals[..., 1:] - tvals[..., :-1]) * (
        1 + jitter * (torch.rand((raydir.shape[0], raydir.shape[1], surface_num), device=campos.device) - 0.5)
    )

    end_point_ts = torch.cumsum(segment_length, dim=2)
    end_point_ts = torch.cat(
        [torch.zeros((end_point_ts.shape[0], end_point_ts.shape[1], 1), device=end_point_ts.device), end_point_ts],
        dim=2,
    )
    end_point_ts = near + end_point_ts
    middle_point_surface = (end_point_ts[:, :, :-1] + end_point_ts[:, :, 1:]) / 2

    middle_point_ts = torch.cat([middle_point_linear, middle_point_surface], dim=-1).sort(dim=-1)[0]
    raypos = campos[:, None, None, :] + raydir[:, :, None, :] * middle_point_ts[:, :, :, None]

    return raypos


def near_far_linear_ray_generation(campos, raydir, point_count, near=0.1, far=10, jitter=0.0, **kargs):
    # inputs
    # campos: N x 3
    # raydir: N x Rays x 3, must be normalized
    # near: N x 1 x 1
    # far:  N x 1 x 1
    # jitter: float in [0, 1), a fraction of step length
    # outputs
    # raypos: N x Rays x Samples x 3
    # segment_length: N x Rays x Samples
    # valid: N x Rays x Samples
    # ts: N x Rays x Samples
    # print("campos", campos.shape)
    # print("raydir", raydir.shape)
    assert type(near) == type(far)
    if isinstance(near, torch.Tensor):
        near, far = near[..., None], far[..., None]

    tvals = torch.linspace(0, 1, point_count + 1, device=campos.device).view(1, 1, -1)
    tvals = near * (1 - tvals) + far * tvals  # N x Rays x Sammples
    segment_length = (tvals[..., 1:] - tvals[..., :-1]) * (
        1 + jitter * (torch.rand((raydir.shape[0], raydir.shape[1], point_count), device=campos.device) - 0.5)
    )

    end_point_ts = torch.cumsum(segment_length, dim=2)
    end_point_ts = torch.cat(
        [torch.zeros((end_point_ts.shape[0], end_point_ts.shape[1], 1), device=end_point_ts.device), end_point_ts],
        dim=2,
    )
    end_point_ts = near + end_point_ts

    middle_point_ts = (end_point_ts[:, :, :-1] + end_point_ts[:, :, 1:]) / 2
    raypos = campos[:, None, None, :] + raydir[:, :, None, :] * middle_point_ts[:, :, :, None]
    # valid = torch.ones_like(middle_point_ts,
    #                         dtype=middle_point_ts.dtype,
    #                         device=middle_point_ts.device)

    # segment_length*=torch.linalg.norm(raydir[..., None, :], axis=-1)
    return raypos


def near_far_disparity_linear_ray_generation(campos, raydir, point_count, near=0.1, far=10, jitter=0.0, **kargs):
    # inputs
    # campos: N x 3
    # raydir: N x Rays x 3, must be normalized
    # near: N x 1 x 1
    # far:  N x 1 x 1
    # jitter: float in [0, 1), a fraction of step length
    # outputs
    # raypos: N x Rays x Samples x 3
    # segment_length: N x Rays x Samples
    # valid: N x Rays x Samples
    # ts: N x Rays x Samples

    tvals = torch.linspace(0, 1, point_count + 1, device=campos.device).view(1, -1)
    tvals = 1.0 / (1.0 / near * (1 - tvals) + 1.0 / far * tvals)  # N x 1 x Sammples
    segment_length = (tvals[..., 1:] - tvals[..., :-1]) * (
        1 + jitter * (torch.rand((raydir.shape[0], raydir.shape[1], point_count), device=campos.device) - 0.5)
    )

    end_point_ts = torch.cumsum(segment_length, dim=2)
    end_point_ts = torch.cat(
        [torch.zeros((end_point_ts.shape[0], end_point_ts.shape[1], 1), device=end_point_ts.device), end_point_ts],
        dim=2,
    )
    end_point_ts = near + end_point_ts

    middle_point_ts = (end_point_ts[:, :, :-1] + end_point_ts[:, :, 1:]) / 2
    raypos = campos[:, None, None, :] + raydir[:, :, None, :] * middle_point_ts[:, :, :, None]
    # print(tvals.shape, segment_length.shape, end_point_ts.shape, middle_point_ts.shape, raypos.shape)
    valid = torch.ones_like(middle_point_ts, dtype=middle_point_ts.dtype, device=middle_point_ts.device)
    # print("campos", campos.shape, campos[0])
    # print("raydir", raydir.shape, raydir[0,0])
    # print("middle_point_ts", middle_point_ts.shape, middle_point_ts[0,0])
    # print("raypos", raypos.shape, raypos[0,0])

    return raypos, segment_length, valid, middle_point_ts
