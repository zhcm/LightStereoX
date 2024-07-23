# @Time    : 2024/7/12 14:56
# @Author  : zhangchenming
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

from .dinov2 import _make_dinov2_model


def dinov2_vitl14():
    vit = _make_dinov2_model(
        arch_name="vit_large",
        pretrained="",
        output_idx=[5, 12, 18, 24],
        checkpoint=False,
        drop_path_rate=0.0,
        num_register_tokens=0,
        use_norm=False,
        export=False
    )
    return vit


def generate_rays(camera_intrinsics, image_shape, noisy=False):
    batch_size, device, dtype = (camera_intrinsics.shape[0], camera_intrinsics.device, camera_intrinsics.dtype)
    height, width = image_shape
    # Generate grid of pixel coordinates
    pixel_coords_x = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    pixel_coords_y = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    if noisy:
        pixel_coords_x += torch.rand_like(pixel_coords_x) - 0.5
        pixel_coords_y += torch.rand_like(pixel_coords_y) - 0.5
    pixel_coords = torch.stack(
        [pixel_coords_x.repeat(height, 1), pixel_coords_y.repeat(width, 1).t()], dim=2
    )  # (H, W, 2)
    pixel_coords = pixel_coords + 0.5

    # Calculate ray directions
    intrinsics_inv = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    intrinsics_inv[:, 0, 0] = 1.0 / camera_intrinsics[:, 0, 0]
    intrinsics_inv[:, 1, 1] = 1.0 / camera_intrinsics[:, 1, 1]
    intrinsics_inv[:, 0, 2] = -camera_intrinsics[:, 0, 2] / camera_intrinsics[:, 0, 0]
    intrinsics_inv[:, 1, 2] = -camera_intrinsics[:, 1, 2] / camera_intrinsics[:, 1, 1]
    homogeneous_coords = torch.cat(
        [pixel_coords, torch.ones_like(pixel_coords[:, :, :1])], dim=2
    )  # (H, W, 3)
    ray_directions = torch.matmul(
        intrinsics_inv, homogeneous_coords.permute(2, 0, 1).flatten(1)
    )  # (3, H*W)
    ray_directions = F.normalize(ray_directions, dim=1)  # (B, 3, H*W)
    ray_directions = ray_directions.permute(0, 2, 1)  # (B, H*W, 3)

    theta = torch.atan2(ray_directions[..., 0], ray_directions[..., -1])
    phi = torch.acos(ray_directions[..., 1])
    # pitch = torch.asin(ray_directions[..., 1])
    # roll = torch.atan2(ray_directions[..., 0], - ray_directions[..., 1])
    angles = torch.stack([theta, phi], dim=-1)
    return ray_directions, angles


def max_stack(tensors):
    if len(tensors) == 1:
        return tensors[0]
    return torch.stack(tensors, dim=-1).max(dim=-1).values


class AnyStereo(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = dinov2_vitl14()

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']
        intrinsics = data['intrinsics']

        encoder_outputs, cls_tokens = self.encoder(image1)
        encoder_outputs = [(x + y.unsqueeze(1)).contiguous() for x, y in zip(encoder_outputs, cls_tokens)]

        rays, angles = generate_rays(intrinsics, image_shape=image1.shape[2:], noisy=self.training)

        slices_encoder_range = [(0, 5), (5, 12), (12, 18), (18, 24)]
        original_encoder_outputs = [max_stack(encoder_outputs[i:j]) for i, j in slices_encoder_range]
        cls_tokens = [cls_tokens[-i - 1] for i in range(len(slices_encoder_range))]
        resolutions = [tuple(sorted([x.shape[1], x.shape[2]])) for x in original_encoder_outputs]
        level_shapes = sorted(list(set(resolutions)))[::-1]
        if len(level_shapes) == 1:
            level_shapes = level_shapes * 4
        input_shapes = [
            level_shapes[i]
            for i, (start, end) in enumerate(self.slices_encoder)
            for _ in range(end - start)
        ]
        print('fuck')

    def get_loss(self):
        pass
