import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean


# N: Batch size
# L_s: number of controllable drones
# L: max number of visible objects
# C: number of channels/features on each object
def relative_positions(
    origin,  # (N, L_s, 2)
    direction,  # (N, L_s, 2)
    positions,  # (N, L, 2)
) -> torch.Tensor:  # (N, L_s, L, 2)
    n, ls, _ = origin.size()
    _, l, _ = positions.size()

    origin = origin.view(n, ls, 1, 2)
    direction = direction.view(n, ls, 1, 2)
    positions = positions.view(n, 1, l, 2)

    positions = positions - origin

    angle = -torch.atan2(direction[:, :, :, 1], direction[:, :, :, 0])
    rotation = torch.cat(
        [
            torch.cat(
                [angle.cos().view(n, ls, 1, 1, 1), angle.sin().view(n, ls, 1, 1, 1)],
                dim=3,
            ),
            torch.cat(
                [-angle.sin().view(n, ls, 1, 1, 1), angle.cos().view(n, ls, 1, 1, 1)],
                dim=3,
            ),
        ],
        dim=4,
    )

    positions_rotated = torch.matmul(rotation, positions.view(n, ls, l, 2, 1)).view(
        n, ls, l, 2
    )

    return positions_rotated


def polar_indices(
    positions, nray, nring, inner_radius  # (N, L_s, L, 2)
):  # (N, L_s, L), (N, L_s, L), (N, L_s, L), (N, L_s, L)
    distances = torch.sqrt(positions[:, :, :, 0] ** 2 + positions[:, :, :, 1] ** 2)
    distance_indices = (
        torch.clamp(distances / inner_radius, min=0, max=nring - 1).floor().long()
    )
    angles = torch.atan2(positions[:, :, :, 1], positions[:, :, :, 0]) + math.pi
    # There is one angle value that can result in index of exactly nray, clamp it to nray-1
    angular_indices = torch.clamp_max(
        (angles / (2 * math.pi) * nray).floor().long(), nray - 1
    )

    distance_offsets = torch.clamp_max(
        distances / inner_radius - distance_indices.float() - 0.5, max=2
    )
    angular_offsets = angles / (2 * math.pi) * nray - angular_indices.float() - 0.5

    assert (
        angular_indices.min() >= 0
    ), f"Negative angular index: {angular_indices.min()}"
    assert (
        angular_indices.max() < nray
    ), f"invalid angular index: {angular_indices.max()} >= {nray}"
    assert (
        distance_indices.min() >= 0
    ), f"Negative distance index: {distance_indices.min()}"
    assert (
        distance_indices.max() < nring
    ), f"invalid distance index: {distance_indices.max()} >= {nring}"

    return distance_indices, angular_indices, distance_offsets, angular_offsets


# N: Batch size
# L: max number of visible objects
# C: number of channels/features on each object
def unbatched_relative_positions(
    origin,  # (N, 2)
    direction,  # (N, 2)
    positions,  # (N, L, 2)
):  # (N, L, 2)
    n, _ = origin.size()
    _, l, _ = positions.size()

    origin = origin.view(n, 1, 2)
    direction = direction.view(n, 1, 2)
    positions = positions.view(n, l, 2)

    positions = positions - origin

    angle = -torch.atan2(direction[:, :, 1], direction[:, :, 0])
    rotation = torch.cat(
        [
            torch.cat(
                [angle.cos().view(n, 1, 1, 1), angle.sin().view(n, 1, 1, 1)],
                dim=2,
            ),
            torch.cat(
                [-angle.sin().view(n, 1, 1, 1), angle.cos().view(n, 1, 1, 1)],
                dim=2,
            ),
        ],
        dim=3,
    )

    positions_rotated = torch.matmul(rotation, positions.view(n, l, 2, 1)).view(n, l, 2)

    return positions_rotated


def varlength_polar_indices(
    positions, indices, nray, nring, inner_radius  # (N, L_s, L, 2)
):  # (N, L_s, L), (N, L_s, L), (N, L_s, L), (N, L_s, L)
    distances = torch.sqrt(positions[:, :, :, 0] ** 2 + positions[:, :, :, 1] ** 2)
    distance_indices = (
        torch.clamp(distances / inner_radius, min=0, max=nring - 1).floor().long()
    )
    angles = torch.atan2(positions[:, :, :, 1], positions[:, :, :, 0]) + math.pi
    # There is one angle value that can result in index of exactly nray, clamp it to nray-1
    angular_indices = torch.clamp_max(
        (angles / (2 * math.pi) * nray).floor().long(), nray - 1
    )

    distance_offsets = torch.clamp_max(
        distances / inner_radius - distance_indices.float() - 0.5, max=2
    )
    angular_offsets = angles / (2 * math.pi) * nray - angular_indices.float() - 0.5

    assert (
        angular_indices.min() >= 0
    ), f"Negative angular index: {angular_indices.min()}"
    assert (
        angular_indices.max() < nray
    ), f"invalid angular index: {angular_indices.max()} >= {nray}"
    assert (
        distance_indices.min() >= 0
    ), f"Negative distance index: {distance_indices.min()}"
    assert (
        distance_indices.max() < nring
    ), f"invalid distance index: {distance_indices.max()} >= {nring}"

    return distance_indices, angular_indices, distance_offsets, angular_offsets


def spatial_scatter(
    items,  # (N, L_s, L, C)
    positions,  # (N, L_s, L, 2)
    nray,
    nring,
    inner_radius,
    embed_offsets=False,
):  # (N, L_s, C', nring, nray) where C' = C + 2 if embed_offsets else C
    n, ls, l, c = items.size()
    assert (
        n,
        ls,
        l,
        2,
    ) == positions.size(), (
        f"Expect size {(n, ls, l, 2)} for positions, actual: {positions.size()}"
    )

    distance_index, angular_index, distance_offsets, angular_offsets = polar_indices(
        positions, nray, nring, inner_radius
    )
    index = distance_index * nray + angular_index
    index = index.unsqueeze(-1)
    scattered_items = (
        scatter_add(items, index, dim=2, dim_size=nray * nring)
        .permute(0, 1, 3, 2)
        .reshape(n, ls, c, nring, nray)
    )

    if embed_offsets:
        offsets = torch.cat(
            [distance_offsets.unsqueeze(-1), angular_offsets.unsqueeze(-1)], dim=3
        )
        scattered_nonshared = (
            scatter_mean(offsets, index, dim=2, dim_size=nray * nring)
            .permute(0, 1, 3, 2)
            .reshape(n, ls, 2, nring, nray)
        )
        return torch.cat([scattered_nonshared, scattered_items], dim=2)
    else:
        return scattered_items


def single_batch_dim_spatial_scatter(
    items,  # (N, L, C)
    positions,  # (N, L, 2)
    nray,
    nring,
    inner_radius,
    embed_offsets=False,
):  # (N, C', nring, nray) where C' = C + 2 if embed_offsets else C
    n, l, c = items.size()
    assert (
        n,
        l,
        2,
    ) == positions.size(), (
        f"Expect size {(n, l, 2)} for positions, actual: {positions.size()}"
    )

    (
        distance_index,
        angular_index,
        distance_offsets,
        angular_offsets,
    ) = single_batch_dim_polar_indices(positions, nray, nring, inner_radius)
    index = distance_index * nray + angular_index
    index = index.unsqueeze(-1)
    scattered_items = (
        scatter_add(items, index, dim=1, dim_size=nray * nring)
        .permute(0, 2, 1)
        .reshape(n, c, nring, nray)
    )

    if embed_offsets:
        offsets = torch.cat(
            [distance_offsets.unsqueeze(-1), angular_offsets.unsqueeze(-1)], dim=2
        )
        scattered_nonshared = (
            scatter_mean(offsets, index, dim=1, dim_size=nray * nring)
            .permute(0, 2, 1)
            .reshape(n, 2, nring, nray)
        )
        return torch.cat([scattered_nonshared, scattered_items], dim=1)
    else:
        return scattered_items


def single_batch_dim_polar_indices(
    positions, nray, nring, inner_radius  # (N, L, 2)
):  # (N, L), (N, L), (N, L), (N, L)
    distances = torch.sqrt(positions[:, :, 0] ** 2 + positions[:, :, 1] ** 2)
    distance_indices = (
        torch.clamp(distances / inner_radius, min=0, max=nring - 1).floor().long()
    )
    angles = torch.atan2(positions[:, :, 1], positions[:, :, 0]) + math.pi
    # There is one angle value that can result in index of exactly nray, clamp it to nray-1
    angular_indices = torch.clamp_max(
        (angles / (2 * math.pi) * nray).floor().long(), nray - 1
    )

    distance_offsets = torch.clamp_max(
        distances / inner_radius - distance_indices.float() - 0.5, max=2
    )
    angular_offsets = angles / (2 * math.pi) * nray - angular_indices.float() - 0.5

    assert (
        angular_indices.min() >= 0
    ), f"Negative angular index: {angular_indices.min()}"
    assert (
        angular_indices.max() < nray
    ), f"invalid angular index: {angular_indices.max()} >= {nray}"
    assert (
        distance_indices.min() >= 0
    ), f"Negative distance index: {distance_indices.min()}"
    assert (
        distance_indices.max() < nring
    ), f"invalid distance index: {distance_indices.max()} >= {nring}"

    return distance_indices, angular_indices, distance_offsets, angular_offsets


class ZeroPaddedCylindricalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ZeroPaddedCylindricalConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.padding = kernel_size // 2

    # input should be of dims (N, C, H, W)
    # applies dimension-preserving conv2d by zero-padding H dimension and circularly padding W dimension
    def forward(self, input):
        input = F.pad(input, [0, 0, self.padding, self.padding], mode="circular")
        input = F.pad(input, [self.padding, self.padding, 0, 0], mode="constant")
        return self.conv(input)
