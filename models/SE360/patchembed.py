import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import get_2d_sincos_pos_embed_np, get_2d_sincos_pos_embed_from_grid_np, get_1d_sincos_pos_embed_from_grid
from diffusers.utils import deprecate
from einops import rearrange
import numpy as np

class OmniGenPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        embed_dim: int = 768,
        bias: bool = True,
        interpolation_scale: float = 1,
        pos_embed_max_size: int = 192,
        base_size: int = 64,
        use_spherical_pe: bool = True,  # New parameter, enable spherical position encoding
        output_image_proj: nn.Conv2d = None,
        input_image_proj: nn.Conv2d = None,
    ):
        super().__init__()

        self.output_image_proj = output_image_proj if output_image_proj is not None else nn.Conv2d(
            5, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )
        self.input_image_proj = input_image_proj if input_image_proj is not None else nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.interpolation_scale = interpolation_scale
        self.pos_embed_max_size = pos_embed_max_size
        self.base_size = base_size
        self.use_spherical_pe = use_spherical_pe

        # Pre-compute fixed-size position encodings

        # Spherical position encoding: pre-compute 32x64 size
        pos_embed = get_3d_spherical_pos_embed(
            embed_dim,
            (32, 64),  # Predefined common panoramic image size
            base_size=base_size,
            interpolation_scale=interpolation_scale,
            output_type="pt",
        )

        # 2D position encoding: use pos_embed_max_size
        pos_embed_2d = get_2d_sincos_pos_embed(
            embed_dim,
            pos_embed_max_size,
            base_size=base_size,
            interpolation_scale=interpolation_scale,
            output_type="pt",
        )
        
        self.register_buffer("pos_embed", pos_embed.float().unsqueeze(0), persistent=True)
        self.register_buffer("pos_embed_2d", pos_embed_2d.float().unsqueeze(0), persistent=True)

    def _cropped_pos_embed(self, height, width, embed_type):
        """Generate position encoding according to specified dimensions (intelligent handling for spherical encoding, cropping for 2D encoding)"""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for position embedding.")

        height = height // self.patch_size
        width = width // self.patch_size
        
        if embed_type == "spherical":
            return self.pos_embed
        else:
            # For 2D position encoding, use original cropping method
            if height > self.pos_embed_max_size:
                raise ValueError(
                    f"Height ({height}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
                )
            if width > self.pos_embed_max_size:
                raise ValueError(
                    f"Width ({width}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
                )

            top = (self.pos_embed_max_size - height) // 2
            left = (self.pos_embed_max_size - width) // 2
            spatial_pos_embed = self.pos_embed_2d.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
            spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
            spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
            return spatial_pos_embed

    def _patch_embeddings(self, hidden_states: torch.Tensor, is_input_image: bool) -> torch.Tensor:
        if is_input_image:
            hidden_states = self.input_image_proj(hidden_states)
        else:
            hidden_states = self.output_image_proj(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return hidden_states

    def forward(
        self, hidden_states: torch.Tensor, is_input_image: bool, padding_latent: torch.Tensor = None, embed_type: str = "spherical"
    ) -> torch.Tensor:
        if isinstance(hidden_states, list):
            if padding_latent is None:
                padding_latent = [None] * len(hidden_states)
            patched_latents = []
            for sub_latent, padding in zip(hidden_states, padding_latent):
                height, width = sub_latent.shape[-2:]
                sub_latent = self._patch_embeddings(sub_latent, is_input_image)
                pos_embed = self._cropped_pos_embed(height, width, embed_type)
                sub_latent = sub_latent + pos_embed
                if padding is not None:
                    sub_latent = torch.cat([sub_latent, padding.to(sub_latent.device)], dim=-2)
                patched_latents.append(sub_latent)
        else:
            height, width = hidden_states.shape[-2:]
            pos_embed = self._cropped_pos_embed(height, width, embed_type)
            hidden_states = self._patch_embeddings(hidden_states, is_input_image)
            patched_latents = hidden_states + pos_embed
        return patched_latents

def erp_to_spherical_coords(height, width, device=None, dtype=torch.float32):
    """
    Convert pixel coordinates of ERP image to spherical coordinates
    
    Args:
        height (int): ERP image height
        width (int): ERP image width
        device: torch device
        dtype: data type
        
    Returns:
        torch.Tensor: Spherical coordinates (H, W, 3) - (x, y, z) Cartesian coordinates
    """
    # Create pixel coordinate grid
    h_coords = torch.arange(height, device=device, dtype=dtype)
    w_coords = torch.arange(width, device=device, dtype=dtype)
    grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing='ij')
    
    # Ensure π constant uses correct data type
    pi = torch.tensor(torch.pi, device=device, dtype=dtype)
    
    # Convert to spherical coordinates (longitude and latitude)
    # ERP mapping: longitude ∈ [-π, π], latitude ∈ [π/2, -π/2]
    longitude = (grid_w / (width - 1)) * 2 * pi - pi  # [-π, π]
    latitude = pi/2 - (grid_h / (height - 1)) * pi    # [π/2, -π/2]
    
    # Convert to Cartesian coordinates
    x = torch.cos(latitude) * torch.cos(longitude)
    y = torch.cos(latitude) * torch.sin(longitude) 
    z = torch.sin(latitude)
    
    # Return 3D Cartesian coordinates
    coords_3d = torch.stack([x, y, z], dim=-1)  # (H, W, 3)
    return coords_3d

def get_3d_spherical_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    interpolation_scale=1.0,
    base_size=16,
    device: Optional[torch.device] = None,
    output_type: str = "pt",
):
    """
    Create 3D spherical position encoding for ERP images
    
    Args:
        embed_dim (int): Embedding dimension
        grid_size (int): Grid size (height and width)
        cls_token (bool): Whether to add classification token
        extra_tokens (int): Number of extra tokens
        interpolation_scale (float): Interpolation scale
        base_size (int): Base size
        device: torch device
        output_type (str): Output type ("pt" or "np")
        
    Returns:
        torch.Tensor: 3D spherical position encoding [grid_size * grid_size, embed_dim]
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)
    
    height, width = grid_size
    
    # Generate ERP spherical coordinates, ensuring float32 data type
    coords_3d = erp_to_spherical_coords(height, width, device=device, dtype=torch.float32)  # (H, W, 3)
    
    # Flatten 3D coordinates
    coords_3d_flat = coords_3d.reshape(-1, 3)  # (H*W, 3)
    
    # Generate spherical position encoding
    pos_embed = get_3d_spherical_pos_embed_from_coords(embed_dim, coords_3d_flat, base_size, interpolation_scale)
    
    if cls_token and extra_tokens > 0:
        pos_embed = torch.cat([torch.zeros([extra_tokens, embed_dim], device=device, dtype=torch.float32), pos_embed], dim=0)
    
    return pos_embed

def get_3d_spherical_pos_embed_from_coords(embed_dim, coords_3d, base_size=16, interpolation_scale=1.0):
    """
    Generate position encoding from 3D spherical coordinates
    
    Args:
        embed_dim (int): Embedding dimension
        coords_3d (torch.Tensor): 3D Cartesian coordinates (N, 3)
        base_size (int): Base size for frequency scaling
        interpolation_scale (float): Interpolation scale
        
    Returns:
        torch.Tensor: Position encoding (N, embed_dim)
    """
    if embed_dim % 3 != 0:
        raise ValueError("embed_dim must be divisible by 3 for 3D spherical encoding")
    
    # Embedding dimension allocated to each coordinate dimension
    dim_per_coord = embed_dim // 3
    
    # Ensure scaling parameters are tensors with correct data type and device
    interpolation_scale = torch.tensor(interpolation_scale, device=coords_3d.device, dtype=coords_3d.dtype)
    base_size = torch.tensor(base_size, device=coords_3d.device, dtype=coords_3d.dtype)
    
    # Generate position encoding for each coordinate dimension
    encodings = []
    for i in range(3):  # x, y, z
        coord = coords_3d[:, i]  # (N,)
        
        # Apply scaling
        coord_scaled = coord / interpolation_scale * base_size
        
        # Use get_1d_sincos_pos_embed_from_grid to generate encoding
        coord_encoding = get_1d_sincos_pos_embed_from_grid(
            dim_per_coord, coord_scaled, output_type="pt"
        )  # (N, dim_per_coord)
        
        # Ensure correct data type is returned
        coord_encoding = coord_encoding.to(dtype=coords_3d.dtype)
        
        encodings.append(coord_encoding)
    
    # Concatenate encodings from all coordinates
    pos_embed = torch.cat(encodings, dim=1)  # (N, embed_dim)
    
    return pos_embed

def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    interpolation_scale=1.0,
    base_size=16,
    device: Optional[torch.device] = None,
    output_type: str = "np",
):
    """
    Creates 2D sinusoidal positional embeddings.

    Args:
        embed_dim (`int`):
            The embedding dimension.
        grid_size (`int`):
            The size of the grid height and width.
        cls_token (`bool`, defaults to `False`):
            Whether or not to add a classification token.
        extra_tokens (`int`, defaults to `0`):
            The number of extra tokens to add.
        interpolation_scale (`float`, defaults to `1.0`):
            The scale of the interpolation.

    Returns:
        pos_embed (`torch.Tensor`):
            Shape is either `[grid_size * grid_size, embed_dim]` if not using cls_token, or `[1 + grid_size*grid_size,
            embed_dim]` if using cls_token
    """
    if output_type == "np":
        deprecation_message = (
            "`get_2d_sincos_pos_embed` uses `torch` and supports `device`."
            " `from_numpy` is no longer required."
            "  Pass `output_type='pt' to use the new version now."
        )
        deprecate("output_type=='np'", "0.33.0", deprecation_message, standard_warn=False)
        return get_2d_sincos_pos_embed_np(
            embed_dim=embed_dim,
            grid_size=grid_size,
            cls_token=cls_token,
            extra_tokens=extra_tokens,
            interpolation_scale=interpolation_scale,
            base_size=base_size,
        )
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = (
        torch.arange(grid_size[0], device=device, dtype=torch.float32)
        / (grid_size[0] / base_size)
        / interpolation_scale
    )
    grid_w = (
        torch.arange(grid_size[1], device=device, dtype=torch.float32)
        / (grid_size[1] / base_size)
        / interpolation_scale
    )
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # here w goes first
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, output_type=output_type)
    if cls_token and extra_tokens > 0:
        pos_embed = torch.concat([torch.zeros([extra_tokens, embed_dim]), pos_embed], dim=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid, output_type="np"):
    r"""
    This function generates 2D sinusoidal positional embeddings from a grid.

    Args:
        embed_dim (`int`): The embedding dimension.
        grid (`torch.Tensor`): Grid of positions with shape `(H * W,)`.

    Returns:
        `torch.Tensor`: The 2D sinusoidal positional embeddings with shape `(H * W, embed_dim)`
    """
    if output_type == "np":
        deprecation_message = (
            "`get_2d_sincos_pos_embed_from_grid` uses `torch` and supports `device`."
            " `from_numpy` is no longer required."
            "  Pass `output_type='pt' to use the new version now."
        )
        deprecate("output_type=='np'", "0.33.0", deprecation_message, standard_warn=False)
        return get_2d_sincos_pos_embed_from_grid_np(
            embed_dim=embed_dim,
            grid=grid,
        )
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0], output_type=output_type)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1], output_type=output_type)  # (H*W, D/2)

    emb = torch.concat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb