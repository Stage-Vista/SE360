import torch
import torch.nn as nn
from ..modules.transformer import BasicTransformerBlock, SphericalPE
from .utils import get_coords, get_masks,get_equicoords
from einops import rearrange, repeat


class WarpAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.transformer = BasicTransformerBlock(
            dim, dim//32, 32, context_dim=dim)
        self.pe = SphericalPE(dim//4)

    def forward(self, pers_x, equi_x, cameras):
        bm, c, pers_h, pers_w = pers_x.shape
        b, c, equi_h, equi_w = equi_x.shape
        m = bm // b
        pers_masks, equi_masks = get_masks( # Provide pixel-level mapping and masks from perspective to panoramic (or vice versa)
            pers_h, pers_w, equi_h, equi_w, cameras, pers_x.device, pers_x.dtype)
        pers_coords, equi_coords = get_coords( # Generate spherical coordinates (longitude and latitude) for each pixel in perspective and panoramic images, used to describe geometric projection relationships
            pers_h, pers_w, equi_h, equi_w, cameras, pers_x.device, pers_x.dtype)

        # # cross attention from perspective to equirectangular
        # pers_xs = rearrange(pers_xs, '(b m) c h w -> b m c h w', m=m)
        # pers_masks = rearrange(pers_masks, '(b m) ... -> b m ...', m=m)
        # equi_masks = rearrange(equi_masks, '(b m) ... -> b m ...', m=m)
        # pers_xs_out, equi_xs_out = [], []
        # for pers_x, equi_x, pers_mask, equi_mask in zip(pers_xs, equi_xs, pers_masks, equi_masks):
        #     key_value = rearrange(pers_x, 'm c h w -> (m h w) c', m=m)
        #     query = rearrange(equi_x, 'c h w -> (h w) c')
        #     pers_mask = rearrange(pers_mask, 'm h w -> b (m h w)', m=m)
        #     out = self.transformer(query, key_value, pers_masks)

        # add positional encoding
        pers_pe = self.pe(pers_coords)
        pers_pe = rearrange(pers_pe, 'b h w c -> b c h w')
        pers_x_wpe = pers_x + pers_pe
        equi_pe = self.pe(equi_coords)
        equi_pe = repeat(equi_pe, 'h w c -> b c h w', b=b)
        equi_x_wpe = equi_x + equi_pe

        # cross attention from perspective to equirectangular
        query = rearrange(equi_x, 'b c h w -> b (h w) c')
        key_value = rearrange(pers_x_wpe, '(b m) c h w -> b (m h w) c', m=m)
        pers_masks = rearrange(pers_masks, '(b m) eh ew ph pw -> b (eh ew) (m ph pw)', m=m)
        equi_pe = rearrange(equi_pe, 'b c h w -> b (h w) c')
        equi_x_out = self.transformer(query, key_value, mask=pers_masks, query_pe=equi_pe)

        # cross attention from equirectangular to perspective
        query = rearrange(pers_x, '(b m) c h w -> b (m h w) c', m=m)
        key_value = rearrange(equi_x_wpe, 'b c h w -> b (h w) c')
        equi_masks = rearrange(equi_masks, '(b m) ph pw eh ew -> b (m ph pw) (eh ew)', m=m)
        pers_pe = rearrange(pers_pe, '(b m) c h w -> b (m h w) c', m=m)
        pers_x_out = self.transformer(query, key_value, mask=equi_masks, query_pe=pers_pe)

        pers_x_out = rearrange(pers_x_out, 'b (m h w) c -> (b m) c h w', m=m, h=pers_h, w=pers_w)
        equi_x_out = rearrange(equi_x_out, 'b (h w) c -> b c h w', h=equi_h, w=equi_w)
        return pers_x_out, equi_x_out
    
class SphericalAttnhhhhhhh(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.transformer = BasicTransformerBlock(
            dim, dim//32, 32, context_dim=None)
        # Spherical harmonic position encoding, input dimension is dim//8, output dimension is dim//8 * 6
        self.sph_pe = SphericalPE(dim//8)  # Output dimension is (dim//8)*6 = 3*dim//4
        # Add Fourier position encoding, output dimension is dim//4
        self.fourier_pe = nn.Sequential(
            nn.Linear(4, dim//4),
            nn.GELU(),
            nn.Linear(dim//4, dim//4)
        )
        # Modify projection layer input dimension to dim, because we concatenated 3*dim//4 + dim//4 = dim dimensional encoding
        self.final_proj = nn.Linear(dim, dim)
        
        print(f"Initializing SphericalAttn with dim={dim}")

    def get_fourier_features(self, coords, num_bands=4):
        # Generate Fourier features
        freq_bands = 2.0 ** torch.linspace(0, num_bands-1, num_bands, device=coords.device)
        freq_bands = freq_bands.view(-1, 1, 1, 1)
        
        angles = coords * freq_bands * 2 * torch.pi
        sin_features = torch.sin(angles)
        cos_features = torch.cos(angles)
        fourier_features = torch.cat([sin_features, cos_features], dim=-1)
        return fourier_features

    def get_spherical_coords(self, h, w, device, dtype):
        """Generate spherical coordinates"""
        phi = torch.linspace(0, 2*torch.pi, w, device=device, dtype=dtype)
        theta = torch.linspace(0, torch.pi, h, device=device, dtype=dtype)
        
        # Generate grid
        theta, phi = torch.meshgrid(theta, phi, indexing='ij')
        
        # Calculate spherical coordinates
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        
        coords = torch.stack([x, y, z], dim=-1)  # [h, w, 3]
        return coords

    def forward(self, equi_x):
        b, c, h, w = equi_x.shape
        # print(f"Input shape: b={b}, c={c}, h={h}, w={w}")
        
        # Get spherical coordinates
        equi_coords = self.get_spherical_coords(h, w, equi_x.device, equi_x.dtype)
        
        # Add spherical harmonic position encoding
        sph_pe = self.sph_pe(equi_coords)  # [h, w, 3*dim//4]
        # Use torch operations instead of einops
        sph_pe = sph_pe.permute(2, 0, 1)  # [3*dim//4, h, w]
        sph_pe = sph_pe.unsqueeze(0).expand(b, -1, -1, -1)  # [b, 3*dim//4, h, w]
        # print(f"sph_pe shape: {sph_pe.shape}")
        
        # Add Fourier position encoding
        grid_coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, h, device=equi_x.device),
            torch.linspace(-1, 1, w, device=equi_x.device),
            indexing='ij'
        ), dim=-1)  # [h, w, 2]
        
        fourier_features = self.get_fourier_features(grid_coords.unsqueeze(0))  # [1, h, w, 4]
        fourier_pe = self.fourier_pe(fourier_features[0])  # [h, w, dim//4]
        # Use torch operations instead of einops
        fourier_pe = fourier_pe.permute(2, 0, 1)  # [dim//4, h, w]
        fourier_pe = fourier_pe.unsqueeze(0).expand(b, -1, -1, -1)  # [b, dim//4, h, w]
        # print(f"fourier_pe shape: {fourier_pe.shape}")
        
        # Combine position encodings
        combined_pe = torch.cat([sph_pe, fourier_pe], dim=1)  # [b, dim, h, w]
        # print(f"combined_pe shape after cat: {combined_pe.shape}")
        
        # Adjust dimensions before linear projection
        combined_pe = combined_pe.flatten(2)  # [b, dim, h*w]
        combined_pe = combined_pe.transpose(1, 2)  # [b, h*w, dim]
        # print(f"combined_pe shape before projection: {combined_pe.shape}")
        # print(f"final_proj weight shape: {self.final_proj.weight.shape}")
        
        combined_pe = self.final_proj(combined_pe)  # [b, h*w, dim]
        combined_pe = combined_pe.transpose(1, 2).reshape(b, -1, h, w)  # [b, dim, h, w]
        
        x_wpe = equi_x + combined_pe

        # Self-attention
        query = rearrange(equi_x, 'b c h w -> b (h w) c')
        key_value = rearrange(x_wpe, 'b c h w -> b (h w) c')
        pe = rearrange(combined_pe, 'b c h w -> b (h w) c')
        
        # Apply transformer, using key_value as context
        out = self.transformer(query, context=key_value, query_pe=pe)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        
        return out
    
    
class SphericalAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.transformer = BasicTransformerBlock(
            dim, dim//32, 32, context_dim=dim)
        self.pe = SphericalPE(dim//4)

    def forward(self, equi_x):
        # bm, c, pers_h, pers_w = pers_x.shape
        b, c, equi_h, equi_w = equi_x.shape
        # m = bm // b
        # pers_masks, equi_masks = get_masks( # Provide pixel-level mapping and masks from perspective to panoramic (or vice versa)
        #     pers_h, pers_w, equi_h, equi_w, cameras, pers_x.device, pers_x.dtype)
        equi_coords = get_equicoords( # Generate spherical coordinates (longitude and latitude) for each pixel in perspective and panoramic images, used to describe geometric projection relationships
            equi_h, equi_w,  equi_x.device, equi_x.dtype)

        # # cross attention from perspective to equirectangular
        # pers_xs = rearrange(pers_xs, '(b m) c h w -> b m c h w', m=m)
        # pers_masks = rearrange(pers_masks, '(b m) ... -> b m ...', m=m)
        # equi_masks = rearrange(equi_masks, '(b m) ... -> b m ...', m=m)
        # pers_xs_out, equi_xs_out = [], []
        # for pers_x, equi_x, pers_mask, equi_mask in zip(pers_xs, equi_xs, pers_masks, equi_masks):
        #     key_value = rearrange(pers_x, 'm c h w -> (m h w) c', m=m)
        #     query = rearrange(equi_x, 'c h w -> (h w) c')
        #     pers_mask = rearrange(pers_mask, 'm h w -> b (m h w)', m=m)
        #     out = self.transformer(query, key_value, pers_masks)

        # add positional encoding

        
        equi_pe = self.pe(equi_coords)
        equi_pe = repeat(equi_pe, 'h w c -> b c h w', b=b)
        equi_x_wpe = equi_x + equi_pe

        # cross attention from perspective to equirectangular
        query = rearrange(equi_x, 'b c h w -> b (h w) c')
        key_value = rearrange(equi_x_wpe, 'b c h w -> b (h w) c')
        
        equi_pe = rearrange(equi_pe, 'b c h w -> b (h w) c')
        equi_x_out = self.transformer(query, key_value, mask=None, query_pe=equi_pe)

        
        equi_x_out = rearrange(equi_x_out, 'b (h w) c -> b c h w', h=equi_h, w=equi_w)
        return equi_x_out
    
