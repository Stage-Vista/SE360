import cv2
import numpy as np
from .p2e1 import p2e


# def mp2e(p_imgs, fov_degs, u_degs, v_degs, out_hw, mode=None):
#     merge_image = np.zeros((*out_hw, 3))
#     merge_mask = np.zeros((*out_hw, 3))
#     for p_img, fov_deg, u_deg, v_deg in zip(p_imgs, fov_degs, u_degs, v_degs):
#         # p_img = p_img.permute(1, 2, 0)
#         # p_img = p_img.detach().cpu().numpy().astype(np.float32)
#         img, mask = p2e(p_img, fov_deg, u_deg, v_deg, out_hw, mode)
#         mask = mask.astype(np.float32)
#         img = img.astype(np.float32)

#         weight_mask = np.zeros((img.shape[0], img.shape[1], 3))
#         w = img.shape[1]
#         weight_mask[:, 0:w//2, :] = np.linspace(0, 1, w//2)[..., None]
#         weight_mask[:, w//2:, :] = np.linspace(1, 0, w//2)[..., None]
#         weight_mask, _ = p2e(weight_mask, fov_deg, u_deg, v_deg, out_hw, mode)
#         blur = cv2.blur(mask, (5, 5))
#         blur = blur * mask
#         mask = (blur == 1) * blur + (blur != 1) * blur * 0.05
#         merge_image += img * weight_mask
#         merge_mask += weight_mask

#     merge_image[merge_mask == 0] = 255.
#     merge_mask = np.where(merge_mask == 0, 1, merge_mask)
#     merge_image = (np.divide(merge_image, merge_mask)).astype(np.uint8)
#     return merge_image


import cv2
import numpy as np
import torch
from kornia.geometry.transform import remap

# Assume p2e function already exists, function: project single perspective image to panorama, returns (img_equi, mask_equi)

def mp2e(p_imgs, fov_degs, u_degs, v_degs, out_hw, mode=None):
    """
    Project and synthesize multiple perspective images into one panoramic image (equirectangular).
    
    Args:
        p_imgs (List[np.ndarray] or List[Tensor]): List of perspective images, shape (H, W, 3) or (B, C, H, W).
        fov_degs, u_degs, v_degs (List[float]): Field of view, horizontal angle, vertical angle (degrees) for each perspective image.
        out_hw (tuple): Output panoramic image (height, width).
        mode (str): Interpolation method, can be 'bilinear', 'bicubic', 'nearest', etc.
        
    Returns:
        np.ndarray: Synthesized panoramic image, uint8, shape = (out_h, out_w, 3).
    """
    out_h, out_w = out_hw
    # Use float32 buffer to accumulate pixel values
    merge_image = np.zeros((out_h, out_w, 3), dtype=np.float32)
    # Use float32 buffer to accumulate weights
    merge_weight = np.zeros((out_h, out_w), dtype=np.float32)
    
    # Process each perspective image sequentially
    for p_img, fov_deg, u_deg, v_deg in zip(p_imgs, fov_degs, u_degs, v_degs):
        # Project this viewpoint to panorama
        img_equi, mask_equi = p2e(p_img, fov_deg, u_deg, v_deg, out_hw, mode)
        
        # If np.uint8, first convert to [0,1] float
        if img_equi.dtype == np.uint8:
            img_equi = img_equi.astype(np.float32) / 255.0
        else:
            img_equi = img_equi.astype(np.float32)
        
        mask_equi = mask_equi.astype(np.float32)
        
        # Accumulate pixels
        merge_image += img_equi * mask_equi[..., None]
        # Accumulate weights
        merge_weight += mask_equi
    
    # Avoid division by 0
    eps = 1e-6
    valid_mask = (merge_weight > eps)
    
    # Average the covered pixels
    merge_image[valid_mask] /= merge_weight[valid_mask, None]
    # For uncovered areas, can fill with white (1.0) or black (0.0)
    merge_image[~valid_mask] = 0  # Fill with white here
    
    # Convert back to [0, 255] uint8
    merge_image = np.clip(merge_image * 255.0, 0, 255).astype(np.uint8)
    return merge_image
