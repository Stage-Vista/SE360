#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import glob
import argparse
import numpy as np
import cv2
from PIL import Image

# Add current directory to path to import py360 module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from py360.e2p import e2p


def load_erp_image(image_path):
    """Load ERP image"""
    if not os.path.exists(image_path):
        print(f"Warning: Image file does not exist: {image_path}")
        return None
    
    try:
        # Load image using PIL
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Convert to RGB if it's a grayscale image
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        
        return img_array
    except Exception as e:
        print(f"Error: Failed to load image {image_path}: {e}")
        return None


def convert_erp_to_perspective(erp_img, fov_deg=90, u_deg=0, v_deg=0, out_hw=(512, 512)):
    """Convert ERP image to perspective view"""
    try:
        # If fov_deg is a tuple/list, take the maximum value as the unified FOV
        if isinstance(fov_deg, (tuple, list)) and len(fov_deg) == 2:
            # Use the larger FOV value to ensure the required area is covered
            fov_value = max(fov_deg)
        else:
            fov_value = fov_deg
        
        # e2p function uses a single FOV value
        pers_img = e2p(erp_img, fov_value, u_deg, v_deg, out_hw)
        return pers_img
    except Exception as e:
        print(f"Error: Failed to convert ERP to perspective: {e}")
        return None


def create_bbox_mask(mask_img):
    """Convert mask to rectangular bounding box mask"""
    try:
        # Convert to grayscale if necessary
        if len(mask_img.shape) == 3:
            gray_mask = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
        else:
            gray_mask = mask_img
        
        # Find positions of non-zero pixels
        coords = np.where(gray_mask > 0)
        
        if len(coords[0]) == 0:  # If there are no non-zero pixels
            print("      Warning: Mask is empty, creating empty bounding box mask")
            return np.zeros_like(gray_mask)
        
        # Get bounding box
        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        
        print(f"      Bounding box: ({x_min}, {y_min}) -> ({x_max}, {y_max})")
        
        # Create rectangular mask
        bbox_mask = np.zeros_like(gray_mask)
        bbox_mask[y_min:y_max+1, x_min:x_max+1] = 255
        
        return bbox_mask
        
    except Exception as e:
        print(f"      Error: Failed to create bounding box mask: {e}")
        return None


def process_json_file(json_path, fov_expansion=1.2, out_hw=(512, 512)):
    """Process a single JSON file"""
    print(f"Processing JSON file: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error: Cannot read JSON file {json_path}: {e}")
        return False
    
    # Check required fields
    if 'perspective_transform_params' not in data:
        print(f"Warning: Missing field in JSON file: perspective_transform_params")
        return False
    
    perspective_params = data['perspective_transform_params']
    required_params = ['center_u_deg', 'center_v_deg', 'final_hfov_deg', 'final_vfov_deg']
    missing_params = [param for param in required_params if param not in perspective_params]
    if missing_params:
        print(f"Warning: Missing fields in perspective_transform_params: {missing_params}")
        return False
    
    center_u_deg = perspective_params['center_u_deg']
    center_v_deg = perspective_params['center_v_deg']
    final_hfov_deg = perspective_params['final_hfov_deg']
    final_vfov_deg = perspective_params['final_vfov_deg']
    
    # Expand FOV by 1.2 times
    expanded_hfov = final_hfov_deg * fov_expansion
    expanded_vfov = final_vfov_deg * fov_expansion
    
    print(f"  Original FOV: H={final_hfov_deg}°, V={final_vfov_deg}°")
    print(f"  Expanded FOV: H={expanded_hfov}°, V={expanded_vfov}°")
    print(f"  Center Position: U={center_u_deg}°, V={center_v_deg}°")
    
    # Find full_mask.png file in the same directory
    json_dir = os.path.dirname(json_path)
    input_path = os.path.join(json_dir, "full_mask.png")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"    Skipping: full_mask.png does not exist")
        return False
    
    print(f"    Processing: full_mask.png")
    
    # Load mask image
    erp_img = load_erp_image(input_path)
    if erp_img is None:
        print(f"      Error: Failed to load full_mask.png")
        return False
    
    # Output directory is the same as the JSON file directory
    output_dir = os.path.dirname(json_path)
    
    print(f"  Generating perspective view: Center({center_u_deg}°, {center_v_deg}°), FOV({expanded_hfov}°, {expanded_vfov}°)")
    
    # Convert to perspective view using expanded FOV
    # Use the max of horizontal and vertical FOV to ensure sufficient coverage
    max_fov = max(expanded_hfov, expanded_vfov)
    pers_img = convert_erp_to_perspective(
        erp_img, 
        fov_deg=max_fov, 
        u_deg=center_u_deg, 
        v_deg=center_v_deg, 
        out_hw=out_hw
    )
    
    if pers_img is None:
        print(f"      Conversion failed: full_mask.png")
        return False
    
    # Ensure image data is within correct range
    if pers_img.dtype == np.float32 or pers_img.dtype == np.float64:
        pers_img = np.clip(pers_img * 255, 0, 255).astype(np.uint8)
    
    # 1. Save perspective mask
    perspective_output_path = os.path.join(output_dir, "full_mask_perspective.png")
    if pers_img.shape[2] == 4:  # If there is an alpha channel
        cv2.imwrite(perspective_output_path, cv2.cvtColor(pers_img, cv2.COLOR_RGBA2BGRA))
    else:
        cv2.imwrite(perspective_output_path, cv2.cvtColor(pers_img, cv2.COLOR_RGB2BGR))
    print(f"      Saving perspective view to: {perspective_output_path}")
    
    # 2. Create rectangular bounding box mask
    print(f"      Creating rectangular bounding box mask")
    bbox_mask = create_bbox_mask(pers_img)
    if bbox_mask is None:
        print(f"      Error: Failed to create bounding box mask")
        return False
    
    # 3. Save bounding box mask
    bbox_output_path = os.path.join(output_dir, "full_bbox_mask_perspective.png")
    cv2.imwrite(bbox_output_path, bbox_mask)
    print(f"      Saving bounding box mask to: {bbox_output_path}")
    
    return True


def find_json_files(base_path):
    """Find all JSON files under the specified path pattern"""
    # Build search pattern
    pattern = os.path.join(base_path, "*/matterport_stitched_images/*/carries_results/*/*_recaption.json")
    json_files = glob.glob(pattern)
    
    print(f"Found {len(json_files)} JSON files under path {pattern}")
    return json_files


def main():
    parser = argparse.ArgumentParser(description="Batch convert full_mask.png to perspective views and generate rectangular bounding box masks")
    parser.add_argument("--base_path", default="./data/Matterport3D/mp3d_hf_first",
                        help="Base path for Matterport3D dataset")
    parser.add_argument("--fov_expansion", type=float, default=1.2,
                        help="FOV expansion factor (default 1.2)")
    parser.add_argument("--width", type=int, default=1024,
                        help="Output image width")
    parser.add_argument("--height", type=int, default=1024,
                        help="Output image height")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to process (for testing)")
    
    args = parser.parse_args()
    
    # Check if base path exists
    if not os.path.exists(args.base_path):
        print(f"Error: Base path does not exist: {args.base_path}")
        return
    
    # Find JSON files
    json_files = find_json_files(args.base_path)
    
    if not json_files:
        print("No JSON files found")
        return
    
    # Limit number of processed files (if specified)
    if args.max_files is not None:
        json_files = json_files[:args.max_files]
        print(f"Limiting processing to first {args.max_files} files")
    
    # Process each JSON file
    out_hw = (args.height, args.width)
    success_count = 0
    
    for i, json_path in enumerate(json_files):
        print(f"\nProgress: {i+1}/{len(json_files)}")
        
        if process_json_file(json_path, args.fov_expansion, out_hw):
            success_count += 1
    
    print(f"\nProcessing complete! Successfully processed {success_count}/{len(json_files)} files")
    print("Mask files have been saved in their respective JSON file directories:")
    print("- full_mask.png -> full_mask_perspective.png (Perspective mask)") 
    print("- full_mask.png -> full_bbox_mask_perspective.png (Rectangular bounding box mask)")


if __name__ == "__main__":
    main()