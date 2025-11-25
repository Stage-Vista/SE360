#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterate to convert fine masks into rectangular bbox masks.
Handles ERP image masks, considering left-right boundary connectivity.
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def find_erp_bbox(mask):
    """
    Find the bounding box of an ERP image mask, considering left-right boundary connectivity.
    
    Args:
        mask: Binary mask image (H, W)
        
    Returns:
        tuple: (top, bottom, left, right) bounding box coordinates
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    
    # Find coordinates of all non-zero pixels
    coords = np.where(mask > 0)
    
    if len(coords[0]) == 0:
        return None
    
    rows, cols = coords
    height, width = mask.shape
    
    # Vertical boundaries (simple top and bottom)
    top = np.min(rows)
    bottom = np.max(rows)
    
    # Horizontal direction needs to consider ERP connectivity
    # Calculate coordinates of all unique columns
    unique_cols = np.unique(cols)
    
    # Check if it crosses the left-right boundaries
    if len(unique_cols) <= 1:
        # Only one column or no columns
        left = unique_cols[0] if len(unique_cols) > 0 else 0
        right = unique_cols[0] if len(unique_cols) > 0 else 0
        return (top, bottom, left, right)
    
    # Sort column coordinates
    sorted_cols = np.sort(unique_cols)
    gaps = np.diff(sorted_cols)
    
    if len(gaps) == 0:
        # Only one column
        left = right = sorted_cols[0]
        return (top, bottom, left, right)
    
    max_gap_idx = np.argmax(gaps)
    max_gap = gaps[max_gap_idx]
    
    # Improved cross-boundary detection logic
    # 1. The gap must be large enough (more than 40% of width)
    # 2. And both left/right parts must be close to their respective boundaries
    gap_threshold = width * 0.4
    
    if max_gap > gap_threshold:
        # Split point is at the maximum gap
        split_idx = max_gap_idx + 1
        
        # Left part: columns before the gap (smaller coordinates)
        left_part = sorted_cols[:split_idx]
        # Right part: columns after the gap (larger coordinates)  
        right_part = sorted_cols[split_idx:]
        
        # Check if it really crosses the boundary:
        # Left part should be near the left boundary (min value close to 0)
        # Right part should be near the right boundary (max value close to width-1)
        left_near_boundary = np.min(left_part) < width * 0.15  # Left part near left boundary
        right_near_boundary = np.max(right_part) > width * 0.85  # Right part near right boundary
        
        if left_near_boundary and right_near_boundary:
            # It is indeed a cross-boundary case
            # Calculate possibilities for two types of bboxes
            # Option 1: Normal bbox (from min of left part to max of right part)
            bbox1_left = np.min(left_part)
            bbox1_right = np.max(right_part)
            bbox1_width = bbox1_right - bbox1_left + 1
            
            # Option 2: Cross-boundary bbox (range of left part + range of right part)
            bbox2_left_start = np.min(left_part)
            bbox2_left_end = np.max(left_part)
            bbox2_right_start = np.min(right_part)
            bbox2_right_end = np.max(right_part)
            bbox2_width = (bbox2_left_end - bbox2_left_start + 1) + (bbox2_right_end - bbox2_right_start + 1)
            
            # Choose the smaller bbox
            if bbox2_width < bbox1_width:
                # Cross-boundary case: return special marker
                # left = min of right part, right = max of left part
                # Thus, right < left indicates a boundary crossing
                left = bbox2_right_start
                right = bbox2_left_end
            else:
                # Normal case
                left = bbox1_left
                right = bbox1_right
        else:
            # Although gap is large, it's not near boundaries, treat as normal
            left = np.min(unique_cols)
            right = np.max(unique_cols)
    else:
        # No boundary crossing, treat as normal
        left = np.min(unique_cols)
        right = np.max(unique_cols)
    
    return (top, bottom, left, right)


def create_bbox_mask(original_mask, bbox):
    """
    Create a rectangular mask based on the bounding box.
    
    Args:
        original_mask: Original mask
        bbox: (top, bottom, left, right) bounding box
        
    Returns:
        numpy.ndarray: Rectangular bbox mask
    """
    if bbox is None:
        return np.zeros_like(original_mask)
    
    top, bottom, left, right = bbox
    height, width = original_mask.shape
    
    bbox_mask = np.zeros_like(original_mask)
    
    # Handle cross-boundary cases
    if right < left:  # Crosses left and right boundaries
        # Left side of the image (corresponds to the right part of the object logic above)
        bbox_mask[top:bottom+1, left:] = 255
        # Right side of the image
        bbox_mask[top:bottom+1, :right+1] = 255
    else:
        # Normal case
        bbox_mask[top:bottom+1, left:right+1] = 255
    
    return bbox_mask


def process_mask_file(mask_path):
    """
    Process a single mask file.
    
    Args:
        mask_path: Path to the mask file
        
    Returns:
        bool: Whether processing was successful
    """
    try:
        # Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Cannot read mask file: {mask_path}")
            return False
        
        # Find bounding box
        bbox = find_erp_bbox(mask)
        
        if bbox is None:
            print(f"Mask is empty: {mask_path}")
            # Create empty bbox mask
            bbox_mask = np.zeros_like(mask)
        else:
            # Create rectangular mask
            bbox_mask = create_bbox_mask(mask, bbox)
        
        # Save bbox mask
        output_path = mask_path.replace('full_mask.png', 'full_bbox_mask.png')
        cv2.imwrite(output_path, bbox_mask)
        
        return True
        
    except Exception as e:
        print(f"Error processing file {mask_path}: {str(e)}")
        return False


def main():
    """Main function"""
    base_path = "./data/Matterport3D/mp3d_base"
    
    # Find all full_mask.png files
    pattern = os.path.join(base_path, "*/matterport_stitched_images/*/carries_results/*/full_mask.png")
    mask_files = glob.glob(pattern)
    
    print(f"Found {len(mask_files)} mask files")
    
    if len(mask_files) == 0:
        print("No mask files found, please check if the path is correct.")
        return
    
    # Process each mask file
    success_count = 0
    total_count = len(mask_files)
    
    for mask_path in tqdm(mask_files, desc="Processing mask files"):
        if process_mask_file(mask_path):
            success_count += 1
    
    print(f"\nProcessing complete:")
    print(f"Total files: {total_count}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed count: {total_count - success_count}")


if __name__ == "__main__":
    main()