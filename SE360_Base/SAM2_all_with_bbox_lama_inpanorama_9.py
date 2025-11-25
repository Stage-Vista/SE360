import torch
import sys
import os
import argparse
import numpy as np
import json
from pathlib import Path
from matplotlib import pyplot as plt
import cv2
import glob
import time
from torch.utils.data import Dataset, DataLoader

# ===== Before importing sam2, add Grounded_SAM_2 directory to sys.path =====
CUR_DIR = Path(__file__).resolve().parent                 # group_sam2
GSAM2_DIR = CUR_DIR / "Grounded_SAM_2"                    # group_sam2/Grounded_SAM_2
# If not already in search path, insert at the beginning
if str(GSAM2_DIR) not in sys.path:
    sys.path.insert(0, str(GSAM2_DIR))
# ================================================================

from Grounded_SAM_2.sam2.build_sam import build_sam2
from Grounded_SAM_2.sam2.sam2_image_predictor import SAM2ImagePredictor
# Add py360 imports
from py360.e2p import e2p
from py360.p2e import p2e

# Add main project path to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Note: SAM-2 is already imported at the top, like the following two lines
# from segment_anything import sam_model_registry, SamPredictor
from lama.lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point
from utils.Segment.seg_utils import get_5_points, get_center_point, get_3_points, get_9_points

# Add function: Filter small regions in the mask
def filter_small_regions(mask, min_size=100):
    """
    Filter small regions/isolated points in the mask
    Args:
        mask: Binary mask
        min_size: Minimum region size; connected components smaller than this will be removed
    Returns:
        Filtered mask
    """
    # Ensure mask is a binary image
    if mask.dtype != np.bool_:
        binary_mask = mask > 0
    else:
        binary_mask = mask
    
    # Find all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8)
    
    # Create a new mask, keeping only regions larger than the threshold
    filtered_mask = np.zeros_like(binary_mask)
    # Start from 1, as 0 is background
    for i in range(1, num_labels):
        # stats[i, cv2.CC_STAT_AREA] is the area of the i-th region
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered_mask[labels == i] = True
    
    # Convert mask back to original type
    if mask.dtype != np.bool_:
        filtered_mask = filtered_mask.astype(mask.dtype) * np.max(mask)
    
    return filtered_mask

# Add unified output resolution constants
OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 1024

# Add function to find ERP bounding box (from make_mask_11.py)
def find_erp_bbox(mask):
    """
    Find the bounding box of an ERP image mask, considering left-right boundary connectivity.
    
    Args:
        mask: Binary mask image (H, W)
        
    Returns:
        tuple: (top, bottom, left, right) bounding box coordinates, or None if mask is empty
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
                print(f"  Detected cross-boundary mask (Right starts at {left}, Left ends at {right}). Will generate two BBoxes.")
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


def create_bounding_box_mask(mask):
    """
    Create a rectangular mask based on the bounding box.
    Handles ERP image masks, considering left-right boundary connectivity.
    
    Args:
        mask: Original mask (numpy array)
        
    Returns:
        numpy.ndarray: Rectangular bbox mask
    """
    # Find bounding box using ERP-aware algorithm
    bbox = find_erp_bbox(mask)
    
    if bbox is None:
        return np.zeros_like(mask)
    
    top, bottom, left, right = bbox
    height, width = mask.shape[:2]
    
    # Get max value of mask to maintain consistency
    max_val = np.max(mask)
    if max_val == 0:
        max_val = 255  # Default to 255 for empty masks
    
    bbox_mask = np.zeros_like(mask)
    
    # Handle cross-boundary cases
    if right < left:  # Crosses left and right boundaries
        # Left side of the image (corresponds to the right part of the object logic above)
        bbox_mask[top:bottom+1, left:] = max_val
        # Right side of the image
        bbox_mask[top:bottom+1, :right+1] = max_val
    else:
        # Normal case
        bbox_mask[top:bottom+1, left:right+1] = max_val
    
    return bbox_mask

# Helper function: Extract Scene ID and View ID from file path
def get_scene_view_from_path(path):
    """Extract Scene ID and View ID from file path"""
    # New path format: /.../mp3d_base/[scene]/matterport_stitched_images/[view]/carries_results/[view]_obj[N]/...
    parts = path.split('/')

    try:
        # Find directory containing scene ID (assuming base directory name contains 'inpaint')
        base_dir_index = -1
        for i, part in enumerate(parts):
            if 'base' in part: # Adapts to 'mp3d_inpaint_all' or 'mp3d_inpaint_all_demo' etc.
                base_dir_index = i
                break
        if base_dir_index == -1:
            raise ValueError("Cannot find base directory ('base')")

        scene_id = parts[base_dir_index + 1]

        # Find matterport_stitched_images
        stitched_idx = parts.index('matterport_stitched_images', base_dir_index)
        view_id = parts[stitched_idx + 1]

        return scene_id, view_id
    except (ValueError, IndexError) as e:
        print(f"Warning: Cannot parse Scene ID and View ID from path: {path} - Error: {e}")
        return None, None

# Check if directory is empty
def is_dir_empty(dir_path):
    return len(os.listdir(dir_path)) == 0

# Modified to independent task collection function
def collect_tasks(base_dir, regenerate_existing=False):
    tasks = []
    # Update glob pattern to match new path structure
    # Path format: /.../[base_dir]/[scene]/matterport_stitched_images/[view]/carries_results/[view]_obj[N]/[view]_obj[N]_detection.json
    detection_files = glob.glob(os.path.join(base_dir, "**", "matterport_stitched_images", "**", "carries_results", "*_obj*", "*_detection.json"), recursive=True)

    for detection_file in detection_files:
        # Read detection JSON file
        try:
            with open(detection_file, 'r', encoding='utf-8') as f:
                detection_data = json.load(f)
        except Exception as e:
            print(f"Skipping {detection_file}: Cannot read JSON file - {str(e)}")
            continue

        # Extract Scene ID and View ID from file path (using updated function)
        scene_id, view_id = get_scene_view_from_path(detection_file)
        if not scene_id or not view_id:
            print(f"Skipping {detection_file}: Cannot identify scene/view information")
            continue

        # Get original ERP image path directly from JSON
        original_image_path = detection_data.get("original_erp_image")
        if not original_image_path:
            print(f"Skipping {detection_file}: JSON missing 'original_erp_image' field")
            continue

        # --- Add path modification logic ---
        if isinstance(original_image_path, str) and original_image_path.startswith("/local"):
            # Find the part after /local/
            # Example: /local/scratch/haoyi/project/PanFusion/data/Matterport3D/data/mp3d_skybox/1LXtFkjw3qL/...
            # We need to keep the part starting from mp3d_skybox
            try:
                # Find start index of 'mp3d_skybox'
                skybox_index = original_image_path.index("mp3d_skybox")
                # Extract 'mp3d_skybox' and subsequent parts
                relative_path = original_image_path[skybox_index:]
                # Construct new full path
                new_base_path = "/home/lab401/zhongfile/Project/PanFusion/data/Matterport3D"
                original_image_path = os.path.join(new_base_path, relative_path)
                print(f"  Path modified to: {original_image_path}")
            except ValueError:
                print(f"Warning: Path {original_image_path} starts with /local but 'mp3d_skybox' not found. Cannot modify path.")
                # Decide whether to skip or use original path
                # continue # if skipping is needed
        # --- End path modification logic ---

        # Check if original image exists
        if not os.path.exists(original_image_path):
            print(f"Skipping {detection_file}: Original image does not exist - {original_image_path}")
            continue

        # Use directory containing detection.json as output directory
        output_dir = os.path.dirname(detection_file)

        # Check if processing results already exist (e.g., check for full_mask.png)
        # Note: This check logic might need adjustment based on specific needs
        full_mask_path = os.path.join(output_dir, "full_mask.png")
        if not regenerate_existing and os.path.exists(full_mask_path):
            print(f"Skipping {detection_file}: Processing result 'full_mask.png' already exists")
            continue

        # Add to task list
        tasks.append({
            "detection_file": detection_file,
            "detection_data": detection_data,
            "original_image_path": original_image_path, # Use potentially modified path
            "output_dir": output_dir,
            "scene_id": scene_id,
            "view_id": view_id
        })

    return tasks

# Modified to function processing a single detection item
def process_single_detection(item, args, predictor, device):
    """Process a single detection task"""
    detection_file = item["detection_file"]
    detection_data = item["detection_data"]
    original_image_path = item["original_image_path"]
    output_dir = item["output_dir"]
    scene_id = item["scene_id"]
    view_id = item["view_id"]

    print(f"\nProcessing Scene: {scene_id}, View: {view_id}")
    print(f"Detection file: {detection_file}")

    try:
        # Get perspective transform parameters
        params = detection_data.get("perspective_transform_params")
        if not params:
            print(f"Skipping: {detection_file} - Missing 'perspective_transform_params'")
            return False # Return failure

        u_deg = params.get("center_u_deg")
        v_deg = params.get("center_v_deg")
        hfov_deg = params.get("final_hfov_deg")
        vfov_deg = params.get("final_vfov_deg")
        pers_h = params.get("output_height")
        pers_w = params.get("output_width")
        erp_h = params.get("erp_image_height") # Get original ERP dimensions
        erp_w = params.get("erp_image_width")

        if None in [u_deg, v_deg, hfov_deg, vfov_deg, pers_h, pers_w, erp_h, erp_w]:
            print(f"Skipping: {detection_file} - Missing necessary parameters in 'perspective_transform_params'")
            return False

        # --- Add parameter type checking and conversion ---
        try:
            # Attempt to convert angles to floats and dimensions to integers
            hfov_deg = float(hfov_deg)
            vfov_deg = float(vfov_deg)
            u_deg = float(u_deg)
            v_deg = float(v_deg)
            pers_h = int(pers_h)
            pers_w = int(pers_w)
            erp_h = int(erp_h)
            erp_w = int(erp_w)
            # Optional: Print converted values for debugging
            # print(f"  Param type check passed: hfov={hfov_deg}, vfov={vfov_deg}, u={u_deg}, v={v_deg}, pers_hw=({pers_h},{pers_w}), erp_hw=({erp_h},{erp_w})")
        except (ValueError, TypeError) as type_e:
            print(f"Skipping: {detection_file} - Invalid parameter types in 'perspective_transform_params': {type_e}")
            # Print original values causing error to help locate issue
            print(f"  Invalid param values: hfov='{hfov_deg}', vfov='{vfov_deg}', u='{u_deg}', v='{v_deg}', pers_h='{pers_h}', pers_w='{pers_w}', erp_h='{erp_h}', erp_w='{erp_w}'")
            return False
        # --- End added checks ---

        # Load complete original ERP image (pre-load, used for inpainting later)
        original_erp_img = load_img_to_array(original_image_path)
        if original_erp_img is None or original_erp_img.size == 0:
             print(f"Skipping: {detection_file} - Cannot load original ERP image")
             return False
        # Ensure ERP image dimensions match JSON record (optional but recommended)
        if original_erp_img.shape[0] != erp_h or original_erp_img.shape[1] != erp_w:
             print(f"Warning: {detection_file} - Loaded ERP image dimensions ({original_erp_img.shape[:2]}) do not match JSON ({erp_h}, {erp_w})")
             # Can choose to skip, or continue but p2e results might be inaccurate
             # return False

        # Generate perspective view using e2p
        print(f"  Generating perspective view: u={u_deg:.2f}, v={v_deg:.2f}, fov=({hfov_deg:.2f}, {vfov_deg:.2f}), size=({pers_w}, {pers_h})")
        pers_img = e2p(original_erp_img, hfov_deg, u_deg, v_deg, (pers_h, pers_w), in_rot_deg=0, mode='bilinear')
        # Ensure image is uint8 type for SAM
        img = pers_img.astype(np.uint8)

        # Check if generated perspective view dimensions match expectation
        if img.shape[0] != pers_h or img.shape[1] != pers_w:
             print(f"Warning: {detection_file} - e2p generated perspective dimensions ({img.shape[:2]}) do not match expectation ({pers_h}, {pers_w})")
             # Handle as needed, might need to adjust e2p params or skip

        # Set image for predictor (predictor is initialized externally and passed in)
        predictor.set_image(img) # Use generated perspective view

        # Parse detections to get bounding box info in perspective view
        all_bboxes_pers = []
        detections = detection_data.get("detections", [])
        if not detections:
             print(f"Skipping: {detection_file} - JSON missing 'detections' or is empty")
             return False

        # --- Modification: Check if 'mirror' category exists and record its bbox ---
        is_mirror_case = False
        mirror_bbox_pers = None # Store bbox for mirror
        for det_item in detections:
            original_category = det_item.get("original_category", "")
            bbox_pers = det_item.get("bbox_in_resized_perspective")

            # Check if it is a mirror category
            if isinstance(original_category, str) and 'mirror' in original_category.lower():
                is_mirror_case = True
                print(f"  Detected 'mirror' category, will add bounding box constraints for prediction.")
                # Attempt to get and validate mirror bbox
                if bbox_pers:
                    x1, y1, x2, y2 = bbox_pers
                    x1 = max(0, min(x1, pers_w - 1))
                    y1 = max(0, min(y1, pers_h - 1))
                    x2 = max(0, min(x2, pers_w - 1))
                    y2 = max(0, min(y2, pers_h - 1))
                    if x1 < x2 and y1 < y2:
                        mirror_bbox_pers = [x1, y1, x2, y2]
                        print(f"  Recorded 'mirror' bounding box: {mirror_bbox_pers}")
                        # Break after finding first valid mirror bbox, or continue depending on strategy
                        # break # if only need the first one
                    else:
                         print(f"Warning: 'mirror' detection item bbox_in_resized_perspective invalid or degenerate: {bbox_pers}")
                else:
                    print(f"Warning: 'mirror' detection item missing 'bbox_in_resized_perspective'")

            # Collect all valid bboxes (whether mirror or not)
            if bbox_pers:
                x1, y1, x2, y2 = bbox_pers
                x1 = max(0, min(x1, pers_w - 1))
                y1 = max(0, min(y1, pers_h - 1))
                x2 = max(0, min(x2, pers_w - 1))
                y2 = max(0, min(y2, pers_h - 1))
                if x1 < x2 and y1 < y2:
                    # Avoid duplicate mirror_bbox_pers (if already recorded)
                    current_bbox = [x1, y1, x2, y2]
                    if current_bbox not in all_bboxes_pers:
                         all_bboxes_pers.append(current_bbox)
                # else: # Repeat warning logic, or simplify
                #     print(f"Warning: Invalid or degenerate bbox_in_resized_perspective: {bbox_pers} in {detection_file}")
            # else: # Repeat warning logic
            #     print(f"Warning: detection missing 'bbox_in_resized_perspective': {det_item}")

        # If mirror case but no valid mirror bbox found, issue warning
        if is_mirror_case and mirror_bbox_pers is None:
            print(f"Warning: Detected 'mirror' category but failed to find its valid bounding box. Center point prediction will not use specific bbox constraints.")

        # --- End modification ---

        if not all_bboxes_pers:
            print(f"Skipping: {detection_file} - No valid 'bbox_in_resized_perspective' found")
            return False

        # SAM segmentation logic (using perspective view bbox)
        final_masks = []
        all_generated_points = []
        center_points = []
        all_initial_masks = []

        for bbox in all_bboxes_pers: # Use perspective view bbox
            # First prediction: Based on bbox, no modifications needed
            masks, _, _ = predictor.predict(
                box=np.array(bbox),
                multimask_output=False
            )
            bbox_mask = np.any(masks, axis=0)

            points = get_5_points(bbox_mask)
            center_point = get_center_point(bbox_mask)

            if isinstance(center_point, (list, np.ndarray)):
                if len(center_point) == 1:
                    center_point = center_point[0].tolist() if isinstance(center_point[0], np.ndarray) else center_point[0]
                    center_points.append(center_point)
                else:
                    for point in center_point:
                        point_as_list = point.tolist() if isinstance(point, np.ndarray) else point
                        center_points.append(point_as_list)

            all_generated_points.extend(points)

            # Ensure points array/list is not empty before prediction
            if len(points) > 0:
                point_coords_np = np.array(points)
                if point_coords_np.ndim == 2 and point_coords_np.shape[1] == 2:
                    # Second prediction: If is_mirror_case is True, add current bbox parameter (logic unchanged)
                    predict_args_points = {
                        'point_coords': point_coords_np,
                        'point_labels': np.ones(len(points)),
                        'multimask_output': False
                    }
                    if is_mirror_case:
                        # For point prediction, it's more reasonable to use currently processed bbox as constraint
                        predict_args_points['box'] = np.array(bbox)

                    point_masks, _, _ = predictor.predict(**predict_args_points)
                    points_mask = np.any(point_masks, axis=0)
                else:
                    print(f"Warning: Generated points format incorrect, skipping point prediction: {points}")
                    points_mask = np.zeros_like(bbox_mask) # Create empty mask
            else:
                 points_mask = np.zeros_like(bbox_mask) # If no points, point mask is empty

            initial_mask = np.logical_or(bbox_mask, points_mask)
            initial_mask = filter_small_regions(initial_mask, min_size=30)
            all_initial_masks.append(initial_mask)

        # Second stage processing, using center points
        if center_points:
            valid_center_points = []
            for point in center_points:
                # Add check to ensure point coords are within image range
                if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
                     px, py = point[:2]
                     if 0 <= px < pers_w and 0 <= py < pers_h:
                         valid_center_points.append(point)
                     else:
                         print(f"Skipping out-of-bounds center point: {point}")
                else:
                    print(f"Skipping incorrectly formatted center point: {point}")

            if valid_center_points:
                point_coords = np.array(valid_center_points).reshape(-1, 2)
                point_labels = np.ones(len(point_coords))

                # --- Modification: Third prediction, if is_mirror_case is True, use recorded mirror_bbox_pers ---
                predict_args_center = {
                    'point_coords': point_coords,
                    'point_labels': point_labels,
                    'mask_input': None, # Or pass previous initial_mask as prompt
                    'multimask_output': False
                }

                if is_mirror_case:
                    # Use recorded mirror bbox (if exists)
                    if mirror_bbox_pers is not None:
                        predict_args_center['box'] = np.array(mirror_bbox_pers)
                        print(f"  Using 'mirror' bounding box for center point prediction: {mirror_bbox_pers}")
                    else:
                        # If is_mirror_case is True but mirror_bbox_pers is None, warning already printed
                        # Don't add box constraint, or consider falling back to merged bbox (if needed)
                        print(f"  Center point prediction not using 'mirror' bbox constraint (no valid bbox found).")
                        pass # Do not add box parameter

                refined_masks, _, _ = predictor.predict(**predict_args_center)
                # --- End modification ---

                refined_mask = np.any(refined_masks, axis=0)

                final_mask = refined_mask
                for mask in all_initial_masks:
                    final_mask = np.logical_or(final_mask, mask)

                final_mask = filter_small_regions(final_mask, min_size=30)
                final_masks = [final_mask] # Finally merge into one mask

        # If no valid center points, use initial mask
        if not final_masks and all_initial_masks:
            combined_initial_mask = np.zeros_like(all_initial_masks[0], dtype=bool)
            for mask in all_initial_masks:
                combined_initial_mask = np.logical_or(combined_initial_mask, mask)
            # Apply filter
            combined_initial_mask = filter_small_regions(combined_initial_mask, min_size=30)
            final_masks = [combined_initial_mask]

        # If still no mask, return error
        if not final_masks or not np.any(final_masks[0]): # Check if mask is all False
            print(f"Skipping: {detection_file} - Failed to generate valid mask")
            return False

        # Note: final_masks is now a list containing a single boolean mask
        # combined_mask_pers = final_masks[0] # This is final boolean mask on perspective view
        # Convert to uint8 for subsequent processing and saving
        combined_mask_pers_uint8 = final_masks[0].astype(np.uint8) * 255

        # Dilate mask on perspective view
        if args.dilate_kernel_size is not None:
            # Ensure input is uint8
            dilated_mask_pers = dilate_mask(combined_mask_pers_uint8, args.dilate_kernel_size)
        else:
            dilated_mask_pers = combined_mask_pers_uint8

        # Save results directory
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Removed creation of bounding box mask on perspective view ---
        # combined_bbox_mask_pers = create_bounding_box_mask(dilated_mask_pers) # Use dilated mask

        # --- Removed LAMA inpainting on perspective view ---
        # print(f"  Starting LAMA inpainting on perspective view...")
        # img_inpainted_pers = inpaint_img_with_lama(
        #     img, # Input perspective view
        #     dilated_mask_pers, # Use dilated mask for inpainting
        #     args.lama_config,
        #     args.lama_ckpt,
        #     device=device
        # )
        # print(f"  LAMA inpainting complete.")

        # --- Back-project to panorama ---
        print(f"  Starting back-projection of mask to panorama ({erp_w}x{erp_h})...")

        # --- Removed back-projection of inpainted image patch ---
        # 1. Back-project inpainted image patch
        # Use original ERP image dimensions h, w
        # inpainted_erp_patch = p2e(
        #     img_inpainted_pers, # Inpainted perspective view
        #     (hfov_deg, vfov_deg), u_deg, v_deg, # Ensure tuple passed
        #     erp_h, erp_w, # Output ERP dimensions
        #     in_rot_deg=0,
        #     mode='bilinear' # Image uses bilinear
        # )
        # inpainted_erp_patch = inpainted_erp_patch.astype(np.uint8)

        # 2. Back-project final mask (using dilated one)
        # Mask uses nearest to avoid interpolation blur
        mask_erp_patch = p2e(
            dilated_mask_pers, # Final used mask (dilated)
            (hfov_deg, vfov_deg), u_deg, v_deg, # Ensure tuple passed
            erp_h, erp_w,
            mode='nearest'
        )
        # Ensure back-projection is binary mask (uint8, 0 or 255)
        mask_erp_patch = (mask_erp_patch > 128).astype(np.uint8) * 255
        print(f"  Back-projection of mask complete.")

        # --- Removed back-projection of bounding box mask ---
        # bbox_mask_erp_patch = p2e(
        #     combined_bbox_mask_pers, # Bbox mask on perspective view
        #     (hfov_deg, vfov_deg), u_deg, v_deg,
        #     erp_h, erp_w,
        #     mode='nearest'
        # )
        # bbox_mask_erp_patch = (bbox_mask_erp_patch > 128).astype(np.uint8) * 255

        # --- Perform LAMA inpainting on panorama ---
        print(f"  Starting LAMA inpainting on panorama...")
        # Use original panorama and back-projected mask for inpainting
        full_result = inpaint_img_with_lama(
            original_erp_img, # Input original panorama
            mask_erp_patch,   # Input mask back-projected to panorama
            args.lama_config,
            args.lama_ckpt,
            device=device
        )
        print(f"  LAMA panorama inpainting complete.")

        # --- Merge and Save ---

        # 1. Save full image inpainting result (now directly the inpainted panorama)
        # --- Removed merge logic ---
        # full_result = original_erp_img.copy()
        # projection_mask_bool = mask_erp_patch > 0
        # if full_result.ndim == 3 and projection_mask_bool.ndim == 2:
        #      projection_mask_3channel = np.stack([projection_mask_bool] * full_result.shape[2], axis=-1)
        # else:
        #      projection_mask_3channel = projection_mask_bool
        # np.copyto(full_result, inpainted_erp_patch, where=projection_mask_3channel)

        full_result_path = out_dir / "full_image_inpainted.png"
        save_array_to_img(full_result, full_result_path) # Save LAMA output panorama directly

        # 2. Save mask of full image (back-projected)
        full_mask_path = out_dir / "full_mask.png"
        save_array_to_img(mask_erp_patch, full_mask_path) # Save back-projected mask

        # 3. Create and save rectangular bounding box mask on panorama
        # Use back-projected mask mask_erp_patch to create bounding box
        full_bbox_mask_erp = create_bounding_box_mask(mask_erp_patch)
        full_bbox_mask_path = out_dir / "full_bbox_mask.png"
        save_array_to_img(full_bbox_mask_erp, full_bbox_mask_path) # Save rectangular bbox mask on panorama

        print(f"Successfully processed: {detection_file}")
        return True # Return success

    except Exception as e:
        print(f"Error processing: {detection_file}")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        return False # Return failure

def parse_args():
    parser = argparse.ArgumentParser(description='Batch Process Image Segmentation')
    parser.add_argument("--dilate_kernel_size", type=int, default=15, help="Dilation kernel size")
    parser.add_argument("--sam2_config", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", help="SAM2 model config file path")
    parser.add_argument("--sam_ckpt", type=str, default="checkpoints/sam2.1_hiera_large.pt", help="SAM2 weights file path")
    parser.add_argument("--lama_config", type=str, default="./lama/configs/prediction/default.yaml", help="LAMA config file path")
    parser.add_argument("--lama_ckpt", type=str, default="./checkpoints/big-lama", help="LAMA model checkpoint path")
    parser.add_argument("--base_dir", type=str, default="./data/Matterport3D/mp3d_base/", help="Base directory")
    parser.add_argument("--regenerate",default=True, action="store_true", help="Regenerate existing results")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Starting processing, base directory: {args.base_dir}")
    print(f"Regenerate: {args.regenerate}")
    print(f"SAM2 Config: {args.sam2_config}")
    print(f"SAM2 Weights: {args.sam_ckpt}")
    print(f"LAMA Checkpoint: {args.lama_ckpt}")

    # Collect all tasks
    print("Collecting tasks...")
    tasks = collect_tasks(args.base_dir, args.regenerate)
    total_tasks = len(tasks)
    print(f"Collected {total_tasks} tasks")

    if total_tasks == 0:
        print("No tasks to process, exiting")
        sys.exit(0)

    # -------- Initialize SAM-2 (outside loop) --------
    print("Initializing SAM-2 model...")
    sam2_model = build_sam2(
        config_file=args.sam2_config,
        ckpt_path=args.sam_ckpt,
        device=device
    )
    predictor = SAM2ImagePredictor(sam2_model, mask_threshold=0.0, max_hole_area=8, max_sprinkle_area=1)   # Keep variable name consistent
    print("SAM-2 model initialization complete.")

    # Record start time
    start_time = time.time()

    # Loop through each task
    completed_tasks = 0
    successful_tasks = 0

    for i, task_item in enumerate(tasks):
        print(f"\n--- Processing task {i+1}/{total_tasks} ---")
        success = process_single_detection(task_item, args, predictor, device)
        completed_tasks += 1
        if success:
            successful_tasks += 1

        # Add more detailed progress report here
        current_time = time.time()
        elapsed = current_time - start_time
        avg_time_per_task = elapsed / completed_tasks if completed_tasks > 0 else 0
        estimated_total_time = avg_time_per_task * total_tasks if completed_tasks > 0 else 0
        remaining_time = estimated_total_time - elapsed if completed_tasks > 0 else 0

        print(f"--- Task {i+1}/{total_tasks} processed {'successfully' if success else 'failed'} ---")
        print(f"Progress: {completed_tasks}/{total_tasks} ({successful_tasks} successful)")
        print(f"Elapsed time: {elapsed:.2f}s")
        if completed_tasks > 0:
            print(f"Avg time per task: {avg_time_per_task:.2f}s")
            print(f"Estimated remaining time: {remaining_time:.2f}s")

    # Calculate total elapsed time
    total_elapsed_time = time.time() - start_time
    print(f"\nAll tasks completed. Processed {completed_tasks} tasks, {successful_tasks} successful. Total time: {total_elapsed_time:.2f}s")

    # Clean up GPU memory (optional)
    # torch.cuda.empty_cache()