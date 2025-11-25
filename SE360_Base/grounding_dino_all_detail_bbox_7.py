# -*- coding: utf-8 -*-
import sys
import os
import json
import cv2
import torch
import numpy as np
import torchvision
import glob
import argparse
import time
# Ensure the path is correct relative to the script's location
# Assuming the script is in a subdirectory and 'utils' is one level up
# Adjust if your structure is different
script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Check if utils path is correct, provide feedback if not
try:
    # Assuming predict_with_grounding_dino_caption is the main detection func needed now
    from utils.Segment.seg_utils import predict_with_grounding_dino_caption, initialize_ground_sam_models
except ImportError as e:
    print(f"Error importing from utils: {e}")
    print(f"Script directory: {script_dir}")
    print(f"Parent directory added to sys.path: {parent_dir}")
    print("Please ensure the 'utils' directory is correctly located relative to the script or adjust the sys.path.append line.")
    sys.exit(1)

import math
from PIL import Image
# Check if py360 is installed, provide feedback if not
try:
    from py360.e2p import e2p
except ImportError:
    print("Error importing py360. Please ensure it is installed (`pip install py360` or similar).")
    sys.exit(1)

import concurrent.futures
from tqdm import tqdm

# ---------------- ① Constants ----------------
MAX_CARRIED_ITEMS = 30      # Max number of carried items to detect initially
MIN_ORIGINAL_BBOX_AREA_RATIO = 0.05 # Min combined area ratio for kept original bboxes
ASSOCIATION_OVERLAP_THRESHOLD = 0.10 # Min overlap for associating carried item with an original item
SURVIVAL_OVERLAP_THRESHOLD = 0.10 # Min overlap for a carried item (of a deleted original) to survive if it overlaps with a KEPT original
# -------------------------------------------------
# Unified output resolution for detection
OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 1024
DINO_RESIZE_DIM = (OUTPUT_WIDTH, OUTPUT_HEIGHT) # (W, H) for cv2 resize

def erp_bbox_to_perspective(erp_image_path, bbox, out_hw=(1024, 1024), is_carried_items_empty=False,
                            override_expansion_factor=None, override_vertical_offset_factor=None):
    """
    Loads a 360 ERP image, expands the specified bbox dynamically, projects the
    expanded region to a perspective view with FOV calculated to tightly fit
    the expanded box (HFOV capped at 125 deg). Output dimensions are calculated
    based on final FOV, aiming for a max dimension of 1024.
    """
    # 1. Load ERP Image
    try:
        img_erp = Image.open(erp_image_path)
        if img_erp.mode != 'RGB':
            img_erp = img_erp.convert('RGB')
        img_erp_np = np.array(img_erp)
        h_erp, w_erp = img_erp_np.shape[:2]
    except FileNotFoundError:
        print(f"Error: ERP image file not found: {erp_image_path}")
        return None, None
    except Exception as e:
        print(f"Error: Error loading ERP image {erp_image_path}: {e}")
        return None, None

    if h_erp <= 0 or w_erp <= 0:
        print(f"Error: Invalid ERP image dimensions: ({h_erp}, {w_erp})")
        return None, None

    # 2. Validate BBox
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        print(f"Error: Invalid bbox format: {bbox}")
        return None, None
    try:
        x1, y1, x2, y2 = map(float, bbox) # Ensure floats
    except (ValueError, TypeError) as e:
        print(f"Error: bbox contains non-numeric coordinates: {bbox} - {e}")
        return None, None

    # Clip bbox to image boundaries
    x1 = max(0.0, min(x1, float(w_erp - 1)))
    y1 = max(0.0, min(y1, float(h_erp - 1)))
    x2 = max(x1 + 1e-6, min(x2, float(w_erp))) # Ensure x2 > x1
    y2 = max(y1 + 1e-6, min(y2, float(h_erp))) # Ensure y2 > y1

    bbox_w_pixels = x2 - x1
    bbox_h_pixels = y2 - y1

    if bbox_w_pixels <= 0 or bbox_h_pixels <= 0:
        print(f"Error: Invalid or negative BBox dimensions: W={bbox_w_pixels}, H={bbox_h_pixels} (from bbox: {bbox})")
        return None, None

    # 3. Calculate BBox center point (pixel coordinates)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # 4. *** Dynamic Expansion Factor and Vertical Offset ***
    ratio_hh = bbox_h_pixels / h_erp
    ratio_wh = bbox_w_pixels / h_erp
    max_ratio = max(ratio_hh, ratio_wh)

    if override_expansion_factor is not None and override_vertical_offset_factor is not None:
        expansion_factor = override_expansion_factor
        vertical_offset_factor = override_vertical_offset_factor
    elif is_carried_items_empty:
        vertical_offset_factor = 0.0
        expansion_factor = 1.3
    else:
        if ratio_hh > 0.5: vertical_offset_factor = 0.05
        elif ratio_hh > 2/5: vertical_offset_factor = 0.1
        elif ratio_hh > 1/3: vertical_offset_factor = 0.15
        elif ratio_hh > 1/4: vertical_offset_factor = 0.2
        elif ratio_hh > 1/5: vertical_offset_factor = 0.3
        elif ratio_hh > 1/6: vertical_offset_factor = 0.35
        else: vertical_offset_factor = 0.4

        factor_huge_bbox = 1.2; factor_large_bbox = 1.3; factor_medium_large_bbox = 1.4
        factor_medium_bbox = 1.6; factor_medium_small_bbox = 1.7; factor_small_bbox = 1.8
        factor_tiny_bbox = 1.9

        if max_ratio > 0.5: expansion_factor = factor_huge_bbox
        elif max_ratio > 2/5: expansion_factor = factor_large_bbox
        elif max_ratio > 1/3: expansion_factor = factor_medium_large_bbox
        elif max_ratio > 1/4: expansion_factor = factor_medium_bbox
        elif max_ratio > 1/5: expansion_factor = factor_medium_small_bbox
        elif max_ratio > 1/6: expansion_factor = factor_small_bbox
        else: expansion_factor = factor_tiny_bbox

    vertical_offset = bbox_h_pixels * vertical_offset_factor
    adjusted_cy = cy - vertical_offset
    adjusted_cy = max(0, min(adjusted_cy, h_erp - 1))
    center_u = (cx / w_erp - 0.5) * 360
    center_v = -(adjusted_cy / h_erp - 0.5) * 180

    # 5. Convert *adjusted* center pixel coords to spherical coordinates (degrees) -> Angular size
    original_angular_width = (bbox_w_pixels / w_erp) * 360
    original_angular_height = (bbox_h_pixels / h_erp) * 180
    original_angular_width = max(0, min(original_angular_width, 360))
    original_angular_height = max(0, min(original_angular_height, 180))

    expanded_angular_width = original_angular_width * expansion_factor
    expanded_angular_height = original_angular_height * expansion_factor
    expanded_angular_width = min(expanded_angular_width, 360.0)
    expanded_angular_height = min(expanded_angular_height, 180.0)
    expanded_angular_width = max(expanded_angular_width, 1e-6)
    expanded_angular_height = max(expanded_angular_height, 1e-6)

    # 7. Calculate Perspective FOV and Output Dimensions
    initial_out_h, initial_out_w = out_hw
    if initial_out_h <= 0 or initial_out_w <= 0:
        print(f"Error: Initial output size hint out_hw must be positive, got {out_hw}")
        return None, None
    aspect_ratio = initial_out_w / initial_out_h if initial_out_h != 0 else float('inf')

    v_fov_needed_for_width = 180.0
    if abs(math.radians(expanded_angular_width / 2)) < math.pi / 2:
        tan_h_fov_half = math.tan(math.radians(expanded_angular_width / 2))
        if aspect_ratio > 1e-9:
            try: v_fov_needed_for_width = math.degrees(2 * math.atan(tan_h_fov_half / aspect_ratio))
            except ValueError: v_fov_needed_for_width = 180.0
    elif expanded_angular_width >= 180: v_fov_needed_for_width = 180.0

    h_fov_needed_for_height = 360.0
    if abs(math.radians(expanded_angular_height / 2)) < math.pi / 2:
        tan_v_fov_half = math.tan(math.radians(expanded_angular_height / 2))
        try: h_fov_needed_for_height = math.degrees(2 * math.atan(aspect_ratio * tan_v_fov_half))
        except ValueError: h_fov_needed_for_height = 360.0
    elif expanded_angular_height >= 180: h_fov_needed_for_height = 360.0

    if v_fov_needed_for_width >= expanded_angular_height:
        final_h_fov = expanded_angular_width
        final_v_fov = v_fov_needed_for_width
    else:
        final_v_fov = expanded_angular_height
        final_h_fov = h_fov_needed_for_height

    final_h_fov = min(final_h_fov, 125.0) # Cap HFOV
    final_h_fov = max(final_h_fov, 1e-6)

    if abs(math.radians(final_h_fov / 2)) < math.pi / 2:
        tan_final_h_fov_half = math.tan(math.radians(final_h_fov / 2))
        if aspect_ratio > 1e-9:
            try: final_v_fov = math.degrees(2 * math.atan(tan_final_h_fov_half / aspect_ratio))
            except ValueError: final_v_fov = 180.0
        else: final_v_fov = 180.0
    else: final_v_fov = 180.0
    final_v_fov = max(final_v_fov, 1e-6)
    final_v_fov = min(final_v_fov, 179.9) # Cap VFOV

    # Calculate output dims based on final FOV, max dim 1024
    max_dim = 1024
    target_h, target_w = 1, 1
    epsilon = 1e-6
    tan_h_half = math.tan(math.radians(final_h_fov / 2)) if abs(math.radians(final_h_fov / 2)) < math.pi/2 - epsilon else float('inf')
    tan_v_half = math.tan(math.radians(final_v_fov / 2)) if abs(math.radians(final_v_fov / 2)) < math.pi/2 - epsilon else float('inf')

    if tan_v_half > epsilon and tan_h_half != float('inf'):
        fov_aspect_ratio = tan_h_half / tan_v_half
        if fov_aspect_ratio >= 1:
            target_w = max_dim
            target_h = int(round(max_dim / fov_aspect_ratio))
        else:
            target_h = max_dim
            target_w = int(round(max_dim * fov_aspect_ratio))
    elif tan_h_half == float('inf'):
        target_w = max_dim
        target_h = int(round(max_dim / aspect_ratio))
    elif tan_v_half <= epsilon:
        target_h = 1
        target_w = max_dim
    else:
        target_h = max_dim
        target_w = max_dim

    target_h = max(1, target_h)
    target_w = max(1, target_w)
    final_out_hw = (target_h, target_w)

    # 8. Execute ERP to Perspective Conversion
    try:
        if final_h_fov <= 1e-6: final_h_fov = 1.0
        img_pers = e2p(
            e_img=img_erp_np, fov_deg=final_h_fov, u_deg=center_u, v_deg=center_v,
            out_hw=final_out_hw, in_rot_deg=0, mode='bilinear'
        )
    except ValueError as ve:
        print(f"Error: Invalid parameters for e2p conversion: {ve}")
        print(f"Params used: fov_deg={final_h_fov}, u_deg={center_u}, v_deg={center_v}, out_hw={final_out_hw}")
        return None, None
    except Exception as e:
        print(f"Error: Error executing e2p conversion: {e}")
        return None, None

    # 9. Return dictionary containing params and image array
    perspective_params = {
        'center_u_deg': center_u, 'center_v_deg': center_v,
        'final_hfov_deg': final_h_fov, 'final_vfov_deg': final_v_fov,
        'output_height': final_out_hw[0], 'output_width': final_out_hw[1],
        'expansion_factor_used': expansion_factor, 'vertical_offset_factor_used': vertical_offset_factor,
        'original_bbox_pixels': [x1, y1, x2, y2],
        'erp_image_height': h_erp, 'erp_image_width': w_erp
    }
    return img_pers, perspective_params


# 1. Utility/Helper Functions ----------------------------------------
# Calculate Overlap Ratio (IoU over minimum area) of two bounding boxes
def calculate_overlap(box1, box2):
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = map(float, box1) # Ensure float
    x1_2, y1_2, x2_2, y2_2 = map(float, box2) # Ensure float

    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    # If no overlap, return 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate areas of both boxes
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Calculate overlap ratio (intersection area / smaller box area)
    min_area = min(box1_area, box2_area)
    if min_area <= 1e-6: # Avoid division by zero or near-zero
        # If intersection exists but min_area is tiny, maybe return intersection/union (IoU)?
        # Or stick to definition: if min_area is 0, overlap ratio is ill-defined or 0.
        return 0.0
    overlap_ratio = intersection_area / min_area

    return overlap_ratio

# Check if boxes touch image borders
def check_boxes_touching_border(detections, image_width, image_height, border_margin=30):
    touching_indices = [] # Store indices relative to the input detections list
    for i, det in enumerate(detections):
        bbox = det['bbox'] # Assume bbox is [x1, y1, x2, y2]
        try:
            x1, y1, x2, y2 = map(float, bbox)
            # Check touching with margin
            is_touching = (x1 <= border_margin or y1 <= border_margin or
                           x2 >= image_width - border_margin or y2 >= image_height - border_margin)
            if is_touching:
                touching_indices.append(i)
        except (ValueError, TypeError):
            print(f"Warning: Skipping invalid bbox coordinates when checking borders: {bbox}")
            continue
    return touching_indices

# Fix bbox coordinates ensuring they are within image range
def clip_bbox(bbox, image_width, image_height):
    try:
        x1, y1, x2, y2 = map(float, bbox)
        x1 = max(0.0, min(float(image_width - 1), x1))
        y1 = max(0.0, min(float(image_height - 1), y1))
        x2 = max(x1 + 1e-6, min(float(image_width), x2)) # Ensure x2 > x1
        y2 = max(y1 + 1e-6, min(float(image_height), y2)) # Ensure y2 > y1
        return [x1, y1, x2, y2]
    except (ValueError, TypeError):
        print(f"Warning: Invalid coordinates encountered during bbox clipping: {bbox}. Returning original.")
        return bbox # Return original if invalid

# Apply coordinate fix to all detections (modifies in place)
def fix_detections_coordinates(detections, image_width, image_height):
    for det in detections:
        det['bbox'] = clip_bbox(det['bbox'], image_width, image_height)
    # No return needed as it modifies in place, but returning allows chaining
    return detections

# 2. Visualization/Output Functions ----------------------------------------
# (visualize_bbox remains the same as provided)
def visualize_bbox(image, detections, original_category=None):
    img_with_boxes = image.copy()

    for item in detections:
        bbox = item.get('bbox_in_resized_perspective', item.get('bbox')) # Handle output format
        if bbox is None: continue # Skip if no bbox found

        try:
            x1, y1, x2, y2 = map(int, bbox)
        except (ValueError, TypeError):
            print(f"Warning: Skipping invalid bbox coordinates during visualization: {bbox}")
            continue

        label = item.get('label', 'N/A')
        score = item.get('score', 0.0)
        level = item.get('level', '') # Get level if available

        color = (0, 255, 0)  # Default Green for carried
        line_thickness = 2
        text_prefix = f"[{level}] " if level else "" # Add level prefix if exists

        if original_category and label == original_category:
            color = (0, 0, 255)  # Red for original
            line_thickness = 3
            text_prefix = "[original] " # Override level prefix

        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, line_thickness)

        short_label = label[:20] + "..." if len(label) > 20 else label
        text = f"{text_prefix}{short_label}: {score:.2f}"
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 15
        text_x = max(0, x1)
        text_y = max(15, min(img_with_boxes.shape[0] - 5, text_y))

        cv2.putText(img_with_boxes, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img_with_boxes


# 3. Core Processing Function (Major Refactoring) -----------------------------

def process_with_perspective_projection(image_path, carries_data, dino_model, sam_model, save_dir=None, save_visualization=True):
    original_erp_image_path = image_path
    original_bbox_erp = carries_data['original_bbox'] # The initial bbox in ERP
    original_category = carries_data.get('original_category', None)
    original_description = carries_data.get('original_description', None)
    carried_items_raw = carries_data.get('carried_items', [])

    # --- 1. Prepare list of items to detect ---
    carried_items_to_detect = []
    if original_category:
        # Exclude original category name from detection list
        carried_items_to_detect = [item.get("name") for item in carried_items_raw if item.get("name") and item.get("name") != original_category]
    else:
        carried_items_to_detect = [item.get("name") for item in carried_items_raw if item.get("name")]

    # Limit number of carried items
    if len(carried_items_to_detect) > MAX_CARRIED_ITEMS:
        print(f"Warning: Carried items count {len(carried_items_to_detect)} exceeds limit {MAX_CARRIED_ITEMS}, detecting only top {MAX_CARRIED_ITEMS}.")
        carried_items_to_detect = carried_items_to_detect[:MAX_CARRIED_ITEMS]

    # Special rule: table + chair/chairs -> detect "chair"
    special_chair_added = False
    has_table = original_category == 'table' or (original_description and 'table' in original_description)
    has_chair_or_chairs = original_category in ['chair', 'chairs'] or (original_description and ('chair' in original_description or 'chairs' in original_description))

    # Check if the *input* carried list already contains chair. If not, add it under the special condition.
    input_contains_chair = any(item.get("name") == 'chair' for item in carried_items_raw)

    if has_table and has_chair_or_chairs:
         if 'chair' not in carried_items_to_detect and not input_contains_chair:
            print("Info: Detected 'table' and 'chair'/'chairs', and input list has no chair. Adding 'chair' for special detection.")
            carried_items_to_detect.append('chair')
            special_chair_added = True # Flag that 'chair' was added by this rule

    is_carried_items_list_empty = not bool(carried_items_to_detect)

    # --- 2. Prepare Save Directory ---
    if save_dir is None:
        # Fallback if save_dir isn't provided by batch processor
        base_name = os.path.splitext(os.path.basename(original_erp_image_path))[0]
        default_base_save_dir = "./group_sam/perspective_detections_mp3d" # Changed dir name
        os.makedirs(default_base_save_dir, exist_ok=True)
        # Need scene/view for a unique default subdir
        scene_id_def, view_id_def = get_scene_view_from_path(original_erp_image_path, is_image_path=True) # Use helper
        if scene_id_def and view_id_def:
             save_dir = os.path.join(default_base_save_dir, f"{scene_id_def}_{view_id_def}_{base_name}")
        else:
             save_dir = os.path.join(default_base_save_dir, base_name) # Less ideal fallback
    os.makedirs(save_dir, exist_ok=True)
    output_basename = os.path.basename(save_dir.rstrip(os.sep))
    if not output_basename: output_basename = f"task_{os.path.basename(image_path)}_{int(time.time())}"


    # --- 3. Retry Loop for Projection and Detection ---
    max_retries = 3
    retry_count = 0
    current_expansion_factor = None
    current_vertical_offset_factor = None
    final_results = None # Store the successful results dict
    final_all_detections_processed = [] # Store final list used for JSON/Vis

    while retry_count <= max_retries:
        print(f"\n--- Round {retry_count} Processing (Total {max_retries} retries) ---")
        # --- 3a. Generate/Resize Perspective Image ---
        perspective_img_np = None
        perspective_params = None
        perspective_img_resized_cv = None
        pers_h, pers_w = DINO_RESIZE_DIM[1], DINO_RESIZE_DIM[0] # Use standard size

        if retry_count == 0:
            perspective_img_np, perspective_params = erp_bbox_to_perspective(
                erp_image_path=original_erp_image_path, bbox=original_bbox_erp,
                out_hw=(OUTPUT_HEIGHT, OUTPUT_WIDTH), is_carried_items_empty=is_carried_items_list_empty
            )
            if perspective_params:
                 current_expansion_factor = perspective_params['expansion_factor_used']
                 current_vertical_offset_factor = perspective_params['vertical_offset_factor_used']
        else:
            # Increase factors for retry (adjust magnitude as needed)
            current_expansion_factor = (current_expansion_factor or 1.3) * 1.15 # Start from 1.3 if None
            current_vertical_offset_factor = (current_vertical_offset_factor or 0.0) * 1.1 # Start from 0.0 if None
            current_expansion_factor = min(current_expansion_factor, 3.0) # Add upper limit
            current_vertical_offset_factor = min(current_vertical_offset_factor, 0.8) # Add upper limit
            print(f"  Retry factors: expansion={current_expansion_factor:.3f}, vertical_offset={current_vertical_offset_factor:.3f}")

            perspective_img_np, perspective_params = erp_bbox_to_perspective(
                erp_image_path=original_erp_image_path, bbox=original_bbox_erp,
                out_hw=(OUTPUT_HEIGHT, OUTPUT_WIDTH), is_carried_items_empty=is_carried_items_list_empty,
                override_expansion_factor=current_expansion_factor,
                override_vertical_offset_factor=current_vertical_offset_factor
            )

        if perspective_img_np is None or perspective_params is None:
            print(f"Error: Failed to generate perspective image in round {retry_count}.")
            if retry_count == max_retries: # If fails on last retry, break loop
                 break
            else:
                 retry_count += 1
                 continue # Try next retry

        # Resize for detection
        try:
            perspective_img_cv = cv2.cvtColor(perspective_img_np, cv2.COLOR_RGB2BGR)
            # Use fixed DINO_RESIZE_DIM for consistency
            perspective_img_resized_cv = cv2.resize(perspective_img_cv, DINO_RESIZE_DIM)
            pers_h, pers_w = perspective_img_resized_cv.shape[:2]
            print(f"  Perspective image size (for detection): {pers_w} x {pers_h}")
        except Exception as e:
            print(f"Error: Failed to convert or resize perspective image: {e}")
            if retry_count == max_retries: break
            else: retry_count += 1; continue

        # --- 3b. Perform Detections ---
        all_detections_current_round = []
        # Detect original category
        if original_category:
            print(f"  Detecting original category: {original_category}")
            orig_detections = predict_with_grounding_dino_caption(
                dino_model, perspective_img_resized_cv, [original_category], box_threshold=0.25, text_threshold=0.25
            )
            # Add detection type info
            for det in orig_detections: det['detection_type'] = 'original'
            all_detections_current_round.extend(orig_detections)
            print(f"    Found {len(orig_detections)} original category instances.")

        # Detect carried items
        print(f"  Detecting carried items: {carried_items_to_detect}")
        for item_name in carried_items_to_detect:
            caption_to_detect = [item_name] # Default caption
            if original_category:
                # Adapt caption based on item/original category relationship
                if item_name == 'chair' and special_chair_added: # Use special relation for added chair near table
                     caption_to_detect = [f"chair next to the {original_category}"] # Assumes original is table here
                     # print(f"    Special detection relation: {caption_to_detect[0]}")
                else:
                     caption_to_detect = [f"{item_name} on the {original_category}"] # General relation
                     # print(f"    Detection relation: {caption_to_detect[0]}")

            item_detections = predict_with_grounding_dino_caption(
                dino_model, perspective_img_resized_cv, caption_to_detect,
                box_threshold=0.25, text_threshold=0.25
            )
            # IMPORTANT: Correct the label back to the item name & add type
            for det in item_detections:
                det['label'] = item_name
                det['detection_type'] = 'carried'
            all_detections_current_round.extend(item_detections)
        print(f"    Total detected items: {len(all_detections_current_round)}.")

        # --- 3c. Fix Coordinates ---
        # Fix coordinates immediately after detection for the current round
        fix_detections_coordinates(all_detections_current_round, pers_w, pers_h)

        # Separate originals and carried items for easier processing
        original_detections = [det for det in all_detections_current_round if det.get('detection_type') == 'original']
        carried_detections = [det for det in all_detections_current_round if det.get('detection_type') == 'carried']

        # --- 3d. Initial Filtering of Originals (Border, Quantity) ---
        deleted_original_indices_round = set()
        kept_original_indices_round = set()

        # Get original category quantity limit
        original_category_quantity = float('inf')
        if original_category:
            for item in carried_items_raw:
                if item.get('name') == original_category and 'quantity' in item:
                    try: original_category_quantity = max(1, int(item['quantity']))
                    except (ValueError, TypeError): original_category_quantity = float('inf')
                    break

        # Border check on original detections
        original_indices_in_all = [i for i, det in enumerate(all_detections_current_round) if det.get('detection_type') == 'original']
        original_touching_relative_indices = check_boxes_touching_border(original_detections, pers_w, pers_h, border_margin=30)

        print(f"  Check original category border touching (margin 30px)...")
        for rel_idx in original_touching_relative_indices:
            if 0 <= rel_idx < len(original_indices_in_all):
                 original_all_idx = original_indices_in_all[rel_idx]
                 deleted_original_indices_round.add(original_all_idx)
                 print(f"    - Original category (index {original_all_idx}) touching border, marked for deletion.")

        # Quantity check (apply only to non-border-touching ones first)
        non_touching_originals = []
        non_touching_original_indices_in_all = []
        for i, det in enumerate(original_detections):
             original_all_idx = original_indices_in_all[i]
             if original_all_idx not in deleted_original_indices_round:
                 non_touching_originals.append(det)
                 non_touching_original_indices_in_all.append(original_all_idx)

        if len(non_touching_originals) > original_category_quantity:
             print(f"  Applying quantity limit: {len(non_touching_originals)} > {original_category_quantity}. Sorting by confidence...")
             # Sort by score (descending) and find indices to delete
             non_touching_originals.sort(key=lambda x: x['score'], reverse=True)
             originals_to_delete_due_to_quantity = non_touching_originals[original_category_quantity:]
             # Find their original indices in all_detections
             scores_to_delete = {det['score'] for det in originals_to_delete_due_to_quantity} # Use score as identifier (can be fragile)

             # More robust: map score/bbox back to original index
             num_deleted_quantity = 0
             for i, original_all_idx in enumerate(non_touching_original_indices_in_all):
                 # Find the corresponding detection in non_touching_originals
                 # This assumes scores are unique enough or order is preserved
                 # Let's re-sort the indices list based on score of corresponding detection
                 indexed_non_touching = sorted(
                     zip(non_touching_original_indices_in_all, non_touching_originals),
                     key=lambda pair: pair[1]['score'],
                     reverse=True
                 )
                 # The indices to delete are from original_category_quantity onwards in this sorted list
                 if i >= original_category_quantity:
                      idx_to_delete = indexed_non_touching[i][0]
                      if idx_to_delete not in deleted_original_indices_round: # Avoid double-adding
                           deleted_original_indices_round.add(idx_to_delete)
                           print(f"    - Original category (index {idx_to_delete}) marked for deletion due to exceeding quantity limit.")
                           num_deleted_quantity += 1


        # Determine kept original indices for this round
        for i in original_indices_in_all:
            if i not in deleted_original_indices_round:
                kept_original_indices_round.add(i)

        print(f"  After initial filter: {len(kept_original_indices_round)} original categories kept, {len(deleted_original_indices_round)} marked for deletion.")

        # --- 3e. Associate and Filter Carried Items ---
        kept_carried_indices_round = set()
        discarded_carried_indices_round = set()
        carried_indices_in_all = [i for i, det in enumerate(all_detections_current_round) if det.get('detection_type') == 'carried']
        kept_original_bboxes = [all_detections_current_round[i]['bbox'] for i in kept_original_indices_round]

        print(f"  Associating and filtering carried items...")
        for carried_idx_in_all in carried_indices_in_all:
            carried_item = all_detections_current_round[carried_idx_in_all]
            carried_bbox = carried_item['bbox']
            best_overlap = -1.0
            primary_assoc_original_idx = -1

            # Find best overlapping original bbox
            for original_idx_in_all in original_indices_in_all: # Check against ALL originals initially
                original_bbox = all_detections_current_round[original_idx_in_all]['bbox']
                overlap = calculate_overlap(carried_bbox, original_bbox)
                if overlap > best_overlap:
                    best_overlap = overlap
                    primary_assoc_original_idx = original_idx_in_all

            # Filter based on association and overlap thresholds
            if best_overlap >= ASSOCIATION_OVERLAP_THRESHOLD:
                carried_item['associated_original_index'] = primary_assoc_original_idx # Store association
                carried_item['association_overlap'] = best_overlap

                if primary_assoc_original_idx in kept_original_indices_round:
                    # Associated with a kept original - keep initially
                    kept_carried_indices_round.add(carried_idx_in_all)
                    # print(f"    - Carried item (idx {carried_idx_in_all}, {carried_item['label']}) associated with kept original ({primary_assoc_original_idx}), kept.")
                elif primary_assoc_original_idx in deleted_original_indices_round:
                    # Associated with a deleted original - check overlap with KEPT ones
                    overlaps_with_kept = False
                    for kept_original_bbox in kept_original_bboxes:
                         if calculate_overlap(carried_bbox, kept_original_bbox) >= SURVIVAL_OVERLAP_THRESHOLD:
                             overlaps_with_kept = True
                             break
                    if overlaps_with_kept:
                        kept_carried_indices_round.add(carried_idx_in_all)
                        print(f"    - Carried item (idx {carried_idx_in_all}, {carried_item['label']}) associated with deleted original ({primary_assoc_original_idx}), but has sufficient overlap (>=10%) with a kept original, kept.")
                    else:
                        discarded_carried_indices_round.add(carried_idx_in_all)
                        print(f"    - Carried item (idx {carried_idx_in_all}, {carried_item['label']}) associated with deleted original ({primary_assoc_original_idx}), and insufficient overlap with kept originals, deleted.")
                else:
                     # Should not happen if sets cover all original indices, but handle defensively
                     discarded_carried_indices_round.add(carried_idx_in_all)
                     print(f"    - Warning: Carried item (idx {carried_idx_in_all}) associated original idx ({primary_assoc_original_idx}) is neither kept nor deleted? Deleted.")

            else:
                # Not significantly overlapping with ANY original - discard
                discarded_carried_indices_round.add(carried_idx_in_all)
                # print(f"    - Carried item (idx {carried_idx_in_all}, {carried_item['label']}) overlaps < {ASSOCIATION_OVERLAP_THRESHOLD:.1f} with any original, deleted.")

        print(f"  After association filter: {len(kept_carried_indices_round)} carried items kept, {len(discarded_carried_indices_round)} deleted.")


        # --- 3f. Apply Special Chair Rule (adapted) ---
        final_kept_indices_round = kept_original_indices_round.union(kept_carried_indices_round)
        if special_chair_added:
            print(f"  Applying special Chair rule...")
            indices_to_remove_chair = set()
            # Check overlap of kept chairs with kept tables
            kept_table_bboxes = [all_detections_current_round[i]['bbox'] for i in kept_original_indices_round if all_detections_current_round[i]['label'] == 'table']

            if not kept_table_bboxes:
                 # If no tables were kept, remove all specially added chairs
                 print("    - No kept tables, removing all specially added chairs.")
                 for idx in list(kept_carried_indices_round): # Iterate over copy
                     if all_detections_current_round[idx]['label'] == 'chair':
                         indices_to_remove_chair.add(idx)
            else:
                 # Check overlap of kept chairs with kept tables
                 for idx in list(kept_carried_indices_round):
                      if all_detections_current_round[idx]['label'] == 'chair':
                           chair_bbox = all_detections_current_round[idx]['bbox']
                           overlaps_with_kept_table = False
                           for table_bbox in kept_table_bboxes:
                               # Use a higher threshold for chair rule (e.g., 0.3)
                               if calculate_overlap(chair_bbox, table_bbox) >= 0.3:
                                    overlaps_with_kept_table = True
                                    break
                           if not overlaps_with_kept_table:
                                indices_to_remove_chair.add(idx)
                                print(f"    - Specially added Chair (idx {idx}) overlap with any kept table < 0.3, removing.")

            # Remove chairs identified by the rule
            kept_carried_indices_round.difference_update(indices_to_remove_chair)
            final_kept_indices_round = kept_original_indices_round.union(kept_carried_indices_round)
            print(f"  Carried items kept after special Chair rule: {len(kept_carried_indices_round)}.")

        # --- 3g. Check Carried Items Border Touching & Decide Retry ---
        # Check if any *kept* carried items touch the border
        kept_carried_detections = [all_detections_current_round[i] for i in kept_carried_indices_round]
        carried_touching_relative_indices = check_boxes_touching_border(kept_carried_detections, pers_w, pers_h, border_margin=30)

        if not carried_touching_relative_indices:
            print(f"Success: No kept carried items touch the border in round {retry_count}. Using this result.")
            # Success condition met! Prepare final results.
            final_kept_indices = final_kept_indices_round

            # --- 3h. Final Check: Original Category Existence & Area ---
            final_kept_originals = [all_detections_current_round[i] for i in final_kept_indices if i in kept_original_indices_round]

            if original_category and not final_kept_originals:
                print(f"Error: Final result contains no valid original category '{original_category}'. Task failed.")
                # Clean up and signal failure (break without setting final_results)
                if os.path.exists(save_dir) and is_dir_empty(save_dir):
                    try:
                         if not os.listdir(save_dir): os.rmdir(save_dir)
                    except OSError: pass
                final_results = None # Explicitly mark as failed
                break # Exit retry loop

            # Check area ratio
            perspective_area = float(pers_w * pers_h)
            total_original_area = 0.0
            if perspective_area > 0 and final_kept_originals:
                 for det in final_kept_originals:
                      bbox = det["bbox"] # Already fixed coords
                      x1, y1, x2, y2 = bbox
                      total_original_area += (x2 - x1) * (y2 - y1)
                 total_original_area_ratio = total_original_area / perspective_area
            else:
                 total_original_area_ratio = 0.0 # Cannot calculate or no originals

            print(f"  Checking final kept original categories ({len(final_kept_originals)} items) total area ratio: {total_original_area_ratio:.4f} (Threshold: {MIN_ORIGINAL_BBOX_AREA_RATIO:.3f})")
            if original_category and total_original_area_ratio < MIN_ORIGINAL_BBOX_AREA_RATIO:
                 print(f"Error: Final kept original category total area ratio insufficient. Task failed.")
                 if os.path.exists(save_dir) and is_dir_empty(save_dir):
                     try:
                         if not os.listdir(save_dir): os.rmdir(save_dir)
                     except OSError: pass
                 final_results = None # Explicitly mark as failed
                 break # Exit retry loop


            # --- All checks passed for this round! Format and store result ---
            print("  All checks passed. Formatting final results...")
            final_all_detections_processed = []
            for idx in sorted(list(final_kept_indices)): # Sort for consistent output order
                det = all_detections_current_round[idx]
                level = "carried" if det['detection_type'] == 'carried' else "original"
                detection_info = {
                    "label": det["label"],
                    "score": float(det["score"]),
                    "bbox_in_resized_perspective": [float(coord) for coord in det["bbox"]], # Use fixed coords
                    "level": level,
                    # Optionally add association info for debugging/analysis
                    # "associated_original_index": det.get("associated_original_index", -1),
                    # "association_overlap": det.get("association_overlap", 0.0)
                }
                final_all_detections_processed.append(detection_info)

            result_image_path_rel = f"detection_result_on_perspective.jpg" # Relative path for JSON
            results = {
                "original_erp_image": original_erp_image_path,
                "perspective_transform_params": perspective_params,
                "detection_image_size": {"width": pers_w, "height": pers_h},
                "original_bbox_erp": [float(coord) for coord in original_bbox_erp],
                "original_category": original_category,
                "original_description": original_description,
                "carried_items_input": carried_items_raw, # Keep original input list
                "detections": final_all_detections_processed,
                "result_image_path": os.path.join(save_dir, result_image_path_rel), # Full path for saving image
                "processing_retries": retry_count,
                "json_output_path": os.path.join(save_dir, f"{output_basename}_detection.json")
            }
            final_results = results # Store the successful result dict
            # Save visualization if needed (using the successful round's image)
            if save_visualization:
                 try:
                     # Use the latest perspective_img_resized_cv from this successful round
                     img_to_visualize = perspective_img_resized_cv
                     # Create list compatible with visualize_bbox format
                     vis_detections = final_all_detections_processed # Already in correct format
                     img_with_boxes = visualize_bbox(img_to_visualize, vis_detections, original_category)
                     cv2.imwrite(results["result_image_path"], img_with_boxes)
                     print(f"  Detection result image saved to: {results['result_image_path']}")
                 except Exception as e:
                     print(f"Error: Failed to save visualization image: {e}")
                     results["result_image_path"] = None # Mark as failed

            break # Exit retry loop successfully

        elif retry_count == max_retries:
            print(f"Warning: Reached maximum retries but still have kept carried items touching borders. Abandoning task.")
            # Clean up and break loop (failure)
            if os.path.exists(save_dir) and is_dir_empty(save_dir):
                 try:
                      if not os.listdir(save_dir): os.rmdir(save_dir)
                 except OSError: pass
            final_results = None # Failed
            break
        else:
            # Need to retry
            print(f"Info: Kept carried items are touching borders, retrying (Next round: {retry_count + 1}).")
            retry_count += 1
            # Loop continues

    # --- End of Retry Loop ---

    if final_results:
        # Save the final JSON result if processing was successful
        try:
            # Update JSON path to relative one for consistency within the dict
            final_results["json_output_path_relative"] = f"{output_basename}_detection.json"
            json_save_path_full = final_results["json_output_path"] # Keep full path for saving

            with open(json_save_path_full, 'w') as f:
                json.dump(final_results, f, indent=4)
            print(f"Final detection JSON saved to: {json_save_path_full}")
            # Return the results dict and the list used for JSON
            return final_results, final_all_detections_processed, None # Vis img already saved if needed
        except Exception as e:
            print(f"Error: Failed to save final JSON file {final_results['json_output_path']}: {e}")
            # Clean up visualization image if JSON saving failed
            if final_results.get("result_image_path") and os.path.exists(final_results["result_image_path"]):
                 try: os.remove(final_results["result_image_path"])
                 except OSError: pass
            if os.path.exists(save_dir) and is_dir_empty(save_dir):
                 try:
                     if not os.listdir(save_dir): os.rmdir(save_dir)
                 except OSError: pass
            return None, [], None # Return failure
    else:
        # Processing failed or was abandoned
        print(f"Task failed or abandoned for {os.path.basename(image_path)}.")
        # Ensure directory is cleaned up if empty
        if os.path.exists(save_dir) and is_dir_empty(save_dir):
            try:
                 if not os.listdir(save_dir): os.rmdir(save_dir)
                 print(f"  Deleted empty output directory: {save_dir}")
            except OSError as e:
                 print(f"  Error deleting empty directory: {e}")
        return None, [], None # Return failure


# --- Helper: Check if directory is empty ---
def is_dir_empty(dir_path):
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        return True
    return not os.listdir(dir_path) # Simpler check: empty if listdir returns empty list

# --- Helper: Extract Scene/View from path (Matterport3D specific) ---
def get_scene_view_from_path(file_path, is_image_path=False):
    """
    Extracts scene_id and view_id from a Matterport3D file path.
    Handles both JSON paths and potentially image paths.
    Expected JSON structure: .../mp3d_base/[scene]/matterport_stitched_images/[view]/carries_results/[filename].json
    Expected Image structure: .../mp3d_skybox/[scene]/matterport_stitched_images/[view].png
    """
    try:
        parts = file_path.replace('\\', '/').split('/')
        scene_id = None
        view_id = None

        if is_image_path:
            # Structure: .../mp3d_skybox/[scene]/matterport_stitched_images/[view].png
            # Indices from end: -1=[view].png, -2=matterport_stitched_images, -3=[scene], -4=mp3d_skybox
            if len(parts) >= 4:
                scene_id = parts[-3]
                view_id_with_ext = parts[-1]
                view_id = os.path.splitext(view_id_with_ext)[0] # Remove extension
            else:
                 print(f"Warning: Image path '{file_path}' not long enough to extract scene_id/view_id by expected Matterport3D structure (needs >= 4 parts).")
                 return None, None
        else: # Assume JSON path
            # Structure: .../mp3d_inpaint_all/[scene]/matterport_stitched_images/[view]/carries_results/[filename].json
            # Indices from end: -1=filename, -2=carries_results, -3=[view], -4=matterport_stitched_images, -5=[scene], -6=mp3d_inpaint_all
            if len(parts) >= 6:
                view_id = parts[-3]
                scene_id = parts[-5]
            else:
                 print(f"Warning: JSON path '{file_path}' not long enough to extract scene_id/view_id by expected Matterport3D structure (needs >= 6 parts).")
                 return None, None

        if scene_id and view_id: # Basic validation
            return scene_id, view_id
        else:
            print(f"Warning: Could not extract valid scene_id/view_id from expected parts of '{file_path}'. Scene: '{scene_id}', View: '{view_id}'")
            return None, None

    except Exception as e:
        print(f"Warning: Error extracting scene_id/view_id from path '{file_path}': {e}")
        return None, None


# --- Batch Processing Function (Sequential) ---
# (batch_process_sequential remains largely the same, calling the modified process_with_perspective_projection)
def batch_process_sequential(carries_results_dir, image_base_dir, regenerate_existing, dino_model, sam_model, save_visualization=True):
    """
    Scans directories, prepares tasks, and processes carries.json files sequentially in the main process (Matterport3D version).
    """
    print(f"Start scanning task directory: {carries_results_dir}")
    tasks_to_process = []
    skipped_count = 0; error_count = 0; missing_image_count = 0; missing_field_count = 0; load_error_count = 0

    search_pattern = os.path.join(carries_results_dir, "**", "*_carries.json")
    all_carries_files = glob.glob(search_pattern, recursive=True)
    print(f"Found {len(all_carries_files)} potential _carries.json files.")

    if not all_carries_files:
        print("Error: No '_carries.json' files found in specified directory. Please check 'carries_results_dir'.")
        return

    print("Preparing tasks...")
    for carries_file_path in tqdm(all_carries_files, desc="Scanning carries files"):
        try:
            # Extract scene/view using the updated helper for MP3D JSON path
            scene_id, view_id = get_scene_view_from_path(carries_file_path, is_image_path=False)
            if not scene_id or not view_id:
                error_count += 1
                continue

            # Construct image path for Matterport3D Skybox
            # /home/lab401/zhongfile/Project/PanFusion/data/Matterport3D/mp3d_skybox/[scene]/matterport_stitched_images/[view].png
            original_image_path = os.path.join(image_base_dir, scene_id, "matterport_stitched_images", f"{view_id}.png")

            if not os.path.exists(original_image_path):
                # Fallback check for jpg
                original_image_path_jpg = original_image_path.replace(".png", ".jpg")
                if os.path.exists(original_image_path_jpg):
                    original_image_path = original_image_path_jpg
                else:
                    print(f"Skipping: Original image {original_image_path} or .jpg version not found (for {carries_file_path})")
                    missing_image_count += 1
                    continue

            file_basename = os.path.basename(carries_file_path).replace("_carries.json", "")
            # Define save_dir relative to the carries.json location
            # e.g., .../carries_results/[view]_obj[n]_carries.json -> .../carries_results/[view]_obj[n]/
            save_dir = os.path.join(os.path.dirname(carries_file_path), file_basename)
            # Define expected output JSON path based on basename of save_dir
            output_basename_for_json = os.path.basename(save_dir.rstrip(os.sep))
            if not output_basename_for_json: output_basename_for_json = file_basename # Fallback
            detection_json_path = os.path.join(save_dir, f"{output_basename_for_json}_detection.json")

            # Check if result exists and regeneration is not forced
            if not regenerate_existing and os.path.exists(detection_json_path):
                skipped_count += 1
                continue

            # Load carries data *after* checking for existing output
            try:
                with open(carries_file_path, 'r', encoding='utf-8') as f:
                    carries_data = json.load(f)
                if not isinstance(carries_data, dict) or \
                   "original_bbox" not in carries_data or \
                   not isinstance(carries_data["original_bbox"], list) or \
                   len(carries_data["original_bbox"]) != 4:
                    print(f"Skipping: File {carries_file_path} missing valid 'original_bbox' field.")
                    missing_field_count += 1
                    # No directory creation attempt before this point, so no cleanup needed here
                    continue
            except json.JSONDecodeError as e:
                print(f"Skipping: Error parsing JSON {carries_file_path}: {e}")
                load_error_count += 1
                continue
            except Exception as e:
                print(f"Skipping: Unexpected error loading carries data {carries_file_path}: {e}")
                load_error_count += 1
                continue

            # Add task to list (directory creation happens inside process func now)
            tasks_to_process.append({
                "carries_file": carries_file_path,
                "carries_data": carries_data,
                "original_image_path": original_image_path,
                "save_dir": save_dir, # Pass intended save dir
                "scene_id": scene_id,
                "view_id": view_id
            })

        except Exception as e:
            print(f"Unexpected error preparing task {carries_file_path}: {e}")
            error_count += 1
            import traceback
            traceback.print_exc()

    total_tasks = len(tasks_to_process)
    print(f"\nTask preparation complete:")
    print(f"  - Total files found: {len(all_carries_files)}")
    print(f"  - Skipped due to parse/path errors: {error_count}")
    print(f"  - Skipped due to missing image: {missing_image_count}")
    print(f"  - Skipped due to missing field: {missing_field_count}")
    print(f"  - Skipped due to load/JSON errors: {load_error_count}")
    print(f"  - Skipped as results exist (regenerate=False): {skipped_count}")
    print(f"  - Tasks to process: {total_tasks}")

    if total_tasks == 0:
        print("No tasks to process.")
        return

    # --- Use loop for sequential processing ---
    processed_count = 0
    failed_count = 0
    print(f"\nStarting sequential processing of {total_tasks} tasks...")
    print(f"Save visualization results: {save_visualization}")

    if tasks_to_process:
        for task_info in tqdm(tasks_to_process, desc="Processing tasks"):
            image_path = task_info['original_image_path']
            carries_data = task_info['carries_data']
            save_dir = task_info['save_dir'] # Intended save dir

            try:
                # Call core processing function (V2)
                # It now handles its own saving and returns dict/list or None/[]
                results_dict, final_detections_list, _ = process_with_perspective_projection(
                    image_path=image_path,
                    carries_data=carries_data,
                    dino_model=dino_model,
                    sam_model=sam_model, # Pass SAM model even if unused currently
                    save_dir=save_dir,   # Pass intended save dir
                    save_visualization=save_visualization
                )

                if results_dict is not None: # Check if the main process function indicated success
                    processed_count += 1
                else:
                    # Failure is handled inside process_with_perspective_projection (including cleanup)
                    failed_count += 1

            except Exception as e:
                print(f"\nUnexpected top-level error processing file {os.path.basename(image_path)}: {e}")
                import traceback
                traceback.print_exc()
                failed_count += 1
                # Attempt cleanup of the save_dir if it exists and is empty, as the process func might have crashed before its own cleanup
                try:
                    if os.path.exists(save_dir) and is_dir_empty(save_dir):
                         if not os.listdir(save_dir): os.rmdir(save_dir)
                except Exception as cleanup_e:
                     print(f"  Error cleaning up directory {save_dir}: {cleanup_e}")


    print(f"\nBatch processing complete. Success: {processed_count}, Failed: {failed_count}")


# --- Main Program ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Matterport3D data and detect carried relationships V2 (Sequential, New Association Logic)") # Updated description
    # Updated default paths
    parser.add_argument('--carries_dir', type=str,
                        default="./data/Matterport3D/mp3d_base/",
                        help='Root directory containing scene subdirectories for Matterport3D carries JSON files.')
    parser.add_argument('--image_dir', type=str,
                        default="./data/Matterport3D/mp3d_skybox/",
                        help='Root directory containing the original Matterport3D skybox ERP images.')
    parser.add_argument('--regenerate', action='store_true', help='Regenerate results even if output JSON already exists.')
    parser.add_argument('--save_visualization', default=True, action='store_true', help='Save the visualization image with bounding boxes.')

    args = parser.parse_args()

    # --- Model Initialization ---
    print("Initializing Grounding DINO and SAM models...")
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"
    print(f"Using device: {device}")

    try:
        grounding_dino_model, sam_predictor = initialize_ground_sam_models(device=device)
        print("Model initialization complete.")
    except Exception as e:
        print(f"Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- Batch Processing Mode (Sequential) ---
    carries_results_dir = args.carries_dir
    image_base_dir = args.image_dir
    regenerate_existing = args.regenerate
    save_visualization_flag = args.save_visualization

    print(f"\nStarting sequential batch processing (Matterport3D - V2 Logic)") # Updated print statement
    print(f"  Carries JSON Root Directory: {carries_results_dir}")
    print(f"  Original Image Root Directory: {image_base_dir}")
    print(f"  Regenerate existing results: {regenerate_existing}")
    print(f"  Save visualization images: {save_visualization_flag}")

    if not os.path.isdir(carries_results_dir):
        print(f"Error: carries_dir '{carries_results_dir}' is not a valid directory.")
        sys.exit(1)
    if not os.path.isdir(image_base_dir):
        print(f"Error: image_dir '{image_base_dir}' is not a valid directory.")
        sys.exit(1)

    start_time = time.time()
    batch_process_sequential(
        carries_results_dir,
        image_base_dir,
        regenerate_existing,
        grounding_dino_model,
        sam_predictor,
        save_visualization=save_visualization_flag
    )
    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")