from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image, ImageDraw, ImageFont
import json
import re
import os
import sys
import numpy as np
import argparse
import glob
import time
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# Import erp_bbox_to_perspective function
# from bbox2pers_all import erp_bbox_to_perspective # REMOVED

# Add necessary imports
import math
from py360.e2p import e2p

# Set unified output resolution (now used as out_hw hint for erp_bbox_to_perspective)
OUTPUT_WIDTH = 1024    # Output width hint
OUTPUT_HEIGHT = 1024  # Output height hint

# Base data path
base_data_path = "./data/Matterport3D/mp3d_base/"
# Add image base path
image_base_path = "./data/Matterport3D/mp3d_skybox/"

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-32B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    cache_dir=None,
)

# default processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct", use_fast=True, cache_dir=None)
# Set padding_side to 'left' to avoid errors during batch generation
processor.tokenizer.padding_side = 'left'

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# --- Add erp_bbox_to_perspective function directly here ---
def erp_bbox_to_perspective(json_path, output_path, detection_index=0, out_hw=(1024, 1024)):
    """
    Loads a 360 ERP image, expands the specified bbox by expansion_factor (limited
    by image boundaries), projects the expanded region to a perspective view
    with FOV calculated to tightly fit the expanded box (FOV capped at 125 deg).
    The output image dimensions are calculated to maintain the aspect ratio
    implied by the final FOVs, with the longest side set to a maximum value (e.g., 1024).

    Args:
        json_path (str): Path to the JSON file.
        output_path (str): Path to save the output perspective image.
        detection_index (int): Index of the detection/bbox to use from the JSON file.
        out_hw (tuple): Initial output height and width hint (h, w) used for
                        intermediate aspect ratio calculation. The final output
                        dimensions will be recalculated based on final FOV.

    Returns:
        dict or None: A dictionary containing perspective parameters if successful,
                      otherwise None.
    """
    # 1. Load JSON file
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found: {json_path}")
        return None # Return None to indicate failure
    except json.JSONDecodeError:
        print(f"Error: Unable to parse JSON file: {json_path}")
        return None # Return None to indicate failure

    image_path = data.get("image_path")
    detections = data.get("detections")

    if not image_path or not detections:
        print("Error: JSON file missing 'image_path' or 'detections' fields.")
        return None # Return None to indicate failure

    if detection_index >= len(detections):
        print(f"Error: detection_index {detection_index} out of range (total {len(detections)} detections).")
        return None # Return None to indicate failure

    # 2. Load ERP image
    try:
        img_erp = Image.open(image_path)
        img_erp_np = np.array(img_erp)
        h_erp, w_erp = img_erp_np.shape[:2]
    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path}")
        return None # Return None to indicate failure
    except Exception as e:
        print(f"Error: Error loading image {image_path}: {e}")
        return None # Return None to indicate failure

    # 3. Get selected bbox
    bbox = detections[detection_index]["bbox"]
    xmin, ymin, xmax, ymax = bbox

    # 4. Calculate bbox center point, dimensions, and relative size
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    bbox_h_pixels = ymax - ymin # Calculate bbox height
    bbox_w_pixels = xmax - xmin # Calculate bbox width

    # *** Dynamically adjust center point cy offset factor and select expansion factor based on bbox max relative size ***
    if h_erp <= 0 or w_erp <= 0:
        print("Warning: Image height or width is zero or negative, cannot calculate ratio, using default offset factor 0.3 and default expansion factor 1.3.")
        vertical_offset_factor = 0.3 # Set default value
        expansion_factor = 1.3 # Set default expansion factor
        ratio_h = 0
        ratio_w = 0
        max_ratio = 0
    else:
        ratio_h = bbox_h_pixels / h_erp # Height ratio
        ratio_w = bbox_w_pixels / h_erp # Width ratio # Note this is w_erp
        max_ratio = max(ratio_h, ratio_w) # Get max ratio

        # --- Calculate vertical offset factor (still based on height ratio ratio_h) ---
        if ratio_h > 0.5:
            vertical_offset_factor = 0.05
            # print(f"Info: BBox height ratio ({ratio_h:.2f}) > 1/2, using offset factor 0.05.")
        elif ratio_h > 2/5:
            vertical_offset_factor = 0.1
            # print(f"Info: BBox height ratio ({ratio_h:.2f}) in (2/5, 1/2], using offset factor 0.1.")
        elif ratio_h > 1/3:
            vertical_offset_factor = 0.15
            # print(f"Info: BBox height ratio ({ratio_h:.2f}) in (1/3, 2/5], using offset factor 0.15.")
        elif ratio_h > 1/4:
            vertical_offset_factor = 0.2
            # print(f"Info: BBox height ratio ({ratio_h:.2f}) in (1/4, 1/3], using offset factor 0.2.")
        elif ratio_h > 1/5:
            vertical_offset_factor = 0.3
            # print(f"Info: BBox height ratio ({ratio_h:.2f}) in (1/5, 1/4], using offset factor 0.3.")
        elif ratio_h > 1/6:
            vertical_offset_factor = 0.35
            # print(f"Info: BBox height ratio ({ratio_h:.2f}) in (1/6, 1/5], using offset factor 0.35.")
        else:
            vertical_offset_factor = 0.4
            # print(f"Info: BBox height ratio ({ratio_h:.2f}) <= 1/6, using offset factor 0.4.")

        # *** Dynamically select expansion_factor based on bbox max relative size (max_ratio) ***
        factor_huge_bbox = 1.2  # for max_ratio > 0.5
        factor_large_bbox = 1.3  # for max_ratio > 2/5
        factor_medium_large_bbox = 1.4 # for max_ratio > 1/3
        factor_medium_bbox = 1.5 # for max_ratio > 1/4
        factor_medium_small_bbox = 1.6 # for max_ratio > 1/5
        factor_small_bbox = 1.7  # for max_ratio > 1/6 # Modified to match previous logic
        factor_tiny_bbox = 1.8  # for max_ratio <= 1/6 # Modified to match previous logic

        # Select corresponding factor based on max_ratio
        if max_ratio > 0.5:
            expansion_factor = factor_huge_bbox
        elif max_ratio > 2/5:
            expansion_factor = factor_large_bbox
        elif max_ratio > 1/3:
            expansion_factor = factor_medium_large_bbox
        elif max_ratio > 1/4:
            expansion_factor = factor_medium_bbox
        elif max_ratio > 1/5:
            expansion_factor = factor_medium_small_bbox
        elif max_ratio > 1/6: # Modified to match previous logic
            expansion_factor = factor_small_bbox
        else: # max_ratio <= 1/6
            expansion_factor = factor_tiny_bbox
        # Update print info, stating it is based on max ratio
        # print(f"Info: BBox relative height {ratio_h:.2f}, relative width {ratio_w:.2f}. Max relative size {max_ratio:.2f}.")
        # print(f"Info: Dynamically selected expansion factor based on max relative size ({max_ratio:.2f}) is: {expansion_factor:.2f}")
        # *** Selection logic ends ***

    vertical_offset = bbox_h_pixels * vertical_offset_factor
    adjusted_cy = cy - vertical_offset
    adjusted_cy = max(0, min(adjusted_cy, h_erp - 1))
    # print(f"Original center cy: {cy:.2f}, Adjusted cy: {adjusted_cy:.2f} (Moved up {vertical_offset:.2f} pixels)")

    # 5. Convert *adjusted* center point pixel coordinates to spherical coordinates (angles)
    center_u = (cx / w_erp - 0.5) * 360
    # Use adjusted_cy to calculate center_v
    center_v = -(adjusted_cy / h_erp - 0.5) * 180

    # 6. Estimate the *original* angular size of the bbox
    # bbox_w_pixels and bbox_h_pixels have been calculated above

    if w_erp == 0 or h_erp == 0:
        print("Error: ERP image width or height is zero.")
        return None # Return None to indicate failure

    original_angular_width = (bbox_w_pixels / w_erp) * 360
    original_angular_height = (bbox_h_pixels / h_erp) * 180

    original_angular_width = max(0, min(original_angular_width, 360))
    original_angular_height = max(0, min(original_angular_height, 180))

    # *** Expand angular size using the dynamically selected expansion_factor ***
    expanded_angular_width = original_angular_width * expansion_factor
    expanded_angular_height = original_angular_height * expansion_factor

    # *** Limit expanded angular dimensions ***
    expanded_angular_width = min(expanded_angular_width, 360.0) # Horizontal not exceeding 360
    expanded_angular_height = min(expanded_angular_height, 180.0) # Vertical not exceeding 180

    # Ensure expanded angular dimensions are at least a tiny positive number
    expanded_angular_width = max(expanded_angular_width, 1e-6)
    expanded_angular_height = max(expanded_angular_height, 1e-6)

    # 7. Calculate FOV for perspective view to tightly fit the expanded BBox
    # Calculate initial aspect_ratio using the passed out_hw
    initial_out_h, initial_out_w = out_hw
    if initial_out_h <= 0 or initial_out_w <= 0:
        print(f"Error: Initial output size hint out_hw must be positive, current is {out_hw}")
        return None # Return None to indicate failure
    aspect_ratio = initial_out_w / initial_out_h if initial_out_h != 0 else float('inf')

    # --- FOV Calculation Logic ---
    v_fov_needed_for_width = 180.0
    if abs(math.radians(expanded_angular_width / 2)) < math.pi / 2:
        tan_h_fov_half = math.tan(math.radians(expanded_angular_width / 2))
        if aspect_ratio > 1e-9:
            try:
                v_fov_needed_for_width = math.degrees(2 * math.atan(tan_h_fov_half / aspect_ratio))
            except ValueError:
                # print(f"Warning: atan parameter out of range when calculating v_fov_needed_for_width")
                v_fov_needed_for_width = 180.0
    elif expanded_angular_width >= 180:
         v_fov_needed_for_width = 180.0

    h_fov_needed_for_height = 360.0
    if abs(math.radians(expanded_angular_height / 2)) < math.pi / 2:
        tan_v_fov_half = math.tan(math.radians(expanded_angular_height / 2))
        try:
            h_fov_needed_for_height = math.degrees(2 * math.atan(aspect_ratio * tan_v_fov_half))
        except ValueError:
            # print(f"Warning: atan parameter out of range when calculating h_fov_needed_for_height")
            h_fov_needed_for_height = 360.0
    elif expanded_angular_height >= 180:
        h_fov_needed_for_height = 360.0

    # Determine final FOV
    if v_fov_needed_for_width >= expanded_angular_height:
        final_h_fov = expanded_angular_width
        final_v_fov = v_fov_needed_for_width
        # print("Info: FOV determined by expanded width.")
    else:
        final_v_fov = expanded_angular_height
        final_h_fov = h_fov_needed_for_height
        # print("Info: FOV determined by expanded height.")

    # *** Limit final HFOV to not exceed 125 degrees ***
    final_h_fov = min(final_h_fov, 125.0)

    # Ensure FOV does not become non-positive due to calculation errors
    final_h_fov = max(final_h_fov, 1e-6)
    # Recalculate final_v_fov to match restricted final_h_fov and initial aspect_ratio
    if abs(math.radians(final_h_fov / 2)) < math.pi / 2:
        tan_final_h_fov_half = math.tan(math.radians(final_h_fov / 2))
        if aspect_ratio > 1e-9:
            try:
                final_v_fov = math.degrees(2 * math.atan(tan_final_h_fov_half / aspect_ratio))
            except ValueError:
                 final_v_fov = 180.0
        else:
            final_v_fov = 180.0
    else:
        final_v_fov = 180.0

    final_v_fov = max(final_v_fov, 1e-6) # Ensure v_fov is also positive
    final_v_fov = min(final_v_fov, 179.9) # VFOV should also be less than 180

    # *** New: Calculate output dimensions based on final FOV, longest side is 1024 ***
    max_dim = 1024
    target_h, target_w = 1, 1 # Default
    epsilon = 1e-6
    h_fov_rad_half = math.radians(final_h_fov / 2)
    v_fov_rad_half = math.radians(final_v_fov / 2)


    target_h = max_dim
    target_w = max_dim

    final_out_hw = (target_h, target_w)


    # print(f"Image size (H, W): ({h_erp}, {w_erp})")
    # print(f"BBox: {bbox}")
    # print(f"Original Center (pixels): ({cx:.2f}, {cy:.2f})")
    # print(f"Adjusted Center (pixels): ({cx:.2f}, {adjusted_cy:.2f})") # Print adjusted center
    # print(f"Center (angles): u={center_u:.2f}, v={center_v:.2f}")
    # print(f"BBox Original Angular Size (H, W): ({original_angular_height:.2f}, {original_angular_width:.2f})")
    # print(f"BBox Expansion Factor: {expansion_factor}")
    # print(f"Restricted Expanded Angular Size (H, W): ({expanded_angular_height:.2f}, {expanded_angular_width:.2f})")
    # print(f"Initial Aspect Ratio Hint (from out_hw): {aspect_ratio:.2f}")
    # print(f"Calculated Final HFOV: {final_h_fov:.2f} (Upper Limit 125.0), VFOV: {final_v_fov:.2f}")
    # print(f"Output Size calculated from Final FOV (H, W): {final_out_hw}") # Print final size

    # 8. Perform ERP to perspective conversion
    try:
        if final_h_fov <= 1e-6:
             print(f"Warning: Final HFOV ({final_h_fov:.2f}) is too small or non-positive, using small default 1.0")
             final_h_fov = 1.0

        img_pers = e2p(
            e_img=img_erp_np,
            fov_deg=final_h_fov,
            u_deg=center_u,
            v_deg=center_v,
            out_hw=final_out_hw, # *** Use calculated final dimensions ***
            in_rot_deg=0,
            mode='bilinear'
        )
    except ValueError as ve:
         print(f"Error: Invalid arguments during e2p conversion: {ve}")
         print(f"Arguments used: fov_deg={final_h_fov}, u_deg={center_u}, v_deg={center_v}, out_hw={final_out_hw}") # Print used final size
         return None # Return None to indicate failure
    except Exception as e:
        print(f"Error: Error during e2p conversion: {e}")
        return None # Return None to indicate failure

    # 9. Save perspective image
    try:
        pers_image = Image.fromarray(img_pers)
        pers_image.save(output_path)
        # print(f"Perspective view saved to: {output_path}")
    except Exception as e:
        print(f"Error: Error saving perspective view {output_path}: {e}")
        # Even if saving fails, the conversion itself might be successful, so still return parameters
        # But if you want saving failure to count as overall failure, return None

    # 10. Return dictionary containing parameters
    perspective_params = {
        'center_u_deg': center_u,
        'center_v_deg': center_v,
        'final_hfov_deg': final_h_fov,
        'final_vfov_deg': final_v_fov,
        'output_height': final_out_hw[0],
        'output_width': final_out_hw[1],
        'expansion_factor_used': expansion_factor,
        'vertical_offset_factor_used': vertical_offset_factor,
        'original_bbox_pixels': bbox,
        'erp_image_height': h_erp,
        'erp_image_width': w_erp
    }
    return perspective_params
# --- erp_bbox_to_perspective function ends ---


# Create custom dataset class
class CarryRelationDataset(Dataset):
    def __init__(self, base_data_path, regenerate_existing=False):
        self.base_data_path = base_data_path
        self.regenerate_existing = regenerate_existing
        
        # If regeneration is needed, clean existing results first
        if self.regenerate_existing:
            self.clean_existing_results()
            
        self.tasks = self.collect_tasks()
        
    def clean_existing_results(self):
        """When regenerate=True, delete all existing result files and images"""
        # Iterate through all scenes
        scene_paths = glob.glob(os.path.join(self.base_data_path, "*"))
        for scene_path in scene_paths:
            # Get all view folders under this scene
            view_paths = glob.glob(os.path.join(scene_path, "matterport_stitched_images", "*"))
            
            for view_path in view_paths:
                # Clean all JSON files in carries_results directory
                carries_save_dir = os.path.join(view_path, "carries_results")
                if os.path.exists(carries_save_dir):
                    json_files = glob.glob(os.path.join(carries_save_dir, "*.json"))
                    for json_file in json_files:
                        os.remove(json_file)
                
                # Clean all images in perspective_views directory
                perspective_save_dir = os.path.join(view_path, "perspective_views")
                if os.path.exists(perspective_save_dir):
                    image_files = glob.glob(os.path.join(perspective_save_dir, "*.jpg"))
                    for image_file in image_files:
                        os.remove(image_file)
        
    def collect_tasks(self):
        tasks = []
        # Iterate through all scenes
        scene_paths = glob.glob(os.path.join(self.base_data_path, "*"))
        print(f"Found {len(scene_paths)} scene directories")
        
        for scene_path in scene_paths:
            scene_id = os.path.basename(scene_path)
            
            # Get all view folders under this scene
            view_paths = glob.glob(os.path.join(scene_path, "matterport_stitched_images", "*"))
            
            for view_path in view_paths:
                view_id = os.path.basename(view_path)
                
                # Modify here: Check if filtered_selected_detections.json exists
                bbox_file_path = os.path.join(view_path, "filtered_selected_detections.json")
                if not os.path.exists(bbox_file_path):
                    print(f"Warning: filtered_selected_detections.json not found in {view_path}, skipping this view.") # Added print info
                    continue
                
                # Modify here: Point to the correct image path
                image_path = os.path.join(image_base_path, scene_id, "matterport_stitched_images", f"{view_id}.png")
                if not os.path.exists(image_path):
                    print(f"Warning: Image {view_id}.png not found in {image_base_path}/{scene_id}/matterport_stitched_images/, skipping this view.") # Added print info
                    continue
                
                # Create directory to save results
                carries_save_dir = os.path.join(view_path, "carries_results")
                
                # Add to task list
                tasks.append((scene_id, view_id, bbox_file_path, image_path, carries_save_dir))
        
        print(f"Collected total of {len(tasks)} tasks")
        return tasks
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        scene_id, view_id, bbox_file_path, image_path, carries_save_dir = self.tasks[idx]
        
        return {
            'scene_id': scene_id,
            'view_id': view_id,
            'bbox_file_path': bbox_file_path,
            'image_path': image_path,
            'carries_save_dir': carries_save_dir
        }

def process_batch_items(batch_items, model, processor, model_batch_size, regenerate_existing=False, save_images=True):
    """
    Processes a batch of panorama items, collects all objects, and then runs model inference in chunks of model_batch_size.
    batch_items: A list of panorama tasks returned by DataLoader.
    model_batch_size: The number of objects processed per batch during model inference.
    """
    success_count = 0
    all_object_messages = []
    all_object_perspective_imgs = []
    all_object_metadata = []
    processed_panorama_dirs_in_batch = set() # <--- New: Record directories processed in this batch

    # --- Phase 1: Collect all objects to be processed from the panorama batch ---
    print(f"Starting to collect objects from {len(batch_items)} panoramas...")
    for item in batch_items: # batch_items is the list of panorama tasks
        scene_id = item['scene_id']
        view_id = item['view_id']
        bbox_file_path = item['bbox_file_path']
        image_path = item['image_path'] # Original panorama path
        carries_save_dir = item['carries_save_dir']

        # Create save directories
        perspective_save_dir = os.path.join(os.path.dirname(carries_save_dir), "perspective_views")
        os.makedirs(carries_save_dir, exist_ok=True)
        os.makedirs(perspective_save_dir, exist_ok=True)

        # Load bbox data
        try:
            with open(bbox_file_path, 'r', encoding='utf-8') as f:
                bbox_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: bbox JSON file not found: {bbox_file_path}")
            continue
        except json.JSONDecodeError:
            print(f"Error: Unable to parse bbox JSON file: {bbox_file_path}")
            continue

        if not bbox_data.get("detections") or len(bbox_data["detections"]) == 0:
            # print(f"No detections found: {bbox_file_path}") # This info might be too much, commented out temporarily
            continue

        # Process each detection object
        for idx, detection in enumerate(bbox_data["detections"]):
            if "bbox" not in detection:
                print(f"Warning: Detection item at index {idx} missing 'bbox' field, skipping. File: {bbox_file_path}")
                continue

            bbox = detection["bbox"] # Original bbox

            # Define paths for perspective view and results
            perspective_path = os.path.join(
                perspective_save_dir,
                f"{view_id}_obj{idx}.jpg"
            )
            carries_result_path = os.path.join(
                carries_save_dir,
                f"{view_id}_obj{idx}_carries.json"
            )

            # Check if existing results need to be skipped
            if not regenerate_existing and os.path.exists(carries_result_path):
                # print(f"Result file exists, skipping: {carries_result_path}")
                processed_panorama_dirs_in_batch.add(carries_save_dir) # <--- Record directory if skipped
                continue

            # --- Call erp_bbox_to_perspective to generate perspective view ---
            try:
                perspective_params = erp_bbox_to_perspective(
                    json_path=bbox_file_path,
                    output_path=perspective_path,
                    detection_index=idx,
                    out_hw=(OUTPUT_HEIGHT, OUTPUT_WIDTH)
                )

                if perspective_params is None:
                    print(f"Error: erp_bbox_to_perspective failed to process bbox index {idx} (file: {bbox_file_path}). Skipping this object.")
                    continue

                # Load generated perspective image for model use
                try:
                    perspective_img = Image.open(perspective_path)
                    if perspective_img.mode != 'RGB':
                        perspective_img = perspective_img.convert('RGB')
                except FileNotFoundError:
                    print(f"Error: Generated image file not found from erp_bbox_to_perspective: {perspective_path}")
                    continue
                except Exception as e:
                    print(f"Error: Error loading perspective image {perspective_path}: {e}")
                    continue

                # If saving images is not required, delete it
                if not save_images:
                    try:
                        os.remove(perspective_path)
                    except OSError as e:
                        print(f"Warning: Failed to delete perspective image {perspective_path}: {e}")

            except Exception as e:
                print(f"Error: Exception occurred while calling erp_bbox_to_perspective (bbox index: {idx}, file: {bbox_file_path}): {e}")
                continue
            # --- erp_bbox_to_perspective call ended ---
            chosen_item_description = detection["category"] if "category" in detection else detection.get("description", "unknown object")
            # Construct messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": perspective_img # Use loaded perspective view
                        },
                        {"type": "text", "text": f'''
The chosen item is: {chosen_item_description}

Your task is to identify the `chosen_item` and list ONLY the items that are **directly and physically supported** by it. This means items that are:
1.  **Resting directly ON TOP** of the chosen item.
2.  **Contained INSIDE** the chosen item (like flowers in a vase).
3.  **Directly attached TO** the chosen item (less common in indoor scenes, but possible).

**Crucially, DO NOT LIST items that are merely:**
* Behind the chosen item.
* In front of the chosen item.
* Next to the chosen item.
* Visible through a gap or underneath the chosen item (like the floor under a table).

Think: If the `chosen_item` were instantly removed, ONLY list the items that would fall *because* the chosen_item is gone. Ignore items that are simply nearby or in the background/foreground.

List each physically supported item (and the chosen item itself) separately, one per line.

### Example 1:
a scene with a table, a vase with flowers and two books on the table.

When the chosen item is a table, the vase, runner, and books are directly ON the table. Output:
- table (1)
- vase with flowers (1)
- table runner (1)
- book (2)

When the chosen item are books. Output:
- book (2)

### Example 2:
a scene with a sofa, some pillows on the sofa.

When the chosen item is a sofa, the pillows are directly ON the sofa. Output:
- sofa (1)
- pillow (4)

When the chosen item are pillows, it rests ON the sofa but does not support other items. Output:
- pillow (4)

### Example 3:
a scene with a vase, some flowers in the vase.

When the chosen item is a vase or vase with flowers, it contains the flowers. Output:
- vase (1)
- flowers (1)

When the chosen item is a flower, it is contained WITHIN the vase but supports nothing else. Output:
- flowers (1)

### The format of the output should be:

- chosen item (quantity)
- supported item1 (quantity)
- supported item2 (quantity)
- ...

or

- chosen item (quantity)

The name of supported item (if there is) just output the name, not the description like "supported item:".
If there is no supported item for chosen item, just output the name of chosen item. The rug and carpet are not special items, if chosen item is rug or carpet, just output the name of chosen item.
Always include the quantity of each item in parentheses (1) if singular or the exact number if multiple. The "()" only contain the quantity, not the description.
For supported objects with thin surfaces, such as table runner, paper, tablecloth, towels, etc., also need to be detected if they are directly supported.
The description of chosen item and supported item should not contain location information.

IMPORTANT: ONLY output the item list in the format shown above. DO NOT include any explanations, reasoning, or additional text. DO NOT include any lines starting with "###" or containing "Explanation". Keep your response concise and limited strictly to the item list format.
'''}
                    ]
                }
            ]

            # Add to list of objects to be processed
            all_object_messages.append(messages)
            all_object_perspective_imgs.append(perspective_img)
            all_object_metadata.append({
                'scene_id': scene_id,
                'view_id': view_id,
                'detection_idx': idx,
                'detection': detection,
                'image_path': image_path,
                'perspective_path': perspective_path if save_images else None,
                'carries_result_path': carries_result_path,
                'original_bbox': bbox,
                'perspective_params': perspective_params,
                'carries_save_dir': carries_save_dir # <--- Ensure metadata contains save directory
            })
        # --- End processing objects for current panorama ---
    # --- End collecting objects from all panoramas ---

    # --- Phase 2: Process collected objects in chunks of model_batch_size ---
    total_objects_collected = len(all_object_messages)
    if total_objects_collected == 0:
        print("No valid objects collected for processing from the current batch of panoramas.")
        return success_count # Return 0

    print(f"Collection complete, {total_objects_collected} objects pending. Starting model inference with batch size {model_batch_size}...")

    for i in range(0, total_objects_collected, model_batch_size):
        # Get object data chunk for current model batch
        chunk_messages = all_object_messages[i : i + model_batch_size]
        # chunk_perspective_imgs = all_object_perspective_imgs[i : i + model_batch_size] # Not directly used
        chunk_metadata = all_object_metadata[i : i + model_batch_size]

        current_chunk_size = len(chunk_messages)
        print(f"\nProcessing model batch {i // model_batch_size + 1}/{math.ceil(total_objects_collected / model_batch_size)} (Object {i+1} to {i + current_chunk_size} / Total {total_objects_collected})")

        if not chunk_messages: # Safety check
            continue

        # --- Prepare inputs for current model batch ---
        try:
            # process_vision_info needs messages list to extract images
            image_inputs, video_inputs = process_vision_info(chunk_messages)

            # Prepare text inputs
            texts = [
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in chunk_messages
            ]

            # Use processor to process text and extracted images/videos
            inputs = processor(
                text=texts,
                images=image_inputs, # Use images extracted from chunk_messages
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
        except Exception as e:
            print(f"Error: Error preparing inputs for model batch: {e}")
            print(f"Skipping this model batch (Object {i+1} to {i + current_chunk_size})")
            continue # Skip to next model batch

        # --- Model batch inference ---
        try:
            with torch.no_grad(): # Gradients not needed for inference
                generated_ids = model.generate(**inputs, max_new_tokens=2048)

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        except Exception as e:
            print(f"Error: Model inference failed: {e}")
            # Consider printing inputs shape for debugging
            # print(f"Input shapes: {inputs['input_ids'].shape if 'input_ids' in inputs else 'N/A'}")
            print(f"Skipping this model batch (Object {i+1} to {i + current_chunk_size})")
            # Cleaning GPU memory might help recovery
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue # Skip to next model batch

        # --- Process each output result of current model batch ---
        for obj_idx_in_chunk, (output_text, metadata) in enumerate(zip(output_texts, chunk_metadata)):
            global_obj_idx = i + obj_idx_in_chunk # Calculate global object index (relative to all collected objects)
            # print(f"Model batch item {obj_idx_in_chunk+1}/{current_chunk_size} (Global {global_obj_idx+1}/{total_objects_collected}) Result:")
            # print(output_text) # Print full output might be too much

            # Extract item name list and quantity from conversation result
            item_names_with_quantity = [item.strip() for item in output_text.split('\n') if item.strip()]
            item_details = []
            for item in item_names_with_quantity:
                if item.startswith("###") or "Explanation" in item:
                    continue
                if item.startswith("- "):
                    item = item[2:]
                quantity_match = re.search(r'\((\d+)\)$', item)
                if quantity_match:
                    quantity = int(quantity_match.group(1))
                    name = item[:item.rfind('(')].strip()
                    item_details.append({"name": name, "quantity": quantity})
                else:
                    item_details.append({"name": item, "quantity": 1}) # Default quantity is 1

            # print("Extracted items and quantities:", item_details) # Print extracted result might be too much

            # Prepare dictionary for saving results
            carry_result = {
                "scene_id": metadata["scene_id"],
                "view_id": metadata["view_id"],
                "original_erp_image": metadata["image_path"],
                "perspective_image": metadata["perspective_path"],
                "original_bbox": metadata["original_bbox"],
                "perspective_transform_params": metadata["perspective_params"],
                "original_category": metadata["detection"].get("category", ""),
                "original_description": metadata["detection"].get("description", ""),
                "carried_items": item_details,
            }

            # Save results to JSON file (includes NumPy type conversion logic)
            try:
                # Attempt direct save
                with open(metadata["carries_result_path"], 'w', encoding='utf-8') as f:
                    json.dump(carry_result, f, indent=2, ensure_ascii=False)
                success_count += 1
                processed_panorama_dirs_in_batch.add(metadata['carries_save_dir']) # <--- Record directory after successful save
            except TypeError:
                # If there are NumPy types, perform conversion
                if 'perspective_transform_params' in carry_result and carry_result['perspective_transform_params']:
                    params = carry_result['perspective_transform_params']
                    for key, value in params.items():
                        if isinstance(value, np.generic):
                            params[key] = value.item()
                        elif isinstance(value, list):
                            params[key] = [v.item() if isinstance(v, np.generic) else v for v in value]
                # Try saving again
                try:
                    with open(metadata["carries_result_path"], 'w', encoding='utf-8') as f:
                        json.dump(carry_result, f, indent=2, ensure_ascii=False)
                    success_count += 1
                    processed_panorama_dirs_in_batch.add(metadata['carries_save_dir']) # <--- Record directory after successful converted save
                except Exception as final_e:
                     print(f"Error: JSON serialization still failed after NumPy type conversion: {final_e}")
                     # Save failed, do not record directory
            except IOError as e:
                print(f"Error: Failed to save JSON file {metadata['carries_result_path']}: {e}")
                # Save failed, do not record directory
            except Exception as e: # Catch other possible save errors
                print(f"Error: Unknown error occurred while saving JSON {metadata['carries_result_path']}: {e}")
                # Save failed, do not record directory

        # --- Clean up memory for current model batch ---
        del inputs, generated_ids, generated_ids_trimmed, output_texts, chunk_messages, chunk_metadata
        # image_inputs and video_inputs are local variables, will be cleaned automatically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # --- Model batch processing ends ---

    # --- New: Print result save directories for panoramas processed in this batch ---
    if processed_panorama_dirs_in_batch:
        print(f"\n--- Panorama results in this batch saved to (or already exist in) the following directories ---")
        # Use sorted to ensure consistent output order
        for save_dir in sorted(list(processed_panorama_dirs_in_batch)):
            print(f"- {save_dir}")
        print("----------------------------------------------------")
    # --- Print ends ---

    print(f"Finished processing {total_objects_collected} collected objects.")
    return success_count # Return total number of successfully processed objects in this panorama batch

def batch_process_with_dataloader(panorama_batch_size=1, model_batch_size=1, num_workers=0, regenerate_existing=False, save_images=True):
    print("Using loaded model...")

    print("Creating dataset...")
    dataset = CarryRelationDataset(base_data_path, regenerate_existing)
    print(f"Collected total of {len(dataset)} panorama processing tasks")

    if len(dataset) == 0:
        print("No tasks to process, exiting program")
        return

    # Create DataLoader, using panorama_batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=panorama_batch_size, # Controls how many panorama tasks are loaded at once
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x
    )

    # Record start time
    start_time = time.time()

    # --- Calculate total object count to provide more accurate progress ---
    total_objects_to_process = 0
    print("Calculating total object count...")
    # ... (Logic for calculating total_objects_to_process remains unchanged) ...
    for task_data in tqdm(dataset.tasks, desc="Calculating total objects"):
        _, _, bbox_file_path, _, carries_save_dir = task_data
        try:
            with open(bbox_file_path, 'r') as f:
                bbox_data = json.load(f)
            num_detections = len(bbox_data.get("detections", []))

            if not regenerate_existing:
                count_existing = 0
                if os.path.exists(carries_save_dir):
                     view_path = os.path.dirname(carries_save_dir)
                     view_id = os.path.basename(view_path)
                     for i in range(num_detections):
                         carries_result_path = os.path.join(
                             carries_save_dir,
                             f"{view_id}_obj{i}_carries.json"
                         )
                         if os.path.exists(carries_result_path):
                             count_existing += 1
                num_detections_to_process = num_detections - count_existing
            else:
                num_detections_to_process = num_detections

            total_objects_to_process += max(0, num_detections_to_process) # Ensure no negative numbers are added
        except FileNotFoundError:
             print(f"Warning: File not found when calculating object count {bbox_file_path}")
        except json.JSONDecodeError:
             print(f"Warning: Unable to parse JSON file when calculating object count {bbox_file_path}")
        except Exception as e:
            print(f"Warning: Unknown error occurred when calculating object count {bbox_file_path}: {e}")
    print(f"Estimated total objects to process: {total_objects_to_process}")
    # --- Total object count calculation ends ---

    processed_objects_count = 0
    # --- Use tqdm to display object processing progress ---
    # Note: dataloader now yields panorama batches
    with tqdm(total=total_objects_to_process, desc="Processing objects") as pbar:
        for panorama_batch in dataloader: # panorama_batch is a list containing panorama_batch_size panorama tasks
            # Call modified batch processing function, passing model batch size
            try:
                # process_batch_items returns the number of objects successfully processed in this panorama batch
                success_count_in_batch = process_batch_items(
                    panorama_batch,
                    model,
                    processor,
                    model_batch_size, # Pass model inference batch size
                    regenerate_existing,
                    save_images
                )
                processed_objects_count += success_count_in_batch
                pbar.update(success_count_in_batch) # Update progress bar
            except Exception as e:
                # Catch unexpected errors not handled within process_batch_items
                print(f"\nCritical error occurred while processing panorama batch: {e}")
                # Try to estimate how many objects this batch might contain to update progress bar, but this is inaccurate
                # num_objects_in_failed_batch = sum(len(json.load(open(item['bbox_file_path']))['detections']) for item in panorama_batch if os.path.exists(item['bbox_file_path']))
                # pbar.update(num_objects_in_failed_batch) # Not recommended, might cause progress bar to exceed 100% or be inaccurate
                print("Attempting to continue to next panorama batch...")
                # Clean up potential GPU memory residue
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue # Continue to next panorama batch

    # Calculate total elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nAll tasks completed, successfully processed and saved {processed_objects_count}/{total_objects_to_process} objects, elapsed time {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze object carrying relations using Qwen VL model (Batch processing objects)')
    # Modified batch_size to model_batch_size
    parser.add_argument('--model_batch_size', type=int, default=6, help='Number of objects processed per batch during model inference (affects GPU memory usage)') # Added default value
    # Added panorama_batch_size
    parser.add_argument('--panorama_batch_size', type=int, default=1, help='Number of panoramas loaded and preprocessed at once (affects memory and preprocessing efficiency)')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of data loading worker processes (affects data preparation speed)')
    parser.add_argument('--regenerate', default=False, action='store_true', help='Regenerate existing result files (Default: False)')
    parser.add_argument('--save-images', default=False, action=argparse.BooleanOptionalAction, help='Whether to save generated perspective images (Default: True, use --no-save-images to disable)') # Default changed to True

    args = parser.parse_args()

    # Use new parameter names
    model_batch_size = args.model_batch_size
    panorama_batch_size = args.panorama_batch_size
    num_workers = args.num_workers
    regenerate_existing = args.regenerate
    save_images = args.save_images

    print(f"Starting processing...")
    print(f"Model inference batch size (objects): {model_batch_size}")
    print(f"Panorama loading batch size: {panorama_batch_size}")
    print(f"Data loading worker processes: {num_workers}")
    print(f"Regenerate existing results: {regenerate_existing}")
    print(f"Save perspective images: {save_images}")

    # Call main function with new parameters
    batch_process_with_dataloader(
        panorama_batch_size=panorama_batch_size,
        model_batch_size=model_batch_size,
        num_workers=num_workers,
        regenerate_existing=regenerate_existing,
        save_images=save_images
    )
