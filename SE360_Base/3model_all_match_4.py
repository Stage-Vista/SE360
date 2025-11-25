import json
import os
import argparse
import cv2
import numpy as np

# --- Constants Definition ---
# Define colors for different box types (BGR format)
COLOR_VLM = (255, 0, 0)         # Blue
COLOR_DINO = (0, 0, 255)        # Red (Now used for final selected DINO)
COLOR_FLORENCE2 = (255, 0, 255) # Purple (Used for final selected Florence2)
COLOR_FINAL_SELECTED = (0, 255, 0) # Green (Backup/Unknown source)
TEXT_COLOR = (255, 255, 255)    # White (For label background)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2

def calculate_area(bbox):
    """Calculate the area of the bounding box."""
    try:
        xmin, ymin, xmax, ymax = map(float, bbox)
        # Ensure xmax >= xmin and ymax >= ymin
        width = max(0, xmax - xmin)
        height = max(0, ymax - ymin)
        return width * height
    except (ValueError, TypeError, IndexError):
        print(f"Warning: Invalid bbox format encountered when calculating area: {bbox}")
        return 0 # Return 0 area

def draw_boxes(image, detections, color, label_prefix=""):
    """Draw bounding boxes and labels on the image."""
    for det in detections:
        # Ensure bbox exists and format is correct
        bbox = None
        if 'bbox' in det and len(det['bbox']) == 4: # DINO format
            bbox = det['bbox']
        elif 'bbox_2d' in det and len(det['bbox_2d']) == 4: # VLM format
            bbox = det['bbox_2d']
        else:
            continue # Skip invalid detections

        # Convert coordinates to integers
        try:
            xmin, ymin, xmax, ymax = map(int, bbox)
        except (ValueError, TypeError):
            print(f"Warning: Invalid bbox coordinate format: {bbox}")
            continue

        # Draw rectangle
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, THICKNESS)

        # Prepare label text - Modified to prioritize category
        label = f"{label_prefix}{det.get('category', 'N/A')}" # Use category as label
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
        # Draw text background
        cv2.rectangle(image, (xmin, ymin - text_height - baseline), (xmin + text_width, ymin), color, -1) # -1 means fill
        # Draw text
        cv2.putText(image, label, (xmin, ymin - baseline // 2), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA)
    return image

def visualize_results(image_path, vlm_detections, final_selected_detections, output_vis_path):
    """Load image and draw only the final selected boxes."""
    try:
        # --- Added: Check and modify image_path ---
        original_image_path = image_path # Keep original path for debugging or error handling
        if image_path and image_path.startswith('/vol/'):
            print(f"Detected image path starting with '/vol/': {image_path}")
            try:
                parts = image_path.strip('/').split('/') # Remove leading slash and split
                # Find positions of 'mp3d_skybox' and 'matterport_stitched_images'
                # Assumes path structure is fixed
                idx_skybox = -1
                idx_images = -1
                for i, part in enumerate(parts):
                    if part == 'mp3d_skybox':
                        idx_skybox = i
                    elif part == 'matterport_stitched_images':
                        idx_images = i
                        break # Stop once images is found

                if idx_skybox != -1 and idx_images != -1 and idx_skybox + 1 < idx_images and idx_images + 1 < len(parts):
                    scene = parts[idx_skybox + 1]
                    view_filename = parts[idx_images + 1]
                    # Construct new fixed base path
                    fixed_base_path = "/home/lab401/zhongfile/Project/PanFusion/data/Matterport3D/mp3d_skybox/"
                    image_path = os.path.join(fixed_base_path, scene, 'matterport_stitched_images', view_filename)
                    print(f"Image path modified to: {image_path}")
                else:
                     print(f"Warning: Could not accurately extract scene and view filenames from path '{original_image_path}'. Attempting to use original path.")
                     image_path = original_image_path # Fallback on failure

            except Exception as e: # Catch broader errors like split or index failure
                print(f"Warning: Error parsing path starting with '/vol/': {original_image_path}. Error: {e}. Attempting to use original path.")
                image_path = original_image_path # Fallback on failure
        # --- End added code ---

        image = cv2.imread(image_path)
        if image is None:
            # Show both attempted and original path in error message if they differ
            error_msg = f"Error: Could not load image for visualization: {image_path}"
            if image_path != original_image_path:
                error_msg += f" (Original path: {original_image_path})"
            print(error_msg)
            return

        # --- Remove VLM box drawing code ---
        # image = draw_boxes(image, vlm_detections, COLOR_VLM, "VLM: ") # Commented out

        # Draw final selected boxes (Colored by source)
        for det in final_selected_detections:
            source = det.get('source', 'Unknown') # Added 'source' key during filtering
            label_prefix = f"{source}: "
            # Determine color and bbox key (Assuming final result uses 'bbox')
            color = COLOR_DINO if source == 'DINO' else COLOR_FLORENCE2 if source == 'Florence2' else COLOR_FINAL_SELECTED
            bbox = det.get('bbox') # Assume final filtered results uniformly use 'bbox'

            if bbox and len(bbox) == 4:
                 # Convert coordinates to integers
                try:
                    xmin, ymin, xmax, ymax = map(int, bbox)
                except (ValueError, TypeError):
                    print(f"Warning: Invalid bbox coordinate format during visualization: {bbox}")
                    continue

                # Draw rectangle
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, THICKNESS)

                # Prepare label text - Prioritize category, otherwise use description
                label_text = det.get('category', det.get('description', 'N/A'))
                label = f"{label_prefix}{label_text}"
                # Calculate text size for background
                (text_width, text_height), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
                # Draw text background
                cv2.rectangle(image, (xmin, ymin - text_height - baseline), (xmin + text_width, ymin), color, -1) # -1 means fill
                # Draw text
                cv2.putText(image, label, (xmin, ymin - baseline // 2), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA)
            else:
                 print(f"Warning: Final detection result missing valid 'bbox': {det}")


        # Save visualization result
        cv2.imwrite(output_vis_path, image)
        print(f"Visualization result saved to: {output_vis_path}")

    except Exception as e:
        print(f"Error: An error occurred during visualization ({image_path}): {e}") # image_path might be the modified one


def calculate_iou(boxA, boxB):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    Bounding box format: [xmin, ymin, xmax, ymax]
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Calculate the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Calculate the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Calculate union area
    unionArea = boxAArea + boxBArea - interArea

    # Calculate IoU
    iou = interArea / unionArea if unionArea > 0 else 0
    return iou

def apply_nms_per_category(detections, nms_threshold):
    """Apply Non-Maximum Suppression (NMS) on detections grouped by description."""
    if not detections:
        return []

    # Group by description
    grouped_detections = {}
    for det in detections:
        # Strictly use description for grouping
        description = det.get('description')
        if description is None:
            print(f"Warning: Detection missing 'description' found during NMS: {det}")
            continue # Skip detections without description
        if description not in grouped_detections:
            grouped_detections[description] = []
        grouped_detections[description].append(det)

    final_nms_detections = []
    for description, group in grouped_detections.items(): # Iterate through descriptions
        if not group:
            continue

        # Sort by confidence descending
        group.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)

        picked_indices_in_group = []
        all_indices_in_group = list(range(len(group)))

        while len(all_indices_in_group) > 0:
            current_best_idx = all_indices_in_group[0]
            picked_indices_in_group.append(current_best_idx)
            current_box = group[current_best_idx]['bbox']
            indices_to_remove = {current_best_idx}
            for i in range(1, len(all_indices_in_group)):
                compare_idx = all_indices_in_group[i]
                compare_box = group[compare_idx]['bbox']
                try:
                    iou = calculate_iou(current_box, compare_box)
                    if iou > nms_threshold:
                        indices_to_remove.add(compare_idx)
                except Exception as e:
                    # Print error using description instead of category
                    print(f"Error: IoU calculation error during NMS (Description: {description}). Boxes: {current_box}, {compare_box}. Error: {e}")

            remaining_indices = []
            for idx in all_indices_in_group:
                if idx not in indices_to_remove:
                    remaining_indices.append(idx)
            all_indices_in_group = remaining_indices

        for idx in picked_indices_in_group:
            final_nms_detections.append(group[idx])

    return final_nms_detections

def filter_and_select_detections(dino_json_path, florence2_json_path, vlm_json_path, output_json_path,
                                 iou_threshold=0.5, visualize=False, apply_nms=True, nms_threshold=0.3):
    """
    Filter DINO and Florence-2 detections based on VLM detection results.
    Core logic is strictly based on 'description'. 'category' is only used for output labels.
    New Rules + Optional NMS:
    1. VLM-driven Matching: For each VLM bbox, find IoU-matching DINO and Florence-2 boxes. From these matches, select the one with the largest area (VLM box itself is not selected).
    2. VLM Uncovered Matching: For categories not detected by VLM, if both DINO and Florence-2 detected them,
       find the DINO box with the highest confidence and match it against all same-category Florence-2 boxes (IoU threshold 0.5).
       From this high-confidence DINO box and all successfully matched Florence-2 boxes, select the one with the largest area.
    3. NMS (Optional): Apply Non-Maximum Suppression to the final list of selected boxes by category.

    Args:
        dino_json_path (str): Path to DINO JSON file.
        florence2_json_path (str): Path to Florence-2 JSON file.
        vlm_json_path (str): Path to VLM JSON file (Qwen).
        output_json_path (str): Path for outputting filtered results JSON.
        iou_threshold (float): IoU threshold used for VLM-driven matching.
        visualize (bool): Whether to generate visualization images.
        apply_nms (bool): Whether to apply Non-Maximum Suppression.
        nms_threshold (float): NMS threshold.
    """
    print(f"Processing: DINO='{os.path.basename(dino_json_path)}', Florence2='{os.path.basename(florence2_json_path)}', VLM='{os.path.basename(vlm_json_path)}' in {os.path.dirname(dino_json_path)}")

    # --- Load JSON Data ---
    def load_json(path, model_name):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            detections = data.get('detections')
            if not isinstance(detections, list):
                 print(f"Warning: {model_name} JSON file ('{path}') format incorrect or 'detections' list is empty/not a list.")
                 return None, []
            valid_detections = []
            bbox_key = 'bbox_2d' if model_name == "VLM (Qwen)" else 'bbox'
            desc_key = 'description'
            conf_key = 'confidence' # Try to read confidence

            for i, det in enumerate(detections):
                if not isinstance(det, dict):
                    print(f"Warning: {model_name} detection result {i} is not a dictionary: {det}")
                    continue
                bbox = det.get(bbox_key)
                desc = det.get(desc_key)
                # Get confidence, if invalid or missing set to 0.0
                conf = det.get(conf_key, 0.0)
                try:
                    conf = float(conf)
                except (ValueError, TypeError):
                    print(f"Warning: {model_name} detection result {i} has invalid confidence, setting to 0.0: {det.get(conf_key)}")
                    conf = 0.0
                det[conf_key] = conf # Ensure stored as float

                if desc is None:
                     print(f"Warning: {model_name} detection result {i} missing '{desc_key}': {det}")
                     continue
                if not isinstance(bbox, list) or len(bbox) != 4:
                    print(f"Warning: {model_name} detection result {i} has invalid or missing '{bbox_key}': {bbox}")
                    continue
                try:
                    coords = [float(coord) for coord in bbox]
                except (ValueError, TypeError):
                    print(f"Warning: {model_name} detection result {i} has non-numeric coordinates in '{bbox_key}': {bbox}")
                    continue
                # Unify bbox key, and add original index and source info (during loading)
                det['bbox'] = coords # Ensure float list
                det['original_index'] = i # Record original index
                det['source'] = model_name # Record source
                if model_name == "VLM (Qwen)" and 'bbox_2d' in det:
                    # If needed, original bbox_2d key can be deleted
                    # del det['bbox_2d']
                    pass

                valid_detections.append(det)

            return data, valid_detections
        except FileNotFoundError:
            print(f"Error: {model_name} JSON file not found: {path}")
            return None, []
        except json.JSONDecodeError:
            print(f"Error: Could not parse {model_name} JSON file: {path}")
            return None, []
        except Exception as e:
            print(f"Error: Unknown error loading {model_name} JSON file: {path}. Error: {e}")
            return None, []

    dino_data, dino_detections = load_json(dino_json_path, "DINO")
    florence2_data, florence2_detections = load_json(florence2_json_path, "Florence-2")
    vlm_data, vlm_detections = load_json(vlm_json_path, "VLM (Qwen)")

    if dino_data is None or florence2_data is None or vlm_data is None:
        print("Error: One or more input JSON files failed to load or were malformed. Skipping processing.")
        return
    # Note: Even if vlm_detections is empty, we still need to process DINO and Florence-2 cases

    final_selected_detections = []
    selected_dino_indices = set()
    selected_florence2_indices = set()

    # --- Step 1: VLM-driven matching (based on description, select max area from DINO/F2 only) ---
    vlm_processed_indices = set()
    vlm_descriptions_processed = set() # Store processed VLM descriptions

    for vlm_idx, vlm_det in enumerate(vlm_detections):
        vlm_processed_indices.add(vlm_idx)
        # Strictly use description for matching
        vlm_desc = vlm_det.get('description')
        if vlm_desc is None:
            print(f"Warning: VLM detection {vlm_idx} missing 'description', skipping.")
            continue
        vlm_descriptions_processed.add(vlm_desc) # Record processed description
        vlm_bbox = vlm_det['bbox']

        matching_dino_f2_candidates = []

        # Find matching DINO boxes (based on description)
        for dino_det in dino_detections:
            dino_desc = dino_det.get('description')
            # Strictly use description comparison
            if dino_desc == vlm_desc and dino_det['original_index'] not in selected_dino_indices:
                dino_bbox = dino_det['bbox']
                try:
                    iou = calculate_iou(vlm_bbox, dino_bbox)
                    if iou > iou_threshold:
                        dino_det_copy = dino_det.copy()
                        dino_det_copy['area'] = calculate_area(dino_bbox)
                        dino_det_copy['source'] = 'DINO'
                        matching_dino_f2_candidates.append(dino_det_copy)
                except Exception as e:
                    print(f"Error: Error calculating DINO-VLM IoU (Desc: {vlm_desc}). VLM bbox: {vlm_bbox}, DINO bbox: {dino_bbox}. Error: {e}")

        # Find matching Florence-2 boxes (based on description)
        for f2_det in florence2_detections:
            f2_desc = f2_det.get('description')
            # Strictly use description comparison
            if f2_desc == vlm_desc and f2_det['original_index'] not in selected_florence2_indices:
                f2_bbox = f2_det['bbox']
                try:
                    iou = calculate_iou(vlm_bbox, f2_bbox)
                    if iou > iou_threshold:
                        f2_det_copy = f2_det.copy()
                        f2_det_copy['area'] = calculate_area(f2_bbox)
                        f2_det_copy['source'] = 'Florence2'
                        matching_dino_f2_candidates.append(f2_det_copy)
                except Exception as e:
                    print(f"Error: Error calculating Florence2-VLM IoU (Desc: {vlm_desc}). VLM bbox: {vlm_bbox}, F2 bbox: {f2_bbox}. Error: {e}")

        if matching_dino_f2_candidates:
            matching_dino_f2_candidates.sort(key=lambda x: x['area'], reverse=True)
            selected_candidate = matching_dino_f2_candidates[0]

            final_det = {k: v for k, v in selected_candidate.items() if k not in ['area', 'iou_with_vlm']}
            if 'confidence' not in final_det and 'confidence' in selected_candidate:
                 final_det['confidence'] = selected_candidate['confidence']
            if 'source' not in final_det:
                final_det['source'] = selected_candidate.get('source', 'Unknown')

            # Set final 'category' field for output/visualization
            # Prioritize 'category' from the selected box, otherwise use its 'description'
            final_det['category'] = selected_candidate.get('category', selected_candidate.get('description'))
            # Ensure 'description' field also exists in final output (use selected box's description)
            if 'description' not in final_det:
                 final_det['description'] = selected_candidate.get('description')


            final_selected_detections.append(final_det)

            if selected_candidate['source'] == 'DINO':
                selected_dino_indices.add(selected_candidate['original_index'])
            elif selected_candidate['source'] == 'Florence2':
                selected_florence2_indices.add(selected_candidate['original_index'])

    # --- Step 2: Handle categories not covered by VLM (based on description, DINO and F2 must match) ---
    all_dino_descriptions = {det.get('description') for det in dino_detections if det.get('description') is not None}
    all_f2_descriptions = {det.get('description') for det in florence2_detections if det.get('description') is not None}
    common_unseen_descriptions = (all_dino_descriptions.intersection(all_f2_descriptions)) - vlm_descriptions_processed

    for desc in common_unseen_descriptions:
        unselected_dino_for_desc = [
            det for det in dino_detections
            if det.get('description') == desc and det['original_index'] not in selected_dino_indices
        ]
        unselected_f2_for_desc = [
            det for det in florence2_detections
            if det.get('description') == desc and det['original_index'] not in selected_florence2_indices
        ]

        # Must have both unselected DINO and F2 boxes
        if not unselected_dino_for_desc or not unselected_f2_for_desc:
            continue

        # Find the DINO box with the highest confidence
        unselected_dino_for_desc.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
        best_conf_dino = unselected_dino_for_desc[0]
        best_conf_dino_bbox = best_conf_dino['bbox']

        # Create paired candidate list, only adding successfully matched F2 boxes
        paired_candidates = []
        f2_match_found = False # Flag to record if a matching F2 box was found
        match_iou_threshold = 0.5

        # Perform IoU match with Florence-2 boxes
        for f2_det in unselected_f2_for_desc:
            f2_bbox = f2_det['bbox']
            try:
                iou = calculate_iou(best_conf_dino_bbox, f2_bbox)
                if iou > match_iou_threshold:
                    f2_det_copy = f2_det.copy()
                    f2_det_copy['area'] = calculate_area(f2_bbox)
                    f2_det_copy['iou_with_best_dino'] = iou # Optional
                    f2_det_copy['source'] = 'Florence2'
                    paired_candidates.append(f2_det_copy)
                    f2_match_found = True # Found at least one matching F2 box
            except Exception as e:
                print(f"Error: Error calculating DINO(best_conf)-F2 IoU (Desc: {desc}). DINO bbox: {best_conf_dino_bbox}, F2 bbox: {f2_bbox}. Error: {e}")

        # Continue only if at least one matching Florence-2 box was found
        if f2_match_found:
            # Add the highest confidence DINO box to candidates to participate in area comparison
            best_conf_dino_copy = best_conf_dino.copy()
            best_conf_dino_copy['area'] = calculate_area(best_conf_dino_bbox)
            best_conf_dino_copy['source'] = 'DINO'
            paired_candidates.append(best_conf_dino_copy)

            # Select the one with the largest area from paired candidates (containing DINO and matched F2)
            paired_candidates.sort(key=lambda x: x['area'], reverse=True)
            selected_unseen_candidate = paired_candidates[0] # Select max area

            # Add to final list
            final_unseen_det = {k: v for k, v in selected_unseen_candidate.items() if k not in ['area', 'iou_with_best_dino']}
            # ... (Logic to ensure confidence, source, category, description fields exist remains unchanged) ...
            if 'confidence' not in final_unseen_det and 'confidence' in selected_unseen_candidate:
                 final_unseen_det['confidence'] = selected_unseen_candidate['confidence']
            if 'source' not in final_unseen_det:
                final_unseen_det['source'] = selected_unseen_candidate.get('source', 'Unknown')
            final_unseen_det['category'] = selected_unseen_candidate.get('category', selected_unseen_candidate.get('description'))
            if 'description' not in final_unseen_det:
                 final_unseen_det['description'] = selected_unseen_candidate.get('description')


            final_selected_detections.append(final_unseen_det)

            # Mark selected DINO or Florence-2 box
            if selected_unseen_candidate['source'] == 'DINO':
                selected_dino_indices.add(selected_unseen_candidate['original_index'])
            elif selected_unseen_candidate['source'] == 'Florence2':
                selected_florence2_indices.add(selected_unseen_candidate['original_index'])
        # else: If f2_match_found is False, do nothing and skip this description

    # --- Step 3: Apply NMS (Optional, based on description) ---
    if apply_nms:
        print(f"Applying NMS (threshold={nms_threshold}) to {len(final_selected_detections)} initially selected boxes (based on description)...")
        count_before_nms = len(final_selected_detections)
        # NMS function modified to group by description
        final_selected_detections = apply_nms_per_category(final_selected_detections, nms_threshold)
        count_after_nms = len(final_selected_detections)
        print(f"Remaining boxes after NMS: {count_after_nms} (Removed {count_before_nms - count_after_nms}).")

    # --- Construct and save output ---
    # Try to get image_path from VLM, DINO, Florence2
    image_path = vlm_data.get("image_path", dino_data.get("image_path", florence2_data.get("image_path", "N/A")))
    if image_path == "N/A":
         print(f"Warning: Failed to find valid 'image_path' in any JSON file for {os.path.dirname(output_json_path)}")
    # Save original image_path to JSON, path modification logic moved to visualize_results
    output_data = {
        "image_path": image_path, # Still saving the original path read from JSON
        "detections": final_selected_detections # Contains 'bbox', 'description', 'source', 'confidence'(optional), 'category'
    }

    output_dir = os.path.dirname(output_json_path)
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Filtering complete. Results saved to: {output_json_path}")
        # Update statistics
        final_dino_count = sum(1 for det in final_selected_detections if det.get('source') == 'DINO')
        final_f2_count = sum(1 for det in final_selected_detections if det.get('source') == 'Florence2')
        final_vlm_count = sum(1 for det in final_selected_detections if det.get('source') == 'VLM') # VLM might also be selected
        print(f"  Original DINO: {len(dino_detections)}, Florence-2: {len(florence2_detections)}, VLM: {len(vlm_detections)}")
        print(f"  Final Selected: {len(final_selected_detections)} (From DINO: {final_dino_count}, Florence2: {final_f2_count}, VLM: {final_vlm_count})")
        # print(f"  Selected DINO indices: {selected_dino_indices}") # Debug
        # print(f"  Selected F2 indices: {selected_florence2_indices}") # Debug
    except IOError as e:
        print(f"Error: Could not write output file {output_json_path}. Error: {e}")
    except TypeError as e:
         print(f"Error: Could not serialize JSON data to write to {output_json_path}. Error: {e}")
         # print(f"Data: {output_data}") # Uncomment cautiously

    # --- Visualization ---
    if visualize:
        # Call visualize_results using original image_path read from JSON
        # visualize_results internally handles path conversion
        if image_path and image_path != "N/A":
            base_name = os.path.basename(output_json_path).replace('.json', '')
            output_vis_path = os.path.join(output_dir, f"{base_name}_visualization.png")
            # Pass original image_path when calling visualize_results
            visualize_results(image_path, vlm_detections, final_selected_detections, output_vis_path)
            # Note: No need to check os.path.exists(image_path) here as visualize_results handles load failure
        elif not image_path or image_path == "N/A":
             print(f"Warning: Cannot visualize, valid image_path not found in JSON")
        # elif not os.path.exists(image_path): # Removed this check, let visualize_results handle it
        #      print(f"Warning: Cannot visualize, image file not found: {image_path}")


def process_directory(base_path, iou_thr=0.5, visualize=False):
    """
    Traverse the specified base path, find and process DINO, Florence-2, and VLM (Qwen) JSON files.
    """
    print(f"Start processing directory: {base_path}")
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Traverse scene directory
    for scene_name in os.listdir(base_path):
        scene_path = os.path.join(base_path, scene_name)
        if not os.path.isdir(scene_path):
            continue

        images_path = os.path.join(scene_path, 'matterport_stitched_images')
        if not os.path.isdir(images_path):
            # print(f"Skipping {scene_path}: 'matterport_stitched_images' directory not found")
            continue

        # Traverse view directory
        for view_name in os.listdir(images_path):
            view_path = os.path.join(images_path, view_name)
            if not os.path.isdir(view_path):
                continue

            dino_json_path = os.path.join(view_path, 'bbox_dino.json')
            florence2_json_path = os.path.join(view_path, 'bbox_florence2.json') # Added
            vlm_json_path = os.path.join(view_path, 'bbox_qwen.json')
            # Modify output filename to reflect new logic
            output_json_path = os.path.join(view_path, 'filtered_selected_detections.json')

            # Check if all required JSON files exist
            required_files_exist = all(os.path.exists(p) for p in [dino_json_path, florence2_json_path, vlm_json_path])

            if required_files_exist:
                try:
                    # Call updated filtering function
                    filter_and_select_detections(dino_json_path, florence2_json_path, vlm_json_path, output_json_path,
                                                 iou_threshold=iou_thr, visualize=visualize)
                    processed_count += 1
                except Exception as e:
                    print(f"Unexpected error while processing {view_path}: {e}")
                    import traceback
                    traceback.print_exc() # Print detailed error stack
                    error_count += 1
            else:
                missing_files = []
                if not os.path.exists(dino_json_path): missing_files.append('bbox_dino.json')
                if not os.path.exists(florence2_json_path): missing_files.append('bbox_florence2.json')
                if not os.path.exists(vlm_json_path): missing_files.append('bbox_qwen.json')
                # print(f"Skipping {view_path}: Missing files: {', '.join(missing_files)}") # Reduce redundant output
                skipped_count += 1

    print(f"\nProcessing complete.")
    print(f"Number of file groups successfully processed: {processed_count}")
    print(f"Number of directories skipped due to missing files: {skipped_count}")
    print(f"Number of directories with errors during processing: {error_count}")


# --- Main Program Entry ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Filter DINO and Florence-2 detections based on VLM results and optionally visualize.")
    parser.add_argument("base_path", nargs='?', default='./data/Matterport3D/mp3d_base',
                        help="Base path containing all scenes (defaults to example path)")
    # Modified visualize default behavior to be False unless --visualize is explicitly specified
    parser.add_argument("--visualize", default=True, action="store_true", # True if present, default False (Note: original code said default=True in logic but argparse set it otherwise, assuming intent was flag toggles it)
                        help="If specified, generate and save visualization images with bounding boxes")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU threshold for filtering (default 0.5)")

    args = parser.parse_args()

    target_base_path = args.base_path
    enable_visualization = args.visualize
    iou_threshold = args.iou

    # Check if base path exists
    if not os.path.isdir(target_base_path):
        print(f"Error: Base path does not exist: {target_base_path}")
    else:
        # Call updated process_directory
        process_directory(target_base_path, iou_thr=iou_threshold, visualize=enable_visualization)