import sys
import os
import glob
import json
import time
import argparse
from pathlib import Path

# Add project root to sys.path if needed (adjust relative path if necessary)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM

# --- Configuration ---
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device used: {device}")

# Base data path (modify to your path)
base_data_path = "./data/Matterport3D/mp3d_base"
# Image file base path (modify to your path)
image_base_path = "./data/Matterport3D/mp3d_skybox"

# Florence-2 Model ID
FLORENCE2_MODEL_ID = "microsoft/Florence-2-large-ft"

# --- Initialize Florence-2 Model (Once) ---
print("Initializing Florence-2 model...")
florence2_model = AutoModelForCausalLM.from_pretrained(
    FLORENCE2_MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float32 # Use float16 for faster loading and inference
).eval().to(device)
florence2_processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)
print(f"Florence-2 model loaded. Model device: {florence2_model.device}, Model dtype: {florence2_model.dtype}")

# --- Helper Functions ---

# run_florence2 function (remains the same as before)
def run_florence2(task_prompt, text_input, model, processor, image):
    device = model.device
    dtype = model.dtype # Get model dtype
    # print(f"run_florence2: Model expects device {device}, dtype {dtype}") # Debug

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    # Ensure input matches the model's expected dtype
    # print("Preparing input data...") # Debug
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    # print(f"Input 'input_ids' device: {inputs['input_ids'].device}") # Debug
    # print(f"Input 'pixel_values' initial device: {inputs['pixel_values'].device}, initial dtype: {inputs['pixel_values'].dtype}") # Debug
    # Convert pixel_values to the model's dtype
    inputs['pixel_values'] = inputs['pixel_values'].to(dtype)
    # print(f"Input 'pixel_values' converted device: {inputs['pixel_values'].device}, converted dtype: {inputs['pixel_values'].dtype}") # Debug
    # If input_ids is not LongTensor, convert it (usually not needed, but just in case)
    if inputs['input_ids'].dtype != torch.long:
        # print(f"Converting 'input_ids' dtype from {inputs['input_ids'].dtype} to long") # Debug
        inputs['input_ids'] = inputs['input_ids'].long()

    # print("Starting generation...") # Debug
    try:
        generated_ids = model.generate(
          input_ids=inputs["input_ids"],
          pixel_values=inputs["pixel_values"],
          max_new_tokens=1024,
          early_stopping=False,
          do_sample=False,
          num_beams=3,
        )
        # print("Generation complete.") # Debug
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        # Ensure post_process_generation receives correct image size
        image_width, image_height = image.size
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image_width, image_height) # Pass PIL image size
        )
    except Exception as e:
        print(f"Error during Florence-2 inference: {e}")
        parsed_answer = {} # Return empty dict to indicate failure

    return parsed_answer

# visualize_bbox function (remains mostly the same, takes numpy array)
def visualize_bbox(image_np, detections):
    img_with_boxes = image_np.copy()

    for item in detections:
        # Get bounding box coordinates
        bbox = item['bbox']
        x1, y1, x2, y2 = map(int, bbox)

        # Get label (detected phrase)
        label = item['label']
        # score = item['score'] # Florence-2 Phrase Grounding does not directly provide scores

        # Draw bounding box
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add label (removed score display)
        short_label = label[:30] + "..." if len(label) > 30 else label # Slightly extend display
        text = f"{short_label}"
        # Adjust text position to prevent going out of bounds
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 15
        cv2.putText(img_with_boxes, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img_with_boxes

# --- Dataset Class ---
class SceneViewDataset(Dataset):
    def __init__(self, base_data_path, image_base_path, regenerate_existing=False):
        self.base_data_path = base_data_path
        self.image_base_path = image_base_path
        self.regenerate_existing = regenerate_existing
        self.tasks = self.collect_tasks()
        print(f"Dataset initialization complete, found {len(self.tasks)} tasks.")

    def collect_tasks(self):
        tasks = []
        print("Starting task collection...")
        # Iterate through all scenes
        scene_paths = glob.glob(os.path.join(self.base_data_path, "*"))
        print(f"Found {len(scene_paths)} scene directories.")
        for scene_path in scene_paths:
            if not os.path.isdir(scene_path): continue
            scene_id = os.path.basename(scene_path)

            # Get all view folder paths under this scene
            view_dir_pattern = os.path.join(scene_path, "matterport_stitched_images", "*")
            view_paths = glob.glob(view_dir_pattern)

            for view_path in view_paths:
                if not os.path.isdir(view_path): continue
                view_id = os.path.basename(view_path)

                # Check if object description JSON file exists
                json_file_path = os.path.join(view_path, "description", "objects.json")
                if not os.path.exists(json_file_path):
                    # print(f"Skipping {scene_id}/{view_id}: objects.json not found") # Debug
                    continue

                # Point to image file
                image_path = os.path.join(self.image_base_path, scene_id, "matterport_stitched_images", f"{view_id}.png")
                if not os.path.exists(image_path):
                    # print(f"Skipping {scene_id}/{view_id}: Image file {image_path} not found") # Debug
                    continue

                # Define output file paths
                view_save_dir = os.path.join(self.base_data_path, scene_id, "matterport_stitched_images", view_id) # Save in place
                bbox_output_path = os.path.join(view_save_dir, "bbox_florence2.json")
                output_img_path = os.path.join(view_save_dir, "detection_result_florence2.jpg")

                # If result file exists, decide whether to skip based on regenerate_existing
                if not self.regenerate_existing and os.path.exists(bbox_output_path) and os.path.exists(output_img_path):
                    # print(f"Skipping {scene_id}/{view_id}: Result already exists") # Debug
                    continue

                # Add to task list
                tasks.append({
                    'scene_id': scene_id,
                    'view_id': view_id,
                    'json_file_path': json_file_path,
                    'image_path': image_path,
                    'bbox_output_path': bbox_output_path,
                    'output_img_path': output_img_path
                })
        print(f"Task collection complete, {len(tasks)} valid tasks in total.")
        return tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        # Return task dict, actual loading happens in processing function
        return self.tasks[idx]

# --- Processing Function ---
def process_single_item_florence2(item, florence2_model, florence2_processor):
    scene_id = item['scene_id']
    view_id = item['view_id']
    json_file_path = item['json_file_path']
    image_path = item['image_path']
    bbox_output_path = item['bbox_output_path']
    output_img_path = item['output_img_path']

    print(f"\n--- Processing started: Scene={scene_id}, View={view_id} ---")
    print(f"Image path: {image_path}")
    print(f"JSON path: {json_file_path}")

    # Load image (using PIL)
    try:
        image_pil = Image.open(image_path).convert("RGB")
        original_width, original_height = image_pil.size
        print(f"Image loaded successfully, size: {original_width}x{original_height}")
    except Exception as e:
        print(f"Error: Unable to load image {image_path}: {e}")
        return False

    # Load object info and generate Caption
    original_objects_data = [] # Store original object info {description, category}
    original_descriptions = []
    # description_to_category = {} # Mapping no longer needed
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            objects_data = json.load(f)

        for obj_item in objects_data:
            # Ensure object has description and category
            if "description" in obj_item and "category" in obj_item:
                desc = obj_item["description"]
                cat = obj_item["category"]
                original_objects_data.append({"description": desc, "category": cat}) # Save original info
                original_descriptions.append(desc)

        if not original_descriptions:
            print("Warning: No valid 'description' and 'category' pairs found in JSON file.")
            # Create empty JSON result file, marking processing as complete but with no valid input
            os.makedirs(os.path.dirname(bbox_output_path), exist_ok=True)
            with open(bbox_output_path, 'w', encoding='utf-8') as f:
                json.dump({"image_path": image_path, "caption_used": "N/A - No valid descriptions found", "detections": []}, f, indent=2, ensure_ascii=False)
            print("Created empty JSON file because no valid descriptions were found.")
            return True # Consider processing complete

        # Concatenate descriptions into a single caption
        caption_text = ". ".join(original_descriptions) + "." # Join with periods
        print(f"Generated Caption: {caption_text[:200]}...") # Print part of caption

    except Exception as e:
        print(f"Error: Failed to load or process JSON file {json_file_path}: {e}")
        return False

    # --- Use Florence-2 for Phrase Grounding ---
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    results = run_florence2(task_prompt, caption_text, florence2_model, florence2_processor, image_pil)

    # Parse Florence-2 output
    parsed_results = results.get(task_prompt)
    if not parsed_results or 'bboxes' not in parsed_results or 'labels' not in parsed_results:
        print(f"Failed to get valid {task_prompt} results from Florence-2.")
        # Create empty JSON result file
        print("Creating empty JSON result file (Florence-2 returned no valid results).")
        os.makedirs(os.path.dirname(bbox_output_path), exist_ok=True)
        with open(bbox_output_path, 'w', encoding='utf-8') as f:
            json.dump({"image_path": image_path, "caption_used": caption_text, "detections": []}, f, indent=2, ensure_ascii=False)
        return True # Mark as complete (even with no results)


    bboxes = parsed_results.get('bboxes', [])
    labels = parsed_results.get('labels', []) # labels are detected phrases

    # Convert results to detections list format (containing phrase and bbox)
    raw_detections = []
    for bbox, label in zip(bboxes, labels):
        # Florence-2 Phrase Grounding has no score, set to 1.0
        raw_detections.append({'bbox': bbox, 'label': label, 'score': 1.0})

    print(f"Florence-2 detected {len(raw_detections)} initial bounding boxes.")
    if not raw_detections:
        print("Florence-2 returned no bounding boxes.")
        # Create empty JSON result file
        print("Creating empty JSON result file (Florence-2 returned empty bounding box list).")
        os.makedirs(os.path.dirname(bbox_output_path), exist_ok=True)
        with open(bbox_output_path, 'w', encoding='utf-8') as f:
            json.dump({"image_path": image_path, "caption_used": caption_text, "detections": []}, f, indent=2, ensure_ascii=False)
        return True

    # --- Apply Bounding Box Filtering ---
    image_area = original_width * original_height
    valid_detections = [] # Store detections passing filters (containing bbox and phrase/label)
    print("Starting bounding box filtering...")
    for det in raw_detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox

        # Check coordinate validity
        if not (0 <= x1 < x2 <= original_width and 0 <= y1 < y2 <= original_height):
             # print(f"  - Filtering: Invalid coordinates {bbox} for label '{det['label']}'") # Debug
             continue

        # 1. Check if bbox touches image boundaries (leave 30px margin)
        if x1 <= 30 or y1 <= 30 or x2 >= original_width - 30 or y2 >= original_height - 30:
            # print(f"  - Filtering (Boundary): {det['label']} bbox={bbox}") # Debug
            continue

        # Calculate bbox width and height
        width = x2 - x1
        height = y2 - y1

        # 2. Check if longest side of bbox exceeds 3/5 of image height
        max_side_length = max(width, height)
        max_allowed_length = original_height * 3 / 5
        if max_side_length > max_allowed_length:
            # print(f"  - Filtering (Size): {det['label']} Max side {max_side_length:.0f} > Allowed {max_allowed_length:.0f}") # Debug
            continue

        # Calculate bbox area and percentage of total image area
        bbox_area = width * height
        area_percentage = (bbox_area / image_area) * 100

        # 3. Check if bbox area is within allowed range (0.3% ~ 40%)
        if area_percentage < 0.3:
            # print(f"  - Filtering (Too small): {det['label']} Area {area_percentage:.2f}% < 0.3%") # Debug
            continue
        elif area_percentage > 40:
            # print(f"  - Filtering (Too large): {det['label']} Area {area_percentage:.2f}% > 40%") # Debug
            continue

        # Passed all checks, keep this bbox and corresponding phrase
        valid_detections.append(det) # det contains {'bbox': ..., 'label': phrase, 'score': 1.0}

    print(f"{len(valid_detections)} remaining after bounding box filtering.")

    # If no qualified detection results
    if len(valid_detections) == 0:
        print("All detection results failed filtering. Not saving visualization image, but saving empty JSON.")
        # Create empty JSON result file
        os.makedirs(os.path.dirname(bbox_output_path), exist_ok=True)
        with open(bbox_output_path, 'w', encoding='utf-8') as f:
            # Save empty list, but include original image path and used caption
            json.dump({"image_path": image_path, "caption_used": caption_text, "detections": []}, f, indent=2, ensure_ascii=False)
        print("Created empty JSON file (no valid bounding boxes passed filtering).")
        return True # Mark as complete

    # --- Build Final JSON Output ---
    # Goal: For each object in original objects.json, if its description's corresponding phrase
    #       finds a valid bbox in valid_detections, save the original description, category, and bbox.

    final_bbox_results = {
        "image_path": image_path,
        "caption_used": caption_text, # Save caption used for grounding
        "detections": [] # Store final results {description, category, bbox, score, matched_phrase}
    }

    # To facilitate searching, organize valid_detections by phrase (label)
    # Note: A phrase might correspond to multiple bboxes (though uncommon in this model)
    #       A description might contain multiple phrases
    # Simple processing: If an original description contains a valid_detection's phrase, associate them
    # More precise: If an original description exactly matches a valid_detection's phrase, associate them
    # Here we use containment because grounding might only recognize part of the description
    # Also, a description might match multiple bboxes, and a bbox might be matched by multiple descriptions
    # We need to find *one* suitable bbox for each original object

    matched_bbox_indices = set() # Track indices of matched valid_detections to avoid reusing bbox
    processed_original_indices = set() # Track processed original object indices

    print("Starting to match filtered bboxes back to original objects...")
    # Prioritize exact matching
    for i, orig_obj in enumerate(original_objects_data):
        if i in processed_original_indices: continue # Skip processed ones

        orig_desc = orig_obj["description"]
        orig_cat = orig_obj["category"]

        best_match_idx = -1
        # Find exact matching phrase
        for j, det in enumerate(valid_detections):
            if j in matched_bbox_indices: continue # Skip used bbox

            if det['label'] == orig_desc: # Exact match
                best_match_idx = j
                break # Use the first exact match found

        if best_match_idx != -1:
            matched_bbox = valid_detections[best_match_idx]['bbox']
            final_bbox_results["detections"].append({
                "description": orig_desc, # Use original description
                "category": orig_cat,     # Use original category
                "bbox": matched_bbox,
                "score": 1.0, # Keep score (from grounding, not matching score)
                "matched_phrase": valid_detections[best_match_idx]['label'] # Record matched phrase
            })
            matched_bbox_indices.add(best_match_idx)
            processed_original_indices.add(i)
            # print(f"  + Exact match: '{orig_desc}' -> bbox {matched_bbox}") # Debug

    # Then perform containment matching (if there are unmatched original objects and unused bboxes)
    # And only match if description contains phrase
    for i, orig_obj in enumerate(original_objects_data):
        if i in processed_original_indices: continue # Skip processed ones

        orig_desc = orig_obj["description"]
        orig_cat = orig_obj["category"]

        best_match_idx = -1
        # Find phrases with containment relationship
        for j, det in enumerate(valid_detections):
            if j in matched_bbox_indices: continue # Skip used bbox

            # Check if original description contains detected phrase (and phrase is not empty)
            if det['label'] and det['label'] in orig_desc:
                # More logic can be added here, e.g., choosing the longest matching phrase, or other heuristics
                # Currently simple processing: use the first containing one found
                best_match_idx = j
                break

        if best_match_idx != -1:
            matched_bbox = valid_detections[best_match_idx]['bbox']
            final_bbox_results["detections"].append({
                "description": orig_desc, # Use original description
                "category": orig_cat,     # Use original category
                "bbox": matched_bbox,
                "score": 1.0, # Keep score
                "matched_phrase": valid_detections[best_match_idx]['label'] # Record matched phrase
            })
            matched_bbox_indices.add(best_match_idx)
            processed_original_indices.add(i)
            # print(f"  + Containment match: '{orig_desc}' (contains '{valid_detections[best_match_idx]['label']}') -> bbox {matched_bbox}") # Debug


    print(f"Finally matched {len(final_bbox_results['detections'])} objects and their bounding boxes.")

    # --- Execute Visualization (Using description from final match results) ---
    # Convert PIL image to OpenCV format (NumPy array, RGB)
    img_rgb_np = np.array(image_pil)
    # Note: visualize_bbox needs a list containing 'bbox' and 'label'
    # We need to rebuild this list from final_bbox_results using description as label
    vis_detections = []
    if final_bbox_results["detections"]: # Check if there are detection results
        for res in final_bbox_results["detections"]:
            vis_detections.append({
                "bbox": res["bbox"],
                "label": res["description"] # Use original description for visualization
            })

    if vis_detections: # Visualize only if there are final results
        img_with_boxes = visualize_bbox(img_rgb_np, vis_detections)

        # Save result image (OpenCV requires BGR format)
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        try:
            cv2.imwrite(output_img_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
            print(f"Detection result image saved to: {output_img_path}")
        except Exception as e:
            print(f"Error: Failed to save visualization image: {e}")
            pass # Try saving JSON even if visualization failed
    else:
        print("No final matching results, visualization image not generated.")


    # --- Save final bounding box info to JSON ---
    # Save final_bbox_results, containing original description and category

    # Check if there are final detection results
    if not final_bbox_results["detections"]:
        print("No objects matched finally, JSON file not saved.")
        # Note: Empty JSON file is not created here
    else:
        # Save JSON only if there are final results
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(bbox_output_path), exist_ok=True)
            with open(bbox_output_path, 'w', encoding='utf-8') as f:
                json.dump(final_bbox_results, f, indent=2, ensure_ascii=False)
            print(f"Final bounding box info saved to: {bbox_output_path}")
        except Exception as e:
            print(f"Error: Failed to save final JSON file: {e}")
            # JSON save failure counts as processing failure, must return False
            print(f"--- Processing failed (JSON save error): Scene={scene_id}, View={view_id} ---")
            return False

    # Consider processing successful regardless of JSON save (as long as no error occurred)
    print(f"--- Processing complete: Scene={scene_id}, View={view_id} ---")
    return True

# --- Sequential Processing Function ---
def process_all_tasks(regenerate_existing=False):
    # Model initialized globally
    print("\n=== Sequential Processing Started ===")

    print("Creating dataset...")
    dataset = SceneViewDataset(base_data_path, image_base_path, regenerate_existing)

    if len(dataset) == 0:
        print("No tasks to process, program exiting.")
        return

    # Record start time
    start_time = time.time()

    # Process each task sequentially
    completed_count = 0
    failed_count = 0
    total_tasks = len(dataset)

    # Iterate through dataset directly
    for idx in range(total_tasks):
        item = dataset[idx]
        
        success = process_single_item_florence2(item, florence2_model, florence2_processor)
        if success:
            completed_count += 1
        else:
            failed_count += 1

        # Print progress
        processed_so_far = completed_count + failed_count
        if processed_so_far % 10 == 0 or processed_so_far == total_tasks: # Print every 10 items or at the last one
            elapsed = time.time() - start_time
            avg_time = elapsed / processed_so_far if processed_so_far > 0 else 0
            print(f"Progress: {processed_so_far}/{total_tasks} | Success: {completed_count} | Failed: {failed_count} | Elapsed: {elapsed:.2f}s | Avg: {avg_time:.2f}s/item")

    # Calculate total elapsed time
    end_time = time.time()
    total_elapsed_time = end_time - start_time
    print("\n=== Sequential Processing Complete ===")
    print(f"Total tasks: {total_tasks}")
    print(f"Successfully processed: {completed_count}")
    print(f"Failed/Skipped: {failed_count}")
    print(f"Total elapsed time: {total_elapsed_time:.2f} seconds")
    if completed_count > 0:
        print(f"Average processing time: {total_elapsed_time / (completed_count + failed_count):.2f} seconds/item") # Based on all attempted items

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phrase Grounding using Florence-2 (Sequential Processing)')
    parser.add_argument('--regenerate', default=False, action='store_true', help='Regenerate existing result files (skips by default)')

    args = parser.parse_args()

    print(f"Starting sequential processing")
    print(f"Regenerate existing results: {args.regenerate}")

    process_all_tasks(regenerate_existing=args.regenerate)