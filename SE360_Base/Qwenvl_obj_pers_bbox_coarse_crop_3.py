from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image, ImageDraw, ImageFont
import json
import re
import os
import sys
import numpy as np
import glob
from pathlib import Path
import time
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- Global Settings ---
# device = "cuda:0" if torch.cuda.is_available() else "cpu" # model loading uses device_map="auto"

# --- Model Loading (Load only once) ---
print("Loading Qwen-VL model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", # Qwen/Qwen2.5-VL-32B-Instruct
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True) # Qwen/Qwen2.5-VL-32B-Instruct
# Set padding_side to support batch generation
processor.tokenizer.padding_side = 'left'
print("Model loading complete.")

# --- New: Dataset Class ---
class BoundingBoxDataset(Dataset):
    """Dataset for loading bounding box detection tasks"""
    def __init__(self, base_data_path, image_base_path, regenerate_existing=False):
        self.base_data_path = base_data_path
        self.image_base_path = image_base_path
        self.regenerate_existing = regenerate_existing
        self.tasks = self._collect_tasks()
        print(f"Dataset initialization complete. Found {len(self.tasks)} tasks.")

    def _collect_tasks(self):
        """Scan directories to collect processing tasks"""
        tasks = []
        print(f"Scanning tasks at: {self.base_data_path}")
        print(f"Image base path: {self.image_base_path}")

        scene_paths = glob.glob(os.path.join(self.base_data_path, "*"))
        for scene_path in scene_paths:
            if not os.path.isdir(scene_path): continue
            scene_id = os.path.basename(scene_path)

            view_paths = glob.glob(os.path.join(scene_path, "matterport_stitched_images", "*"))
            for view_path in view_paths:
                if not os.path.isdir(view_path): continue
                view_id = os.path.basename(view_path)

                json_file_path = os.path.join(view_path, "description", "objects.json")
                if not os.path.exists(json_file_path): continue

                image_path = os.path.join(self.image_base_path, scene_id, "matterport_stitched_images", f"{view_id}.png")
                if not os.path.exists(image_path): continue

                output_save_dir = os.path.join(self.base_data_path, scene_id, "matterport_stitched_images", view_id)
                bbox_output_path = os.path.join(output_save_dir, "bbox_qwen.json")
                output_img_path = os.path.join(output_save_dir, "detection_result_qwen.jpg")

                if not self.regenerate_existing and os.path.exists(bbox_output_path) and os.path.exists(output_img_path):
                    continue

                tasks.append({
                    "scene_id": scene_id,
                    "view_id": view_id,
                    "image_path": image_path,
                    "bbox_json_path": json_file_path,
                    "output_save_dir": output_save_dir, # Pass output directory
                    "marked_image_path": output_img_path,
                    "output_json_path": bbox_output_path,
                    "base_data_path": self.base_data_path # Keep for future use
                })

        print(f"Collected {len(tasks)} processing tasks.")
        if tasks:
            print("Example task paths:")
            for i in range(min(3, len(tasks))):
                print(f"  Image: {tasks[i]['image_path']}")
                print(f"  Input JSON: {tasks[i]['bbox_json_path']}")
                print(f"  Output JSON: {tasks[i]['output_json_path']}")
        else:
             print("Warning: No tasks collected. Please check directory structure and regenerate_existing settings.")

        return tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]

# --- Modified: Batch VLM Function ---
def process_batch_bboxes(batch_items, model, processor, no_visualization=False):
    """Batch process bounding box detection tasks"""
    batch_pil_images = []
    batch_prompts = []
    batch_metadata = [] # Store metadata for each item for later use
    batch_original_objects = [] # New: Store original object info (description -> category mapping) for each task
    processed_count_in_batch = 0

    # 1. Prepare batch data (Load images, JSON, build prompts)
    for item in batch_items:
        image_path = item['image_path']
        bbox_json_path = item['bbox_json_path']
        output_save_dir = item['output_save_dir']
        os.makedirs(output_save_dir, exist_ok=True) # Ensure output directory exists

        # Load Image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error: Cannot load image {image_path}: {e}")
            continue # Skip this corrupted image

        # Load Input JSON (objects.json)
        try:
            with open(bbox_json_path, 'r', encoding='utf-8') as f:
                bbox_data = json.load(f)
        except Exception as e:
            print(f"Error: Cannot load or parse input JSON {bbox_json_path}: {e}")
            continue # Skip this task

        # Check bbox_data type and get detections list
        detections_list = []
        if isinstance(bbox_data, dict):
            detections_list = bbox_data.get("detections", [])
        elif isinstance(bbox_data, list):
            detections_list = bbox_data # If it's a list directly, use it
        else:
            print(f"Warning: Input JSON {bbox_json_path} unknown format (neither dict nor list), skipping.")
            continue

        # --- Modified: Extract object descriptions and original categories, enforce 'description' and 'category' existence ---
        target_objects_info = {} # Use dictionary to store description -> category mapping
        target_desc_cat_pairs_for_prompt = [] # List of (description, category) pairs for prompt building
        skipped_detections_count = 0
        for detection in detections_list:
            # Ensure detection is a dictionary
            if not isinstance(detection, dict):
                print(f"Warning: Found non-dict detection entry in {bbox_json_path}, skipping: {detection}")
                skipped_detections_count += 1
                continue

            desc = detection.get("description")
            cat = detection.get("category") # Get original category

            if desc and cat: # Both description and category must exist
                # If description exists, choose to skip or overwrite (assuming unique descriptions here)
                if desc not in target_objects_info:
                    target_objects_info[desc] = cat
                    target_desc_cat_pairs_for_prompt.append({"description": desc, "category": cat}) # Store dictionary pair
                else:
                    # If description is duplicated, log warning
                    print(f"Warning: Duplicate description '{desc}' found in {bbox_json_path}, using first encountered category '{target_objects_info[desc]}'")
            else: # If description or category does not exist, skip
                skipped_detections_count += 1
                continue
        # --- End Modification ---

        if skipped_detections_count > 0:
             print(f"Warning: Skipped {skipped_detections_count} entries missing 'description' or 'category' in {bbox_json_path}.")

        if not target_desc_cat_pairs_for_prompt:
            print(f"Warning: Cannot extract valid 'description' and 'category' pairs from {bbox_json_path}, skipping {image_path}")
            continue # Skip tasks without valid descriptions and categories

        # --- Modified: Build VLM prompt containing categories and descriptions ---
        descriptions_categories_text = "\n".join([f'- description: "{pair["description"]}", category: "{pair["category"]}"' for pair in target_desc_cat_pairs_for_prompt])

        prompt_text = f'''
        Please locate the objects in the image based on the following descriptions and their associated categories. For each object found, return its bounding box coordinates (as a list [x1, y1, x2, y2]) along with the original description and category provided.

        Descriptions and Categories:
        {descriptions_categories_text}

        For each description-category pair provided above, find the corresponding object in the image and return its bounding box.

        ###
        # Required Output Format:
        The output MUST be a single JSON list enclosed in ```json ... ```.
        Each element in the list MUST be a JSON object.
        Each JSON object MUST contain EXACTLY THREE keys: "description", "category", and "bbox_2d".
        - The value for "description" MUST be one of the original descriptions provided in the input.
        - The value for "category" MUST be the category associated with that description in the input.
        - The value for "bbox_2d" MUST be a list containing EXACTLY FOUR numbers (integer or float): [x1, y1, x2, y2].
        NO OTHER FORMATS ARE ACCEPTABLE. Do NOT put coordinates in a string. Do NOT omit keys.

        Output format example:
        ```json
        [
         {{"description": "description_1", "category": "category_name_1", "bbox_2d": [x1, y1, x2, y2]}},
         {{"description": "description_2", "category": "category_name_2", "bbox_2d": [x1, y1, x2, y2]}},
         ...
        ]
        ```
        ###
        # Important Rules:
        - If you find multiple instances for a description, choose the larger one. The bbox need totally contain all the object in description. Use the original description and category.
        - If you cannot confidently locate a bounding box for a specific description-category pair, OMIT that object entirely from the JSON list output. Do not include entries with null, empty, or incorrectly formatted values.
        - Ensure all coordinate values in the "bbox_2d" list are numbers.
        - Ensure the returned "description" and "category" exactly match the ones provided in the input list for the located object.
        - If the category of the description carries other objects, you need to output the category and object as a whole in a single bbox.
        ###

        Provide the JSON output containing the bounding boxes, descriptions, and categories for the located objects, strictly following the required format.
        '''
        # --- End Modification ---

        batch_pil_images.append(image)
        batch_prompts.append(prompt_text)
        batch_metadata.append(item) # Store original task info
        batch_original_objects.append(target_objects_info) # Keep original mapping to verify if VLM returned category is correct

    # Return if no valid tasks in this batch
    if not batch_pil_images:
        print("No valid tasks to process in current batch.")
        return 0

    # 2. Build VLM Batch Inputs
    batch_messages = []
    for img, prompt in zip(batch_pil_images, batch_prompts):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        batch_messages.append(messages)

    try:
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
    except Exception as e:
        print(f"Error: Failed to prepare VLM batch inputs: {e}")
        return 0 # Cannot process this batch

    # 3. Run VLM Batch Inference
    try:
        with torch.no_grad(): # Explicitly use no_grad
            generated_ids = model.generate(**inputs, max_new_tokens=4096) # May need to adjust max_new_tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    except Exception as e:
        print(f"Error: VLM batch inference failed: {e}")
        return 0 # This batch failed

    # 4. Process and Save Each Result
    for i, output_text in enumerate(output_texts):
        metadata = batch_metadata[i]
        pil_image = batch_pil_images[i] # Get corresponding original PIL image
        original_objects_info = batch_original_objects[i] # Get current task's original object info (desc -> cat mapping)
        output_json_path = metadata['output_json_path']
        marked_image_path = metadata['marked_image_path']
        image_path = metadata['image_path'] # Used for output JSON
        original_width, original_height = pil_image.size
        image_area = original_width * original_height

        print(f"\nProcessing results for: {metadata['scene_id']}/{metadata['view_id']}")
        print(f"  Raw VLM Output:\n{output_text}")

        # --- JSON Parsing Logic (Keep mostly unchanged) ---
        json_pattern = r'```json\s*(\[[\s\S]+?\])\s*```' # Find [...]
        match = re.search(json_pattern, output_text)
        parsed_list = None
        json_content_to_parse = None

        if match:
            json_content_to_parse = match.group(1)
            print("  Found ```json``` block. Attempting strict list parsing...")
            try:
                parsed_list = json.loads(json_content_to_parse)
                if not isinstance(parsed_list, list):
                    print(f"  Warning: Parsed content is not a list (in ```json``` block), type: {type(parsed_list)}")
                    parsed_list = None
                else:
                    print("  Strict JSON list parsing successful.")
            except json.JSONDecodeError as e_strict:
                print(f"  Strict JSON list parsing failed: {e_strict}")
            except Exception as e_general_strict:
                print(f"  Other error during JSON list parsing: {e_general_strict}")
        else:
            start = output_text.find('[')
            end = output_text.rfind(']')
            if start != -1 and end != -1 and start < end:
                json_content_to_parse = output_text[start:end+1]
                print(f"  ```json``` block not found, found potential JSON list structure. Attempting strict parsing: {json_content_to_parse[:100]}...")
                try:
                    parsed_list = json.loads(json_content_to_parse)
                    if not isinstance(parsed_list, list):
                        print(f"  Warning: Parsed content is not a list (no ```json``` block), type: {type(parsed_list)}")
                        parsed_list = None
                    else:
                        print("  Strict JSON list parsing successful (no ```json``` block).")
                except json.JSONDecodeError as e_strict_loose:
                     print(f"  Strict JSON list parsing failed (no ```json``` block): {e_strict_loose}")
                except Exception as e_general_loose:
                     print(f"  Other error during JSON list parsing (no ```json``` block): {e_general_loose}")
            else:
                print("  No properly formatted JSON bbox output found (no ```json``` and no []).")

        output_detections = [] # List to store final valid detection results

        # If list parsing was successful, proceed to validation
        if parsed_list and isinstance(parsed_list, list):
            print(f"  Parsed {len(parsed_list)} entries from VLM output, starting validation...")
            valid_count = 0
            for item in parsed_list:
                # --- Modified: Validation logic, requires description, category, bbox_2d ---
                if isinstance(item, dict) and all(k in item for k in ["description", "category", "bbox_2d"]):
                    desc = item["description"]
                    cat = item["category"] # Category returned by VLM
                    bbox = item["bbox_2d"]

                    # --- New: Verify if VLM returned category matches original input ---
                    original_cat = original_objects_info.get(desc)
                    if not original_cat:
                        print(f"  Warning: VLM returned description '{desc}' not found in original object info, skipping.")
                        continue
                    if cat != original_cat:
                        print(f"  Warning: VLM returned category '{cat}' does not match original category '{original_cat}' for description '{desc}', skipping.")
                        continue
                    # --- End New Validation ---

                    if isinstance(desc, str) and isinstance(cat, str) and \
                       isinstance(bbox, list) and len(bbox) == 4 and \
                       all(isinstance(coord, (int, float)) for coord in bbox):

                        # Coordinate validity check
                        x1_f, y1_f, x2_f, y2_f = bbox[0], bbox[1], bbox[2], bbox[3]
                        if x1_f > x2_f or y1_f > y2_f:
                            print(f"  Warning: Skipping invalid coordinates (x1>x2 or y1>y2) for '{desc}': [{x1_f}, {y1_f}, {x2_f}, {y2_f}]")
                            continue

                        # --- DINO-style BBox Filtering (Logic unchanged) ---
                        x1, y1, x2, y2 = int(x1_f), int(y1_f), int(x2_f), int(y2_f)

                        if x1 <= 30 or y1 <= 30 or x2 >= original_width - 30 or y2 >= original_height - 30:
                            print(f"  Warning: BBox touches image boundary, skipping: '{desc}' [{x1}, {y1}, {x2}, {y2}]")
                            continue
                        width = x2 - x1
                        height = y2 - y1
                        max_side_length = max(width, height)
                        max_allowed_length = original_height * 3 / 5
                        if max_side_length > max_allowed_length:
                            print(f"  Warning: BBox max side length ({max_side_length}) exceeds 3/5 of image height ({max_allowed_length:.0f}), skipping: '{desc}'")
                            continue
                        bbox_area = width * height
                        area_percentage = (bbox_area / image_area) * 100
                        if area_percentage < 0.3:
                            print(f"  Warning: BBox area too small ({area_percentage:.2f}%), less than 0.3% of image, skipping: '{desc}'")
                            continue
                        elif area_percentage > 40:
                            print(f"  Warning: BBox area too large ({area_percentage:.2f}%), greater than 40% of image, skipping: '{desc}'")
                            continue
                        # --- End DINO-style Filtering ---

                        # --- Modified: Validation passed, use VLM returned (verified matching) data ---
                        output_detections.append({
                            "description": desc,
                            "category": cat, # Use VLM returned and verified category
                            "bbox_2d": [x1, y1, x2, y2], # Use filtered integer coordinates
                        })
                        valid_count += 1
                    else:
                        print(f"  Skipping malformed entry (desc, cat, or bbox_2d): {item}")
                else:
                    print(f"  Skipping non-dict entry or missing keys ('description', 'category', 'bbox_2d'): {item}")
            print(f"  Extracted {valid_count} valid bounding boxes after validation, filtering, and category matching.") # Update log info
            # --- End Modification ---
        else:
             print("  Warning: Failed to parse valid JSON list from VLM output.")
             continue

        if not output_detections:
            print("  Warning: No valid bbox entries extracted from VLM output.")
            continue

        # --- Modified: Fix f-string quote issue (use .format() to avoid nesting) ---
        formatted_detections = [f"{d['description']} ({d['category']})" for d in output_detections]
        print(f"  Final extracted objects: {formatted_detections}")
        # --- End Modification ---

        # --- Draw Bounding Boxes and Prepare Output JSON (Logic unchanged) ---
        img_with_bboxes = None
        if not no_visualization:
            img_with_bboxes = pil_image.copy()
            draw = ImageDraw.Draw(img_with_bboxes)
            colors = ["red", "blue", "green", "purple", "orange", "cyan", "magenta", "yellow", "lime", "pink", "teal", "lavender", "brown", "beige", "maroon", "olive"]
            detection_colors = {i: colors[i % len(colors)] for i in range(len(output_detections))}
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except IOError: font = ImageFont.load_default()

        for idx, detection in enumerate(output_detections):
            description = detection["description"]
            category = detection["category"] # From VLM output, verified
            bbox = detection["bbox_2d"]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            if not no_visualization and img_with_bboxes:
                color = detection_colors.get(idx, "gray")
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
                display_text = f"{category}: {description[:25]}" + ('...' if len(description) > 25 else '')
                text_y = y1 - 20 if y1 >= 20 else y1 + 5
                draw.text((x1, text_y), display_text, fill=color, font=font)

        # --- Save Results (Logic unchanged) ---
        try:
            if not no_visualization and img_with_bboxes:
                img_with_bboxes.save(marked_image_path)

            output_json_data = {"image_path": image_path, "detections": output_detections}
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(output_json_data, f, indent=2, ensure_ascii=False)
            print(f"  BBox detection results saved to: {output_json_path}")
            processed_count_in_batch += 1

        except Exception as e:
            print(f"  Error: Failed to save results for {metadata['scene_id']}/{metadata['view_id']}: {e}")

    return processed_count_in_batch

# --- Main Program ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch object detection using Qwen-VL (mimicking DINO paths)')
    parser.add_argument('--base_data_path', type=str, default="./data/Matterport3D/mp3d_base", help='Base data path containing scenes and matterport_stitched_images subdirectories (for finding input JSON and saving output)')
    parser.add_argument('--image_base_path', type=str, default="./data/Matterport3D/mp3d_skybox", help='Image base path containing scenes and matterport_stitched_images image files (for loading images)')
    parser.add_argument('--regenerate', default=False, action='store_true', help='Regenerate existing result files')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (adjust based on VRAM)')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers (adjust based on CPU cores)')
    parser.add_argument('--no_visualization', action='store_true', help='Do not save visualization images with bounding boxes')

    args = parser.parse_args()

    print("Creating dataset...")
    dataset = BoundingBoxDataset(
        args.base_data_path,
        args.image_base_path,
        args.regenerate
    )

    if len(dataset) == 0:
        print("No processing tasks found. Exiting.")
        sys.exit(0)

    print(f"Creating DataLoader (Batch Size: {args.batch_size}, Workers: {args.num_workers})...")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"\nResults will be saved back to corresponding subdirectories in {args.base_data_path}")
    print(f"Regenerate existing: {args.regenerate}")

    start_time = time.time()
    total_processed_count = 0
    total_tasks = len(dataset)

    for batch in tqdm(dataloader, desc="Processing batches"):
        if not batch:
            continue
        processed_in_batch = process_batch_bboxes(
            batch,
            model,
            processor,
            args.no_visualization
        )
        total_processed_count += processed_in_batch

    elapsed_time = time.time() - start_time
    print("\n--- Processing Complete ---")
    print(f"Total tasks (from dataset): {total_tasks}")
    print(f"Tasks successfully processed and saved: {total_processed_count}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    if total_processed_count > 0:
        print(f"Average processing speed: {elapsed_time / total_processed_count:.2f} seconds/image")