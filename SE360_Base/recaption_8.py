# --- Import statements and other functions remain the same ---
import argparse
import os
import torch
import numpy as np
import cv2  # Using cv2 for image loading as e2p might prefer BGR numpy arrays
from PIL import Image
import json
import glob
from tqdm import tqdm
import time
import re # For parsing VLM output and filenames

# --- Import py360 ---
try:
    from py360.e2p import e2p
except ImportError:
    print("Error: py360 library not found. Please install it: pip install py360")
    exit(1)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset, DataLoader

# --- Constants ---
PANO_IMAGE_DIR_TEMPLATE = "{pano_base_dir}/{scene}/matterport_stitched_images/{view}.png"

# --- Dataset Class ---
class DetectionJsonDataset(Dataset):
    """Loads tasks based on existing detection JSON files."""
    def __init__(self, json_base_dir, pano_base_dir, regenerate=False):
        self.json_base_dir = json_base_dir
        self.pano_base_dir = pano_base_dir
        self.regenerate = regenerate
        self.tasks = self._collect_tasks()
        print(f"Dataset initialization complete. Found {len(self.tasks)} JSON file tasks.")

    def _collect_tasks(self):
        """Collect all detection JSON files to be processed."""
        tasks = []
        print(f"Starting search for detection JSON files in base directory: {self.json_base_dir}")
        search_pattern = os.path.join(self.json_base_dir, '**', '*_detection.json')
        json_files = glob.glob(search_pattern, recursive=True)
        print(f"Found {len(json_files)} potential JSON files via glob.")

        processed_files_count = 0
        skipped_files_count = 0
        already_recaptioned_count = 0

        for json_path in tqdm(json_files, desc="Checking JSON files"):
            # Ensure it's an object detection file, not the main scene detection etc.
            if not re.search(r'_obj\d+_detection\.json$', os.path.basename(json_path)):
                 skipped_files_count += 1
                 continue

            try:
                # --- Path Parsing Logic (Remains the same) ---
                # Attempt to extract scene and view robustly
                parts = json_path.split(os.sep)
                view_obj_part = parts[-2] # e.g., 'uuid_obj0'
                view = view_obj_part.split('_obj')[0]

                # Search backwards for the structure /scene/matterport_stitched_images/view/carries_results/view_objX/...
                view_dir_index = -1
                for i in range(len(parts) - 3, 0, -1): # Start checking from potential 'view' directory index
                    # Check if parts[i] is the view folder name and it's preceded by scene/matterport_stitched_images
                    if (parts[i] == view and i > 1 and
                        parts[i+1] == "carries_results" and
                        parts[i-1] == "matterport_stitched_images"):
                        view_dir_index = i
                        break # Found the structure

                if view_dir_index != -1:
                    scene = parts[view_dir_index - 2] # scene is two levels up from view
                else:
                    # Fallback attempt: Find 'carries_results' and assume structure
                    try:
                        carries_index = parts.index("carries_results")
                        # Check if the assumed structure around carries_results is plausible
                        if (carries_index >= 3 and
                            parts[carries_index-1] == view and
                            parts[carries_index-2] == "matterport_stitched_images"):
                             scene = parts[carries_index - 3]
                        else:
                             # print(f"Warning: Path structure mismatch (Fallback failed): {json_path}")
                             skipped_files_count += 1
                             continue
                    except ValueError:
                        # print(f"Warning: 'carries_results' not found in path: {json_path}")
                        skipped_files_count += 1
                        continue

                pano_path = PANO_IMAGE_DIR_TEMPLATE.format(
                    pano_base_dir=self.pano_base_dir,
                    scene=scene,
                    view=view
                )

                if not os.path.exists(pano_path):
                    # print(f"Warning: Corresponding panorama not found, skipping task: {pano_path} for json {json_path}")
                    skipped_files_count += 1
                    continue

                recaption_filename = os.path.basename(json_path).replace("_detection.json", "_recaption.json")
                recaption_path = os.path.join(os.path.dirname(json_path), recaption_filename)

                if not self.regenerate and os.path.exists(recaption_path):
                    already_recaptioned_count += 1
                    # Skip adding to tasks if not regenerating
                    continue # Don't increment skipped_files_count here, already counted

                tasks.append({
                    "json_path": json_path,
                    "pano_path": pano_path,
                    "scene": scene,
                    "view": view,
                    "recaption_output_path": recaption_path
                })
                processed_files_count += 1 # Counts files added to tasks list

            except IndexError:
                print(f"Warning: Error parsing path (Index Error, likely structure mismatch): {json_path}")
                skipped_files_count += 1
            except Exception as e:
                print(f"Unknown error collecting task for {json_path}: {e}")
                skipped_files_count += 1

        tasks.sort(key=lambda x: x['json_path']) # Sort for potentially more consistent processing order

        print(f"Total collected {len(tasks)} tasks to process (Skipped {skipped_files_count}, of which {already_recaptioned_count} already have recaption files and --regenerate was not set)")
        if tasks:
            print("Example tasks:")
            for i in range(min(3, len(tasks))):
                print(f"  Detection JSON: {tasks[i]['json_path']}")
                print(f"  Pano: {tasks[i]['pano_path']}")
                print(f"  Output Recaption JSON: {tasks[i]['recaption_output_path']}")
        elif processed_files_count == 0 and already_recaptioned_count == 0 and len(json_files) > 0:
             print(f"Warning: Found {len(json_files)} *_detection.json files, but none matched the processing criteria or path parsing failed. Please check path structure and PANO_IMAGE_DIR_TEMPLATE.")
        elif not tasks and already_recaptioned_count > 0:
            print(f"Warning: No tasks collected to process, but found {already_recaptioned_count} existing recaption files. Use --regenerate to overwrite them.")
        elif not tasks:
             print("Warning: No tasks collected to process. Please check json_base_dir and pano_base_dir.")


        return tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]


# --- VLM Helper Function (Not used in batch processing, but kept for potential single use) ---
# <<< NOTE: This function would ALSO need updating if used independently >>>
def get_refined_description_from_vlm(perspective_pil, original_category, original_description, model, processor):
    """Generates a refined description using the VLM. (Needs update for dual output)"""
    # THIS PROMPT WOULD NEED UPDATING TOO IF THIS FUNCTION IS USED
    vlm_prompt = f"""
Analyze the provided perspective IMAGE focusing on the object identified as "{original_category}".
Also consider the Original Description provided ONLY for its location context: "{original_description}"

Your goal is to generate a new, simple single-sentence description of the "{original_category}".

Follow these critical steps:
1. Describe the visual appearance (color, shape, material, anything *on* it) based *strictly and ONLY* on what you see in the perspective IMAGE.
2. Completely DISREGARD the appearance details mentioned in the Original Description. DO NOT use its visual words for appearance.
3. Extract the spatial location context (e.g., "on the wall", "next to the sofa", "above the table") *strictly and ONLY* from the Original Description. Do not guess the location from the limited perspective image.
4. Combine the visual details from the IMAGE (Step 1) and the location context from the Original Description (Step 3) into one fluent sentence.

--- Examples ---
Example 1: Input Category: wooden table; Input Original Description (for location): a wooden table surrounded by chairs, centerly located in the room. Output Simple Description: a wooden table with a vase with flowers, surrounded by chairs, centerly located in the room.
Example 2: Input Category: blue armchair; Input Original Description (for location): a comfortable blue armchair situated by the window. Output Simple Description: a blue armchair with a knitted yellow blanket draped over its back, situated by the window.
Example 3: Input Category: metal desk lamp; Input Original Description (for location): a metal desk lamp positioned on the left side of the image. Output Simple Description: a silver metal desk lamp, positioned on the left side of the image.
--- End Examples ---

Now, based on the IMAGE provided and the specific inputs below, generate the new description:
Input Category: "{original_category}"
Input Original Description (for location): "{original_description}"

Output ONLY the single combined sentence. No extra text or prefixes. No more than 20 words.
"""
    print("WARNING: get_refined_description_from_vlm function prompt not updated for dual output yet.")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": perspective_pil},
                {"type": "text", "text": vlm_prompt}
            ]
        }
    ]
    # ... rest of the function remains the same for now ...
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
    ).to(model.device)
    try:
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=200) # Increased max tokens slightly for two outputs
        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # --- Parsing logic here would need significant update ---
        refined_description = output_text.strip()
        # ... (parsing logic for two descriptions needed here) ...
        return refined_description # And return both descriptions
    except Exception as e:
        print(f"VLM inference failed for single item: {e}")
        return None # Or (None, None)


# --- Modified Batch Processing Function ---
# <<< MAJOR MODIFICATIONS FOR VLM PROMPT, PARSING, AND SAVING FORMAT >>>
def process_batch_refinement(batch_items, model, processor):
    """
    Processes a batch of JSON refinement tasks.
    Loads original detection JSON, gets *two* descriptions (standard refined, simple)
    from VLM, updates/adds these to the original data structure, and saves
    it to a new recaption JSON file with the identical overall format but updated/new fields.
    """
    processed_count_in_batch = 0
    batch_perspective_pil = []
    batch_vlm_call_params = []
    batch_task_info = [] # Store task info (paths, original data, etc.)

    # 1. Prepare batch data (Load JSON, Load Pano, Project, Prepare VLM Input)
    for item in batch_items:
        json_path = item['json_path']
        pano_path = item['pano_path']
        recaption_output_path = item['recaption_output_path']

        try:
            # Load Original JSON data
            with open(json_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)

            params = original_data.get("perspective_transform_params")
            if not params:
                print(f"Warning: perspective_transform_params not found in {json_path}, skipping.")
                continue

            center_u = params.get("center_u_deg")
            center_v = params.get("center_v_deg")
            hfov = params.get("final_hfov_deg")
            vfov = params.get("final_vfov_deg", hfov) # Use default if needed
            height = params.get("output_height", 1024) # Use default if needed
            width = params.get("output_width", 1024) # Use default if needed

            if "original_description" not in original_data:
                 print(f"Warning: Input JSON {json_path} is missing 'original_description' key, skipping.")
                 continue
            original_description = original_data["original_description"]
            original_category = original_data.get("original_category", "object")

            if None in [center_u, center_v, hfov]:
                 print(f"Warning: Missing necessary projection parameters (u, v, hfov) in {json_path}, skipping.")
                 continue

            erp_image_bgr = cv2.imread(pano_path)
            if erp_image_bgr is None:
                print(f"Warning: Could not load panorama {pano_path} with cv2, skipping.")
                continue

            perspective_view_bgr = e2p(
                erp_image_bgr,
                fov_deg=hfov,
                u_deg=center_u,
                v_deg=center_v,
                out_hw=(height, width),
                mode='bilinear'
            )

            perspective_view_rgb = cv2.cvtColor(perspective_view_bgr, cv2.COLOR_BGR2RGB)
            perspective_pil = Image.fromarray(perspective_view_rgb)

            batch_perspective_pil.append(perspective_pil)
            batch_vlm_call_params.append({
                "original_category": original_category,
                "original_description": original_description
            })
            batch_task_info.append({
                "json_path": json_path,
                "recaption_output_path": recaption_output_path,
                "original_data": original_data
            })

        except FileNotFoundError:
            print(f"Error: File not found {json_path} or {pano_path}, skipping.")
        except json.JSONDecodeError:
             print(f"Error: Failed to parse JSON file {json_path}, skipping.")
        except KeyError as e:
             print(f"Error: Error accessing original data {json_path}, key {e}, skipping.")
        except Exception as e:
            print(f"Unexpected error preparing task {os.path.basename(json_path)}: {e}")

    if not batch_perspective_pil:
        return 0

    # 2. Prepare VLM Batch Input Messages
    # <<< MODIFIED VLM PROMPT >>>
    batch_messages = []
    for img, params in zip(batch_perspective_pil, batch_vlm_call_params):
        # --- NEW VLM PROMPT REQUESTING TWO OUTPUTS ---
        vlm_prompt = f"""
Analyze the provided perspective IMAGE focusing on the object identified as "{params['original_category']}".
Use the Original Description ONLY for its location context: "{params['original_description']}"

Your task is to generate TWO descriptions based on the IMAGE and the location context:

1.  **Standard Refined Description:**
    * Describe the visual appearance (color, shape, material, state, anything *on* it) based *strictly and ONLY* on what you see in the perspective IMAGE.
    * Completely DISREGARD the appearance details mentioned in the Original Description.
    * Extract the spatial location context (e.g., "on the wall", "next to the sofa", "above the table") *strictly and ONLY* from the Original Description. Do not guess location from the image.
    * Combine the visual details from the IMAGE and the location context from the Original Description into ONE fluent sentence (around 10-25 words).

2.  **Simple Description:**
    * Generate an EXTREMELY simple description (max 5-7 words).
    * State only the main color + object category + basic location derived from the Original Description's context.
    * Examples of desired simple output: "A blue armchair by the window.", "A silver lamp on the desk.", "A wooden table in the center."

--- Output Format ---
Provide your response EXACTLY in this format, with each description on a new line:

Standard Description: [Your standard refined description here]
Simple Description: [Your extremely simple description here]

--- Examples based on Input ---
Input Category: wooden table; Input Original Description (for location): a wooden table surrounded by chairs, centerly located in the room.
Output:
Standard Description: A light brown wooden table with a white vase containing pink flowers on it, surrounded by chairs in the center of the room.
Simple Description: A wooden table in the center.

Input Category: blue armchair; Input Original Description (for location): a comfortable blue armchair situated by the window.
Output:
Standard Description: A blue fabric armchair with a yellow knitted blanket draped over its back, situated by the window.
Simple Description: A blue armchair by the window.

Input Category: metal desk lamp; Input Original Description (for location): a metal desk lamp positioned on the left side of the desk.
Output:
Standard Description: A sleek silver metal desk lamp with an adjustable arm, positioned on the left side of the desk.
Simple Description: A silver lamp on the desk.
--- End Examples ---

Now, generate the two descriptions for:
Input Category: "{params['original_category']}"
Input Original Description (for location): "{params['original_description']}"

Output ONLY using the specified format. No extra text before or after.
"""
        messages = [
             {
                 "role": "user",
                 "content": [
                     {"type": "image", "image": img},
                     {"type": "text", "text": vlm_prompt}
                 ]
             }
         ]
        batch_messages.append(messages)

    # 3. Tokenize and prepare batch for model (Increase max_new_tokens slightly)
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in batch_messages
    ]
    if not texts: return 0
    try:
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=texts, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)
    except Exception as e:
        print(f"Error during VLM input processing: {e}")
        return 0

    # 4. Execute VLM Batch Inference
    try:
        with torch.no_grad():
            # Increased max_new_tokens slightly to accommodate two descriptions + labels
            generated_ids = model.generate(**inputs, max_new_tokens=250)
        generated_ids_trimmed = [
             out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    except Exception as e:
        print(f"VLM batch inference failed: {e}")
        return 0

    # 5. Process and Save Each Result
    # <<< MODIFICATION START: Parse two descriptions and save both >>>
    for i, raw_output_text in enumerate(output_texts):
        if i >= len(batch_task_info):
             print(f"Warning: Output index {i} out of batch task info range ({len(batch_task_info)}), skipping.")
             continue

        task_info = batch_task_info[i]
        original_json_path = task_info["json_path"]
        recaption_output_path = task_info["recaption_output_path"]
        output_data = task_info["original_data"].copy() # Work on a copy

        # --- Parse VLM output for BOTH descriptions ---
        refined_description = None
        simple_description = None
        original_desc_fallback = output_data.get("original_description", "") # Fallback if parsing fails

        try:
            lines = raw_output_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("Standard Description:"):
                    refined_description = line.replace("Standard Description:", "").strip()
                elif line.startswith("Simple Description:"):
                    simple_description = line.replace("Simple Description:", "").strip()

            # Basic validation and fallbacks
            if not refined_description:
                print(f"Warning: Failed to parse 'Standard Description' from VLM output for {os.path.basename(original_json_path)}. Raw: '{raw_output_text}'")
                # Attempt to grab the first non-empty line if standard is missing but simple might be there or vice-versa
                if lines and lines[0] and not lines[0].startswith("Simple Description:"):
                    refined_description = lines[0].strip() # Best guess
                else:
                    refined_description = original_desc_fallback # Use original as last resort
                    print(f"  -> Using original description as fallback for Standard Description.")

            if not simple_description:
                print(f"Warning: Failed to parse 'Simple Description' from VLM output for {os.path.basename(original_json_path)}. Raw: '{raw_output_text}'")
                # Attempt to grab the second non-empty line if simple is missing
                if len(lines) > 1 and lines[1] and not lines[1].startswith("Standard Description:"):
                    simple_description = lines[1].strip() # Best guess
                else:
                    simple_description = "" # Use empty string as fallback for simple
                    print(f"  -> Setting Simple Description to empty string.")

        except Exception as parse_err:
            print(f"Error parsing VLM output for {os.path.basename(original_json_path)}: {parse_err}. Raw: '{raw_output_text}'")
            refined_description = original_desc_fallback # Fallback to original
            simple_description = "" # Fallback to empty

        # --- >>> Update/Add the fields in the output dictionary <<< ---
        output_data["original_description"] = refined_description # Update existing key
        output_data["simple_original_description"] = simple_description # Add new key

        # --- Save the modified JSON data ---
        try:
            os.makedirs(os.path.dirname(recaption_output_path), exist_ok=True)
            with open(recaption_output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            processed_count_in_batch += 1
        except Exception as e:
            print(f"Error saving new recaption JSON file {recaption_output_path}: {e}")
    # <<< MODIFICATION END >>>

    return processed_count_in_batch


# --- Main Function (No changes needed from previous version) ---
def main():
    parser = argparse.ArgumentParser(description="Batch process detection JSONs using VLM to generate standard and simple descriptions, and save to new recaption JSON files (preserving original format, adding new fields)") # Updated description
    parser.add_argument(
        "--json_base_dir",
        type=str,
        default="./data/Matterport3D/mp3d_base/",
        help="Base JSON root directory containing scene/view/object subdirectories (looks for _detection.json)"
    )
    parser.add_argument(
        "--pano_base_dir",
        type=str,
        default="./data/Matterport3D/mp3d_skybox/",
        help="Base panorama root directory containing scene subdirectories"
    )
    parser.add_argument(
        "--vlm_model_path",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="VLM model path (Hugging Face)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Hugging Face model and processor cache directory"
    )
    parser.add_argument('--batch_size', type=int, default=7, help='Batch size (adjust according to VRAM)')
    parser.add_argument('--num_workers', type=int, default=7, help='Number of data loading worker processes (adjust according to CPU/Memory)')
    parser.add_argument(
        '--regenerate',
        action='store_true',
        default=False,
        help='If set, regenerates and overwrites existing _recaption.json files'
    )

    args = parser.parse_args()

    # --- Model and Processor Loading ---
    print("Loading VLM model and processor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("Warning: Running on CPU. This will be slow. Forcing batch_size=1.")
        args.batch_size = 1
        torch_dtype = torch.float32
        attn_implementation = None
        device_map = "cpu"
    else:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"CUDA Device: {gpu_name}")
        if torch.cuda.is_bf16_supported():
             torch_dtype=torch.bfloat16
             print("GPU supports bfloat16, using torch.bfloat16")
        else:
             print("Warning: Current GPU does not support bfloat16, using torch.float16.")
             torch_dtype=torch.float16

        # Try flash attention 2 if available and on CUDA
        attn_implementation = "flash_attention_2"
        device_map = "auto" # Let HF handle device mapping for multi-GPU or large models

    try:
        vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.vlm_model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation if attn_implementation and device=="cuda" else None,
            device_map=device_map,
            cache_dir=args.cache_dir,
        )
        processor = AutoProcessor.from_pretrained(
            args.vlm_model_path, cache_dir=args.cache_dir, use_fast=True
        )
        processor.tokenizer.padding_side = 'left' # Important for batch generation
        if processor.tokenizer.pad_token is None:
             processor.tokenizer.pad_token = processor.tokenizer.eos_token
             print("Tokenizer pad_token set to eos_token.")
        print("Model and processor loaded successfully.")
        if attn_implementation and device=="cuda":
            print("Using Flash Attention 2.")
        elif device=="cuda":
            print("Flash Attention 2 not used (possibly not installed or supported).")

    except ImportError as e:
         print(f"Failed to load model or processor (possibly missing flash-attn): {e}")
         print("If you want to use Flash Attention 2, please install: pip install flash-attn")
         print("Attempting to load without Flash Attention...")
         try:
              attn_implementation = None # Force disable
              vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                  args.vlm_model_path, torch_dtype=torch_dtype, attn_implementation=None,
                  device_map=device_map, cache_dir=args.cache_dir,
              )
              processor = AutoProcessor.from_pretrained(
                  args.vlm_model_path, cache_dir=args.cache_dir, use_fast=True
              )
              processor.tokenizer.padding_side = 'left'
              if processor.tokenizer.pad_token is None:
                   processor.tokenizer.pad_token = processor.tokenizer.eos_token
              print("Model loaded successfully (without Flash Attention)")
         except Exception as e2:
              print(f"Failed to load even without Flash Attention: {e2}")
              import traceback
              traceback.print_exc()
              return
    except Exception as e:
        print(f"Failed to load model or processor: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Dataset and DataLoader ---
    print("Creating dataset...")
    dataset = DetectionJsonDataset(args.json_base_dir, args.pano_base_dir, args.regenerate)

    if len(dataset) == 0:
        print("No tasks found to process. Please check input directories and --regenerate flag (if expecting to process existing files).")
        return

    print(f"Creating DataLoader (Batch Size: {args.batch_size}, Workers: {args.num_workers})...")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False, # Keep order for easier debugging if needed
        num_workers=args.num_workers,
        collate_fn=lambda x: x, # Custom collate because we process items individually after batch VLM
        pin_memory=(device == "cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    # --- Batch Processing Loop ---
    print("Starting batch processing...")
    total_processed_count = 0
    total_tasks_in_dataloader = len(dataset) # Use dataset length as total target
    start_time = time.time()

    process_times = []

    for batch in tqdm(dataloader, desc="Processing Batches", total=len(dataloader)):
        if not batch: # Should not happen with standard dataloader unless dataset is empty
            continue
        batch_start_time = time.time()
        processed_in_batch = process_batch_refinement(
            batch,
            vlm_model,
            processor
        )
        batch_end_time = time.time()
        total_processed_count += processed_in_batch
        if processed_in_batch > 0:
             process_times.append(batch_end_time - batch_start_time)


    end_time = time.time()
    elapsed_time = end_time - start_time
    avg_batch_time = sum(process_times) / len(process_times) if process_times else 0
    items_per_sec = total_processed_count / elapsed_time if elapsed_time > 0 else 0


    print("\n--------------------")
    print("Batch processing complete!")
    print(f"Total generated/updated {total_processed_count} / {total_tasks_in_dataloader} recaption JSON file tasks.")
    print(f"Total time: {elapsed_time:.2f} seconds")
    if process_times:
         print(f"Average batch processing time: {avg_batch_time:.3f} seconds (based on {len(process_times)} non-empty batches)")
    if total_processed_count > 0:
         print(f"Average processing speed: {items_per_sec:.3f} JSONs/sec (based on successfully processed)")
         print(f"Average processing time: {elapsed_time / total_processed_count:.3f} sec/JSON (based on successfully processed)")
    if total_tasks_in_dataloader > 0:
         print(f"Overall processing speed (based on dataloader tasks): {elapsed_time / total_tasks_in_dataloader:.3f} sec/task")

    print("--------------------")


if __name__ == "__main__":
    main()