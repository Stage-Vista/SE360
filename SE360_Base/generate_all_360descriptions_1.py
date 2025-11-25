import argparse
import os
import torch
import numpy as np
import cv2
from PIL import Image
import json
import glob
from tqdm import tqdm
import time  # import time module

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset, DataLoader # import Dataset and DataLoader

# --- New: Dataset Class ---
class DescriptionDataset(Dataset):
    """Dataset for loading image description tasks"""
    def __init__(self, source_dir, target_dir, regenerate_existing=False):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.regenerate_existing = regenerate_existing
        self.tasks = self._collect_tasks()
        print(f"Dataset initialized, found {len(self.tasks)} tasks.")

    def _collect_tasks(self):
        """Collect all image tasks to be processed"""
        tasks = []
        print(f"Start searching images in source directory: {self.source_dir}")
        try:
            scene_dirs = [os.path.join(self.source_dir, d) for d in os.listdir(self.source_dir)
                        if os.path.isdir(os.path.join(self.source_dir, d))]
            print(f"Found {len(scene_dirs)} scene directories")

            for scene_dir in scene_dirs:
                images_dir = os.path.join(scene_dir, "matterport_stitched_images")
                if os.path.exists(images_dir) and os.path.isdir(images_dir):
                    scene_found_files = []
                    for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
                        pattern = os.path.join(images_dir, ext)
                        files = glob.glob(pattern)
                        scene_found_files.extend(files)

                    if not scene_found_files:
                        print(f"Warning: No image files found in {images_dir}")
                        try:
                            all_files = os.listdir(images_dir)
                            print(f"File list in directory: {all_files[:10]}..." if len(all_files) > 10 else all_files)
                        except Exception as list_err:
                            print(f"Cannot list directory content: {list_err}")
                        continue # if directory is empty, skip this scene

                    print(f"Found {len(scene_found_files)} images in matterport_stitched_images folder of {scene_dir}")

                    # Create task for each found file
                    for source_file_path in scene_found_files:
                        rel_path = os.path.relpath(source_file_path, self.source_dir)
                        file_name_without_ext = os.path.splitext(os.path.basename(source_file_path))[0]
                        dir_part = os.path.dirname(rel_path)

                        description_dir = os.path.join(self.target_dir, dir_part, file_name_without_ext, "description")
                        json_path = os.path.join(description_dir, "objects.json")

                        # if not regenerate_existing and file already exists, skip adding task
                        if not self.regenerate_existing and os.path.exists(json_path):
                            # print(f"Result already exists, skip task: {rel_path}")
                            continue

                        tasks.append({
                            "source_path": source_file_path,
                            "output_json_path": json_path,
                            "description_dir": description_dir,
                            "relative_path": rel_path
                        })

        except Exception as e:
            print(f"Error finding image files: {e}")

        tasks = list({task['source_path']: task for task in tasks}.values()) # remove duplicates
        tasks.sort(key=lambda x: x['source_path']) # sort by source_path

        print(f"Total tasks to process: {len(tasks)}")
        if tasks:
            print("Example task paths:")
            for i in range(min(5, len(tasks))):
                print(f"  Source: {tasks[i]['source_path']}")
                print(f"  Target: {tasks[i]['output_json_path']}")
        else:
             print("Warning: No tasks to process. Please check source directory structure and regenerate_existing setting.")

        return tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]

# --- modified: batch process VLM function ---
def process_batch_descriptions(batch_items, model, processor, vlm_prompt):
    """Batch process image description tasks"""
    batch_pil_images = []
    batch_metadata = [] # store metadata for each item for later use
    processed_count_in_batch = 0

    # 1. prepare batch data
    for item in batch_items:
        source_path = item['source_path']
        output_json_path = item['output_json_path']
        description_dir = item['description_dir']
        relative_path = item['relative_path']

        # ensure output directory exists
        os.makedirs(description_dir, exist_ok=True)

        # load image (using PIL)
        try:
            image = Image.open(source_path).convert('RGB') # ensure it is RGB
            batch_pil_images.append(image)
            batch_metadata.append({
                "output_json_path": output_json_path,
                "source_path": source_path,
                "relative_path": relative_path
            })
        except Exception as e:
            print(f"Cannot load image {source_path}: {e}")
            continue # skip this damaged image

    # if this batch has no valid images, return
    if not batch_pil_images:
        print("No valid images to process in this batch.")
        return 0

    # 2. build VLM batch input
    batch_messages = []
    for img in batch_pil_images:
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

    # prepare inference
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in batch_messages
    ]
    # note: here directly pass PIL Image list to process_vision_info
    image_inputs, video_inputs = process_vision_info(batch_messages) # process_vision_info should be able to handle PIL list
    inputs = processor(
        text=texts,
        images=image_inputs, # Use image inputs obtained from process_vision_info
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # 3. execute VLM batch inference
    try:
        with torch.no_grad(): # explicitly use no_grad
            generated_ids = model.generate(**inputs, max_new_tokens=1000)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    except Exception as e:
        print(f"VLM batch inference failed: {e}")
        # can choose how to handle failure, e.g. return 0 or raise exception
        # here we record the error and continue processing the next batch (by returning 0)
        return 0

        # 4. process and save each result
    for i, output_text in enumerate(output_texts):
        metadata = batch_metadata[i]
        output_json_path = metadata['output_json_path']
        relative_path = metadata['relative_path']
        source_path = metadata['source_path']

        # print(f"processed result for {relative_path}")
        # print("original VLM output:", output_text) # optional debug info

        # --- reuse the previous parsing logic ---
        objects_list = []
        try:
            json_start = output_text.find('[')
            json_end = output_text.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_text = output_text[json_start:json_end]
                objects_list = json.loads(json_text)
                # print(f"Successfully parsed JSON: {len(objects_list)} objects")
            else:
                # print("JSON not found, trying code block...")
                json_block_start = output_text.find("```json")
                if json_block_start >= 0:
                    json_start = output_text.find('[', json_block_start)
                    json_block_end = output_text.find("```", json_block_start + 7)
                    if json_start >= 0 and json_block_end > json_start:
                        json_end = output_text.rfind(']', json_start, json_block_end) + 1
                        if json_end > json_start:
                            json_text = output_text[json_start:json_end]
                            objects_list = json.loads(json_text)
                            # print(f"parsed {len(objects_list)} objects from code block")

        except Exception as e:
            print(f"error parsing JSON for {relative_path}: {e}")

        if not objects_list:
            try:
                # print("trying to extract from text...")
                lines = output_text.strip().split('\n')
                current_obj = {}
                for line in lines:
                    line = line.strip()
                    if "description" in line.lower() and ":" in line:
                        description_part = line.split(":", 1)[1].strip().strip('",')
                        current_obj["description"] = description_part
                    elif "category" in line.lower() and ":" in line:
                        category_part = line.split(":", 1)[1].strip().strip('",')
                        current_obj["category"] = category_part
                        if "description" in current_obj and "category" in current_obj:
                            objects_list.append(current_obj.copy())
                            current_obj = {}
                    elif line and (line[0].isdigit() and ". " in line[:3]) or ("-" in line and len(line.split("-")) == 2):
                         try:
                            if "-" in line:
                                parts = line.split("-")
                                category = parts[0].strip()
                                if category.startswith(tuple(f"{i}." for i in range(10))): # Handle "1. category - desc"
                                    category = category[category.find(".")+1:].strip()
                                description = parts[1].strip()
                            else: # Handle "1. category - desc" without '-' case, might need more complex logic, skip for now
                                continue

                            objects_list.append({"description": description, "category": category})
                         except:
                             continue # Ignore unparsable lines
            except Exception as e:
                print(f"failed to extract information from text for {relative_path}: {e}")
        # --- parsing logic ends ---

        # save object descriptions to JSON file
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(objects_list, f, indent=4, ensure_ascii=False)
            # print(f"successfully saved descriptions to {output_json_path}")
            processed_count_in_batch += 1
        except Exception as e:
            print(f"error saving JSON file {output_json_path}: {e}")

    return processed_count_in_batch

# --- Removed old get_objects_from_vlm function ---
# def get_objects_from_vlm(...):
#     ...

# --- Removed old process_dataset function ---
# def process_dataset():
#     ...

def main():
    parser = argparse.ArgumentParser(description="use VLM to batch recognize objects in images and generate description files")
    # remove single image input/output parameters
    # parser.add_argument("--input_img", type=str, help="input image path")
    # parser.add_argument("--output_dir", type=str, help="output directory")
    # default process dataset
    # parser.add_argument("--process_dataset", default=True, action="store_true", help="whether to process dataset")

    # keep and possibly adjust default values
    parser.add_argument(
        "--source_dir",
        type=str,
        default="./data/Matterport3D/mp3d_skybox/",
        help="source image root directory containing scene subdirectories"
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="./data/Matterport3D/mp3d_base/",
        help="target root directory to save description files"
    )
    parser.add_argument(
        "--vlm_model_path",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct", #Qwen/Qwen2.5-VL-32B-Instruct
        help="VLM model path"
    )
    # keep default prompt, but allow command line override
    parser.add_argument(
        "--vlm_prompt",
        type=str,
        default="""
        Describe the main objects in the foreground of the image. Don't include background or structure elements like walls, sky, stairs, ground, door, window, pole, railing, etc. List at most 8 objects.
         Prefer to choose compeltely visible objects, no occlusion. Don't choose objects that are close to the image edge.
         Please respond using the following fixed format for easier processing:

         ```json
         [
           {
             "description": "Detailed description of the object including color, shape and relative position. Only one sentence",
             "category": "Specific object name (be precise, not general category, only one name, don't use "/" and "or", 1-2 words)"
           },
         ]
         ```

         Notes:
         1. The category name should be the specific object name (e.g., "wooden coffee table", "ceramic vase", "leather sofa"), not just its general type
         2. Each description should include color, shape/type and relative position (next to, behind, in front of, etc.).
         3. If you can't find 8 objects that meet the criteria, you can return fewer items, never choose room structure, even you only find one object.
         4. Prioritize objects in the center of the image
         5. Don't choose same objects multiple times.
         6. As some items together, you can describe them as a group. For example, some books are arranged in a row, you can describe them as a group. Some chairs around a table, you can describe the table and chairs as a group.
         7. Don't include background or structure elements like walls, sky, stairs, ground, door, windows, pole, railing, etc.
        """,
        help="VLM prompt"
    )
    # add batch processing related parameters
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--regenerate', action='store_true', default=False, help='whether to regenerate existing description files')

    args = parser.parse_args()

    # --- load VLM model and processor (only once) ---
    print("loading VLM model and processor...")
    try:
        vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.vlm_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            cache_dir=None,#"/vol/grid-solar/sgeusers/zhonghaoy/huggingface/hub",
        )
        processor = AutoProcessor.from_pretrained(
            args.vlm_model_path,
            cache_dir=None,#"/vol/grid-solar/sgeusers/zhonghaoy/huggingface/hub",
            use_fast=True
        )
        # set padding_side to 'left' to support batch generation
        processor.tokenizer.padding_side = 'left'
        print("VLM model and processor loaded successfully.")
    except Exception as e:
        print(f"failed to load VLM model or processor: {e}")
        return # cannot continue, exit

    # --- dataset and dataloader ---
    print("creating dataset...")
    dataset = DescriptionDataset(args.source_dir, args.target_dir, args.regenerate)

    if len(dataset) == 0:
        print("no tasks to process. please check source directory, target directory and 'regenerate' flag.")
        return

    print(f"creating DataLoader (Batch Size: {args.batch_size}, Workers: {args.num_workers})...")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False, # usually better to process sequentially for easier tracking
        num_workers=args.num_workers,
        collate_fn=lambda x: x, # use default collate function, return dictionary list
        pin_memory=True if torch.cuda.is_available() else False # if using GPU, enable pin_memory
    )

    # --- batch processing loop ---
    print("starting batch processing...")
    total_processed_count = 0
    start_time = time.time()

    # use tqdm to display progress
    for batch in tqdm(dataloader, desc="processing batches"):
        if not batch: # if batch is empty (may happen at the end)
            continue
        processed_in_batch = process_batch_descriptions(
            batch,
            vlm_model,
            processor,
            args.vlm_prompt
        )
        total_processed_count += processed_in_batch

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\n--------------------")
    print("batch processing completed!")
    print(f"total processed: {total_processed_count} / {len(dataset)} image description tasks.")
    print(f"total time: {elapsed_time:.2f} seconds")
    if total_processed_count > 0:
         print(f"average processing speed: {elapsed_time / total_processed_count:.2f} seconds/image")
    print("--------------------")


if __name__ == "__main__":
    main()