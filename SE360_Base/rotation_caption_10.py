import argparse
import os
import torch
import numpy as np
import json
import glob
from tqdm import tqdm
import time
import traceback

# --- Standard LLM Imports ---
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# --- Dataset Class (Unchanged) ---
class DescriptionDataset(Dataset):
    """Dataset used to load JSON file paths containing object descriptions."""
    def __init__(self, json_root_dir):
        self.json_root_dir = json_root_dir
        self.tasks = self._collect_tasks()
        print(f"Dataset initialization complete. Found {len(self.tasks)} JSON file tasks.")

    def _collect_tasks(self):
        """Collect all JSON file tasks that need processing."""
        tasks = []
        print(f"Starting search for JSON files in root directory: {self.json_root_dir}")
        search_pattern = os.path.join(self.json_root_dir, '**', '*_obj*_recaption.json')
        print(f"Search pattern: {search_pattern}")

        try:
            json_files = glob.glob(search_pattern, recursive=True)
            print(f"Found {len(json_files)} matching JSON files.")

            for json_file_path in json_files:
                if os.path.isfile(json_file_path):
                    tasks.append({
                        "json_path": json_file_path,
                        "relative_path": os.path.relpath(json_file_path, self.json_root_dir)
                    })

        except Exception as e:
            print(f"Error searching for JSON files: {e}")

        tasks.sort(key=lambda x: x['json_path'])
        print(f"Collected a total of {len(tasks)} JSON tasks to process.")
        if not tasks:
             print("Warning: No JSON tasks collected for processing. Please check the JSON root directory and search pattern.")
        return tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]


# --- Modification: Function for batch processing analysis tasks ---
# Analyze only 'original_description' and force 'simple_caption' to align with it
def process_batch_analysis(batch_items, model, tokenizer, device, regenerate=False):
    """
    Batch process JSON description analysis tasks.
    1. Use only 'original_description' for LLM analysis.
    2. Create caption_*.txt based on LLM's judgment (absolute/relative) of 'original_description'.
    3. If 'simple_original_description' exists, create an aligned simple_caption_*.txt file.
    Decide whether to overwrite existing marker files based on the regenerate flag.

    Returns:
        Tuple: (absolute_count, relative_count, unknown_count, error_count, txt_created_count, txt_skipped_count)
              These counts reflect analysis results based on original_description and file operations.
    """
    # --- Modification: Store metadata, including both types of description text (if they exist) ---
    batch_metadata = [] # Each element corresponds to a JSON file
    llm_input_descriptions = [] # Contains only original_description
    llm_metadata_indices = [] # Records the index in batch_metadata for each description in llm_input_descriptions
    batch_read_errors = [] # Stores errors during JSON reading

    # 1. Read description information from JSON files in the batch
    for idx, item in enumerate(batch_items):
        json_path = item['json_path']
        original_desc_text = None
        simple_desc_text = None
        read_error_for_item = False

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check 'original_description'
            if 'original_description' in data and isinstance(data['original_description'], str) and data['original_description'].strip():
                original_desc_text = data['original_description']

            # Check 'simple_original_description' (only record existence and text)
            if 'simple_original_description' in data and isinstance(data['simple_original_description'], str) and data['simple_original_description'].strip():
                simple_desc_text = data['simple_original_description']

            # Only add to LLM processing list if original_description is found
            if original_desc_text:
                llm_input_descriptions.append(original_desc_text)
                llm_metadata_indices.append(idx) # Record which item in batch_metadata this description corresponds to
            elif not simple_desc_text: # If even simple description is missing, report error
                # Note: If only simple exists without original, this task will skip LLM analysis
                 batch_read_errors.append({'json_path': json_path, 'error': 'no_original_description_found_for_llm'})
                 read_error_for_item = True # Mark this item as having a read/preparation error

        except FileNotFoundError:
            batch_read_errors.append({'json_path': json_path, 'error': 'file_not_found'})
            read_error_for_item = True
        except json.JSONDecodeError:
            batch_read_errors.append({'json_path': json_path, 'error': 'json_decode_error'})
            read_error_for_item = True
        except Exception as e:
            batch_read_errors.append({'json_path': json_path, 'error': f'reading_json: {e}'})
            read_error_for_item = True

        # Store metadata for this JSON file
        batch_metadata.append({
            'json_path': json_path,
            'original_description': original_desc_text,
            'simple_original_description': simple_desc_text,
            'had_read_error': read_error_for_item
        })

    current_error_count = len(batch_read_errors) # Initial error count comes from file reading

    if not llm_input_descriptions:
        # print("No valid original_description in batch to send to LLM.") # Debugging info
        # Count all items without original_description that might need to be reported as errors
        additional_errors = sum(1 for meta in batch_metadata if meta['had_read_error'])
        return 0, 0, 0, additional_errors, 0, 0 # abs, rel, unk, err, txt_created, txt_skipped

    # 2. Prepare batch prompts for valid original_descriptions
    batch_prompts_text = []
    prompt_template = """Given the following object description from a 360-degree image context:
"{description}"

Determine if the description primarily uses absolute positioning or relative positioning.
- Absolute positioning refers to the object's location only concerning the overall image frame or room structure (e.g., 'center of the image', 'left side of the room', 'right side of the room', 'right of the image', 'centerly in the image').
- Relative positioning refers to the object's location concerning other specific objects (e.g., 'next to the chair', 'on top of the table', 'behind the sofa', 'left of the sink', 'right of the side table').

For example:
- "A white toilet with a closed lid, situated to the right of the sink area, near a textured wall." -> "relative"
- "A white toilet with a closed lid, situated to the right of the room." -> "relative" 
- "A black sofa, situated to the left of the image." -> "absolute"
- "A yellow chair, situated to the left of the table." -> "relative"
- "A blue desk on the center of the image." -> "absolute"
- "A red vase, placed on the right of the table." -> "relative"
- "A large illuminated mirror with a rectangular frame, mounted above the sink, reflecting part of the room." -> "absolute" 
- "A white rectangular sink with a modern design, positioned centrally on a curved vanity counter." -> "relative"
- "A flat-screen television mounted on the wall to the left, displaying colorful graphics." -> "absolute"
- "A large abstract painting with warm tones, hanging on the wall to the right of the dining area." -> "relative"


Respond with only one word: 'absolute' or 'relative'.
"""
    for desc in llm_input_descriptions:
        current_prompt = prompt_template.format(description=desc)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": current_prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        batch_prompts_text.append(text)

    # 3. Batch process prompts (LLM Inference)
    current_absolute = 0
    current_relative = 0
    current_unknown = 0
    current_txt_created = 0
    current_txt_skipped = 0
    vlm_error_count = 0 # Errors during LLM processing or file writing

    try:
        # --- Tokenizer Padding ---
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))
                print(f"Warning: Added and set new pad_token: {tokenizer.pad_token}")

        # --- LLM Inference ---
        inputs = tokenizer(
            text=batch_prompts_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_ids = inputs['input_ids']
        with torch.no_grad():
            generated_ids_with_input = model.generate(
                **inputs,
                max_new_tokens=10
            )

        generated_ids = [
            output_ids[len(input_id):]
            for input_id, output_ids in zip(input_ids, generated_ids_with_input)
        ]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # 4. Parse each response and create files (align simple_caption)
        if len(responses) != len(llm_input_descriptions):
             print(f"\nWarning: Mismatch between LLM response count ({len(responses)}) and pending description count ({len(llm_input_descriptions)})! Skipping marker file creation for this batch.")
             vlm_error_count = len(llm_input_descriptions) # Count all items in this LLM batch as errors
        else:
            for i, response in enumerate(responses):
                metadata_index = llm_metadata_indices[i]
                metadata = batch_metadata[metadata_index]

                # Skip items that had errors during the reading phase (although they shouldn't be in the LLM list)
                if metadata['had_read_error']:
                    continue

                json_path = metadata['json_path']
                original_desc_text = metadata['original_description'] # Guaranteed to exist here
                simple_desc_text = metadata['simple_original_description'] # Might be None

                response_clean = response.strip().lower()
                position_type = 'unknown' # Default based on original_description analysis
                if 'absolute' in response_clean:
                    position_type = 'absolute'
                    current_absolute += 1
                elif 'relative' in response_clean:
                    position_type = 'relative'
                    current_relative += 1
                else:
                    current_unknown += 1
                    # print(f"\nWarning: Unable to parse LLM response: '{response}' -> '{response_clean}' for original_desc in {os.path.basename(json_path)}")

                # --- File creation logic ---
                if position_type in ['absolute', 'relative']:
                    json_dir = os.path.dirname(json_path)

                    # --- Process caption_*.txt files ---
                    caption_prefix = "caption_"
                    caption_files_exist = [
                        os.path.join(json_dir, f"{caption_prefix}absolute.txt"),
                        os.path.join(json_dir, f"{caption_prefix}relative.txt")
                    ]
                    should_create_caption_file = False
                    caption_marker_exists = any(os.path.exists(f) for f in caption_files_exist)

                    if regenerate or not caption_marker_exists:
                        should_create_caption_file = True

                    if should_create_caption_file:
                        try:
                            # Delete old caption files
                            for old_file in caption_files_exist:
                                if os.path.exists(old_file):
                                    os.remove(old_file)

                            # Create new caption file
                            txt_filename = f"{caption_prefix}{position_type}.txt"
                            txt_filepath = os.path.join(json_dir, txt_filename)
                            modified_description = f"add {original_desc_text.lower()}"
                            with open(txt_filepath, 'w', encoding='utf-8') as f:
                                f.write(modified_description)
                            current_txt_created += 1
                        except OSError as e:
                            print(f"\nError: Error processing caption marker file (Action: {'Overwrite' if regenerate else 'Create'}) at {json_dir}: {e}")
                            vlm_error_count += 1
                        except Exception as e:
                            print(f"\nUnexpected error: Error processing caption marker file at {json_dir}: {e}")
                            vlm_error_count += 1
                    elif caption_marker_exists: # File exists and not regenerating
                         current_txt_skipped += 1

                    # --- Process simple_caption_*.txt files (if simple_desc exists) ---
                    if simple_desc_text:
                        simple_prefix = "simple_caption_"
                        simple_files_exist = [
                            os.path.join(json_dir, f"{simple_prefix}absolute.txt"),
                            os.path.join(json_dir, f"{simple_prefix}relative.txt")
                        ]
                        should_create_simple_file = False
                        simple_marker_exists = any(os.path.exists(f) for f in simple_files_exist)

                        if regenerate or not simple_marker_exists:
                            should_create_simple_file = True

                        if should_create_simple_file:
                            try:
                                # Delete old simple_caption files
                                for old_file in simple_files_exist:
                                    if os.path.exists(old_file):
                                        os.remove(old_file)

                                # Create new simple_caption file (use same position_type as caption)
                                txt_filename = f"{simple_prefix}{position_type}.txt" # Align!
                                txt_filepath = os.path.join(json_dir, txt_filename)
                                # Use text from simple_original_description
                                modified_description = f"add {simple_desc_text.lower()}"
                                with open(txt_filepath, 'w', encoding='utf-8') as f:
                                    f.write(modified_description)
                                current_txt_created += 1 # Also count towards total creation
                            except OSError as e:
                                print(f"\nError: Error processing simple_caption marker file (Action: {'Overwrite' if regenerate else 'Create'}) at {json_dir}: {e}")
                                vlm_error_count += 1
                            except Exception as e:
                                print(f"\nUnexpected error: Error processing simple_caption marker file at {json_dir}: {e}")
                                vlm_error_count += 1
                        elif simple_marker_exists: # File exists and not regenerating
                             current_txt_skipped += 1
                # --- End of file creation logic ---
                elif position_type == 'unknown':
                    # For 'unknown' cases, decide whether to delete old files based on the regenerate flag
                    if regenerate:
                        json_dir = os.path.dirname(json_path)
                        try:
                            # Delete caption files
                            caption_files = glob.glob(os.path.join(json_dir, "caption_*.txt"))
                            for f in caption_files: os.remove(f)
                            # Delete simple_caption files
                            simple_files = glob.glob(os.path.join(json_dir, "simple_caption_*.txt"))
                            for f in simple_files: os.remove(f)
                        except OSError as e:
                             print(f"\nError: Error deleting old marker files for unknown type in regenerate mode at {json_dir}: {e}")
                             vlm_error_count += 1


    except Exception as e:
        print(f"\nError: LLM batch processing failed: {e}")
        traceback.print_exc()
        vlm_error_count = len(llm_input_descriptions) # Count all LLM inputs in this batch as errors
        current_absolute = 0
        current_relative = 0
        current_unknown = 0
        current_txt_created = 0
        current_txt_skipped = 0

    # Return total counts for this batch (based on original_description analysis)
    # Add errors from the reading phase
    total_errors_this_batch = current_error_count + vlm_error_count
    # If original description is unknown, also consider it an error/incomplete status
    total_errors_this_batch += current_unknown

    return current_absolute, current_relative, current_unknown, total_errors_this_batch, current_txt_created, current_txt_skipped

# --- main function (Mostly unchanged, updated summaries) ---
def main():
    parser = argparse.ArgumentParser(description="Use LLM to batch analyze 'original_description' in JSON files, determine absolute/relative positioning, and create aligned caption_*.txt and simple_caption_*.txt markers.") # Updated description

    parser.add_argument(
        "--vlm_model_path",
        type=str,
        default="Qwen/Qwen3-8B",
        help="LLM model path or Hugging Face Hub model identifier"
    )
    parser.add_argument(
        "--json_root_dir",
        type=str,
        default="./data/Matterport3D/mp3d_base/", # Please ensure this path is correct
        help="Root directory containing scene subdirectories and *_recaption.json files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on ('cuda' or 'cpu')"
    )
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of data loading workers')
    parser.add_argument(
        '--regenerate',
        action='store_true',
        help="If specified, forces regeneration of all .txt marker files even if they exist. Defaults to not regenerating."
    )

    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"JSON root directory: {args.json_root_dir}")
    print(f"LLM model: {args.vlm_model_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Data loader workers: {args.num_workers}")
    print(f"Force regenerate marker files (--regenerate): {args.regenerate}")
    print("\nNote: This version analyzes only 'original_description' and forces 'simple_caption_*.txt' to align with it.\n")


    print(f"Loading LLM model and tokenizer: {args.vlm_model_path} ...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.vlm_model_path,
            torch_dtype="auto",
            attn_implementation=None,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.vlm_model_path, use_fast=True)

        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                print(f"Pad token set to EOS token: {tokenizer.pad_token}")
            else:
                print("Warning: Tokenizer has no pad_token or eos_token. Adding [PAD] as pad_token.")
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))
                print(f"Set pad_token to {tokenizer.pad_token}")

        model.eval()
        print("LLM model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Failed to load LLM model or tokenizer: {e}")
        traceback.print_exc()
        return

    print("Creating dataset to find JSON files...")
    dataset = DescriptionDataset(args.json_root_dir)

    if len(dataset) == 0:
        print("No JSON files found for processing. Please check JSON root directory and search pattern (*_recaption.json).")
        return

    print(f"Creating DataLoader (Batch Size: {args.batch_size}, Workers: {args.num_workers})...")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
        pin_memory=True if device.type == 'cuda' else False
    )

    print("Starting batch analysis of 'original_description' and creating aligned marker files...")
    total_absolute = 0
    total_relative = 0
    total_unknown = 0 # Judgments from LLM on original_description
    total_error = 0 # Includes reading errors, LLM errors, file writing errors
    total_txt_created = 0
    total_txt_skipped = 0
    start_time = time.time()
    total_original_desc_processed_by_llm = 0 # Actual quantity sent to LLM for analysis

    for batch in tqdm(dataloader, desc="Processing batches"):
        if not batch:
            continue

        abs_count, rel_count, unk_count, err_count, txt_created_count, txt_skipped_count = process_batch_analysis(
            batch,
            model,
            tokenizer,
            device,
            regenerate=args.regenerate
        )

        total_absolute += abs_count
        total_relative += rel_count
        total_unknown += unk_count # Count where LLM could not judge original_description
        total_error += err_count # Accumulate all types of errors
        total_txt_created += txt_created_count
        total_txt_skipped += txt_skipped_count
        total_original_desc_processed_by_llm += (abs_count + rel_count + unk_count)


    end_time = time.time()
    elapsed_time = end_time - start_time
    total_json_files = len(dataset)

    print("\n--------------------")
    print("Batch analysis complete!")
    print(f"Total scanned {total_json_files} JSON file tasks.")
    print(f"Total {total_original_desc_processed_by_llm} 'original_description' items submitted to LLM for analysis.")
    print(f"  - LLM judged as absolute: {total_absolute}")
    print(f"  - LLM judged as relative: {total_relative}")
    print(f"  - LLM unknown or unable to judge 'original_description': {total_unknown}") # Updated description
    print(f"Total errors during processing (JSON reading/LLM/File creation/Unknown): {total_error}") # Updated description
    print(f"Total marker files successfully created/overwritten (including caption and simple_caption): {total_txt_created}") # Updated description
    if not args.regenerate:
        print(f"Total files skipped because marker files already exist: {total_txt_skipped}")
    print(f"Total time: {elapsed_time:.2f} seconds")

    if total_original_desc_processed_by_llm > 0 and elapsed_time > 0:
         print(f"Average LLM analysis speed: {total_original_desc_processed_by_llm / elapsed_time:.2f} original_description/sec")
    if total_json_files > 0 and elapsed_time > 0:
        print(f"Average JSON file processing speed: {total_json_files / elapsed_time:.2f} files/sec")
    print("--------------------")


if __name__ == "__main__":
    main()