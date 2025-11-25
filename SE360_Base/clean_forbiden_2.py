import os
import json
import argparse
import glob
from tqdm import tqdm

def filter_json_file(json_path, forbidden_categories):
    """
    Reads a JSON file, filters out objects where the category contains forbidden words, 
    and overwrites the original file.

    Args:
        json_path (str): Path to the objects.json file.
        forbidden_categories (list): List of forbidden words in lowercase.

    Returns:
        bool: True if the file was modified, False otherwise.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                objects_list = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Unable to parse JSON file, skipping: {json_path}")
                return False # File could not be parsed, not modified

        if not isinstance(objects_list, list):
            print(f"Warning: JSON file root element is not a list, skipping: {json_path}")
            return False # Format incorrect, not modified

        initial_count = len(objects_list)
        filtered_objects_list = []
        modified = False

        for obj in objects_list:
            if not isinstance(obj, dict):
                print(f"Warning: Found non-dict entry in {json_path}, keeping as is: {obj}")
                filtered_objects_list.append(obj) # Keep items that cannot be processed
                continue

            category = obj.get("category", "").lower() # Get category and convert to lowercase
            category_words = category.split() # Split category into a list of words
            is_forbidden = False

            # Check if the category matches any forbidden word exactly
            if category in forbidden_categories:
                 is_forbidden = True
                 # print(f"Info: Filtered out object in {json_path} (category: '{obj.get('category')}') because it exactly matches forbidden word '{category}'")
                 modified = True
            else:
                # Check if the list of category words contains any forbidden word (as an independent word)
                for forbidden_word in forbidden_categories:
                    if forbidden_word in category_words:
                        is_forbidden = True
                        # print(f"Info: Filtered out object in {json_path} (category: '{obj.get('category')}') because it contains forbidden word '{forbidden_word}' as an independent word")
                        modified = True
                        break # Finding one matching forbidden word is enough

            if not is_forbidden:
                filtered_objects_list.append(obj)

        # Write back to file only if content has changed
        if modified:
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(filtered_objects_list, f, indent=4, ensure_ascii=False)
                # print(f"File updated: {json_path} (reduced from {initial_count} objects to {len(filtered_objects_list)})")
                return True
            except Exception as e:
                print(f"Error: Failed to write JSON file {json_path}: {e}")
                return False # Write failed, counted as not successfully modified
        else:
            # print(f"No changes needed for file: {json_path}")
            return False # File content unchanged

    except FileNotFoundError:
        print(f"Warning: File not found, skipping: {json_path}")
        return False
    except Exception as e:
        print(f"Error: Unknown error processing file {json_path}: {e}")
        return False

def process_directory(target_dir, forbidden_categories):
    """
    Traverses the target directory to find and filter all objects.json files.

    Args:
        target_dir (str): Root target directory containing scene subdirectories.
        forbidden_categories (list): List of forbidden words in lowercase.
    """
    # Build search pattern to match all objects.json files under the target directory structure
    # Example: /path/to/target_dir/SCENE_ID/matterport_stitched_images/IMAGE_ID/description/objects.json
    search_pattern = os.path.join(target_dir, "*", "matterport_stitched_images", "*", "description", "objects.json")
    print(f"Starting search for JSON files with pattern: {search_pattern}")

    json_files = glob.glob(search_pattern, recursive=True) # recursive=True might not be strictly necessary depending on glob impl and pattern

    if not json_files:
        print("Warning: No matching objects.json files found. Check target_dir and directory structure.")
        # Try a more general search in case the structure differs slightly
        print("Attempting recursive search for all objects.json under target_dir...")
        search_pattern_alt = os.path.join(target_dir, "**", "objects.json")
        json_files = glob.glob(search_pattern_alt, recursive=True)
        if json_files:
             print(f"Found {len(json_files)} files via alternative pattern.")
        else:
             print("Alternative search also found no files.")
             return

    print(f"Found {len(json_files)} objects.json files. Starting processing...")

    modified_count = 0
    processed_count = 0

    # Use tqdm to show progress
    for json_path in tqdm(json_files, desc="Filtering JSON files"):
        if filter_json_file(json_path, forbidden_categories):
            modified_count += 1
        processed_count += 1

    print("\n--------------------")
    print("Processing complete!")
    print(f"Total JSON files checked: {processed_count}.")
    print(f"Files modified (contained forbidden categories): {modified_count}.")
    print("--------------------")

def main():
    parser = argparse.ArgumentParser(description="Filter objects containing forbidden categories in JSON files")
    parser.add_argument(
        "--target_dir",
        type=str,
        default="./data/Matterport3D/mp3d_base/",
        help="Target root directory containing JSON files (e.g., ./data/Matterport3D/mp3d_base/)"
    )
    parser.add_argument(
        '--forbidden_categories',
        type=str,
        default="wall,sky,stair,ground,pole,railing,floor,tile,tiles,panels,walls,floors", # Default forbidden list, including singular and plural
        help='Comma-separated list of forbidden category words (case-insensitive)'
    )
    args = parser.parse_args()

    # Process forbidden words list
    forbidden_categories = [word.strip().lower() for word in args.forbidden_categories.split(',') if word.strip()]
    if not forbidden_categories:
        print("Error: Forbidden words list cannot be empty.")
        return
    print(f"Will filter out objects with categories containing these words: {forbidden_categories}")

    process_directory(args.target_dir, forbidden_categories)

if __name__ == "__main__":
    main()