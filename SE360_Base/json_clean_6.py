import os
import json
import glob
import re
import copy

def count_words_excluding_parentheses(text):
    """Calculate the word count of the text, excluding content within parentheses."""
    # Use regex to remove content inside parentheses (including round, square, and curly brackets)
    text_without_parentheses = re.sub(r'\([^)]*\)|\[[^\]]*\]|\{[^}]*\}', '', text)
    # Split by whitespace to count words (handling extra spaces)
    words = text_without_parentheses.split()
    return len(words)

def remove_parentheses_content(text):
    """Remove parentheses and their content (including round, square, and curly brackets) from the text."""
    # Remove parentheses/content and remove potential extra spaces
    cleaned_text = re.sub(r'\([^)]*\)|\[[^\]]*\]|\{[^}]*\}', '', text)
    return ' '.join(cleaned_text.split()) # Merge extra spaces

def clean_json_files():
    # Base path
    base_path = "./data/Matterport3D/mp3d_base/"

    # Counters
    processed_files = 0
    modified_files = 0
    deleted_files = 0
    
    # Store modified/deleted file paths
    modified_file_paths = []
    deleted_file_paths = []

    # Record reasons for filtering items
    filter_reasons = {"explanation": 0, "too_long": 0, "rug_category": 0} 

    # Define rug-related keywords (lowercase)
    RUG_LIKE_KEYWORDS = {"rug", "blanket"} # Add more words as needed

    # Iterate through all scene directories
    for scene_dir in os.listdir(base_path):
        scene_path = os.path.join(base_path, scene_dir)
        if not os.path.isdir(scene_path):
            continue

        # Construct path pattern
        pattern = os.path.join(scene_path, "matterport_stitched_images", "*", "carries_results", "*.json")

        # Get all matching JSON files
        json_files = glob.glob(pattern)

        # Process each JSON file
        for json_file in json_files:
            processed_files += 1
            original_data = None
            data_changed = False # Used to mark if write-back is needed, confirmed by final comparison

            try:
                # Read JSON file
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    original_data = copy.deepcopy(data) # Create a deep copy of the original data

                # Check if 'carried_items' field exists, skip if not
                if "carried_items" not in data:
                    continue

                original_carried_items = data.get("carried_items", []) # Get original list
                items_to_process = original_carried_items # Default to processing all original items

                # --- Logic: Filter based on original_category ---
                original_category = data.get("original_category", "")
                apply_rug_filter = False
                if isinstance(original_category, str) and original_category:
                    category_lower = original_category.lower()
                    for keyword in RUG_LIKE_KEYWORDS:
                        if keyword in category_lower:
                            apply_rug_filter = True
                            break

                if apply_rug_filter and original_carried_items: # Apply only if category matches and list is not empty
                    items_before_filter = len(items_to_process)
                    items_to_process = [original_carried_items[0]] # Keep only the first item
                    items_after_filter = len(items_to_process)
                    # Record items removed due to rug category rule
                    filter_reasons["rug_category"] += (items_before_filter - items_after_filter)
                # --- End Logic ---

                # --- Existing Filtering Logic ---
                filtered_items = []
                should_delete_file = False # Flag to check if file needs deletion

                for item in items_to_process: # Note: items_to_process might have been modified above
                    # Skip conditions:
                    # 1. name contains "Explanation"
                    # 2. name length exceeds 10 words after excluding content in parentheses
                    if "name" in item:
                        original_name = item["name"] # This is the original name of the current item

                        if "Explanation" in original_name:
                            filter_reasons["explanation"] += 1
                            continue # Skip this item

                        # Remove parentheses and content from name, and clean spaces
                        cleaned_name = remove_parentheses_content(original_name)
                        # Update name in item if it has changed
                        if cleaned_name != item["name"]:
                             item["name"] = cleaned_name
                             # Note: Modifying 'item' directly affects subsequent data != original_data check
                             # Since we overwrite data["carried_items"] with filtered_items and compare against deepcopy, this is safe.

                        words_count = count_words_excluding_parentheses(cleaned_name) # Count using cleaned name
                        # Check for overly long names
                        if words_count > 10:
                            filter_reasons["too_long"] += 1
                            # Print sample
                            if filter_reasons["too_long"] <= 10:
                                print(f"Sample of overly long name (from file {os.path.basename(json_file)}): {cleaned_name}")
                                print(f"Word count excluding parentheses: {words_count}")
                            should_delete_file = True  # Mark file for deletion
                            break  # Stop checking this file once a long name is found

                    # Keep items that were not filtered out
                    filtered_items.append(item)
                # --- End Existing Filtering Logic ---

                # Process file: Delete or Modify
                if should_delete_file:
                    # Delete file containing overly long name
                    os.remove(json_file)
                    deleted_files += 1
                    deleted_file_paths.append(json_file)
                else:
                    # Update data with filtered list (which might be empty)
                    data["carried_items"] = filtered_items

                    # Compare processed data with original data (most reliable check)
                    if data != original_data:
                        data_changed = True

                    # Only count as modified and write back if data actually changed
                    if data_changed:
                        modified_files += 1
                        modified_file_paths.append(json_file)
                        # Write back modified file
                        with open(json_file, 'w') as f:
                            json.dump(data, f, indent=2)

            except Exception as e:
                print(f"Error processing file {json_file}: {str(e)}")

    # Print filter statistics
    print("\nFilter Statistics:")
    print(f"Items filtered due to containing 'Explanation': {filter_reasons['explanation']} items")
    print(f"Files deleted due to overly long names: {filter_reasons['too_long']} times (Note: one deletion may correspond to multiple items that would have been filtered)")
    print(f"Items removed due to 'rug/blanket' category rule: {filter_reasons['rug_category']} items (only the first item was kept)")

    # Print deleted files info
    print(f"\nA total of {deleted_files} files containing overly long names were deleted")
    if deleted_file_paths:
        print("\nTop 10 deleted file paths:")
        for i, path in enumerate(deleted_file_paths[:10]):
            print(f"{i+1}. {path}")
    else:
        print("No files were deleted due to overly long names.")
        
    # Print modified files info (Optional)
    print(f"\nA total of {modified_files} files were modified")
    if modified_file_paths:
        print("\nTop 10 modified file paths:")
        for i, path in enumerate(modified_file_paths[:10]):
             print(f"{i+1}. {path}")
    # else:
    #      print("No files were modified.")


    return processed_files, modified_files, deleted_files

if __name__ == "__main__":
    processed, modified, deleted = clean_json_files()
    print(f"\nProcessing complete! Processed {processed} files, modified {modified} files, deleted {deleted} files.")