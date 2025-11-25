import os
import glob
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
import math

def calculate_mask_area(mask_path):
    """Calculate the area (number of white pixels) and dimensions of the mask image."""
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return 0, 0, 0, 0, 0  # area, mask_width, mask_height, img_width, img_height
        
        # Get image dimensions
        img_height, img_width = mask.shape
        
        # Calculate the number of non-zero pixels (assuming the mask is a binary image)
        area = np.count_nonzero(mask > 128)  # Threshold set to 128; pixels > 128 are considered white
        
        # Calculate the bounding box of the mask
        coords = np.where(mask > 128)
        if len(coords[0]) == 0:  # No white pixels found
            return area, 0, 0, img_width, img_height
        
        # Calculate bounding box dimensions
        min_row, max_row = coords[0].min(), coords[0].max()
        min_col, max_col = coords[1].min(), coords[1].max()
        mask_width = max_col - min_col + 1
        mask_height = max_row - min_row + 1
        
        return area, mask_width, mask_height, img_width, img_height
    except Exception as e:
        print(f"Error processing image {mask_path}: {e}")
        return 0, 0, 0, 0, 0

def is_mask_size_valid(mask_width, mask_height, img_width, img_height, threshold_ratio=0.8):
    """Check if the mask size is valid (does not exceed the threshold ratio of the image dimensions)."""
    width_ratio = mask_width / img_width if img_width > 0 else 1
    height_ratio = mask_height / img_height if img_height > 0 else 1
    
    # If width or height exceeds the threshold ratio, it is considered invalid
    return width_ratio <= threshold_ratio and height_ratio <= threshold_ratio

def find_all_mask_files(base_path):
    """Find all full_mask.png files matching the path pattern."""
    pattern = os.path.join(base_path, "*/matterport_stitched_images/*/carries_results/*/full_mask.png")
    mask_files = glob.glob(pattern)
    return mask_files

def extract_path_info(mask_path, base_path):
    """Extract scene, view, and obj information from the mask file path."""
    relative_path = os.path.relpath(mask_path, base_path)
    parts = relative_path.split(os.sep)
    
    scene = parts[0]
    view = parts[2]
    obj_folder = parts[4]
    
    return scene, view, obj_folder

def divide_into_groups(mask_areas):
    """Divide masks into three groups (large, medium, small) based on area."""
    total_count = len(mask_areas)
    
    # Divide into three equal parts
    large_end = total_count // 3
    medium_end = (total_count * 2) // 3
    
    large_group = mask_areas[:large_end]          # Top 1/3 with the largest areas
    medium_group = mask_areas[large_end:medium_end] # Middle 1/3 with medium areas
    small_group = mask_areas[medium_end:]         # Bottom 1/3 with the smallest areas
    
    return large_group, medium_group, small_group

def main():
    # Source and target paths
    source_base = "./data/Matterport3D/mp3d_base"
    target_base = "./data/Matterport3D/mp3d_hf_first"
    
    # Set the total number of selections
    total_select_count = 10000
    
    # Set selection ratios for each group
    large_ratio = 0.7   # 70%
    medium_ratio = 0.2  # 20%
    small_ratio = 0.1   # 10%
    
    print("Starting search for all full_mask.png files...")
    mask_files = find_all_mask_files(source_base)
    print(f"Found {len(mask_files)} mask files")
    
    if len(mask_files) == 0:
        print("No mask files found. Please check if the path is correct.")
        return
    
    # Calculate all mask areas
    print("Calculating mask areas...")
    mask_areas = []
    filtered_count = 0
    
    for mask_path in tqdm(mask_files, desc="Calculating areas"):
        area, mask_width, mask_height, img_width, img_height = calculate_mask_area(mask_path)
        scene, view, obj_folder = extract_path_info(mask_path, source_base)
        
        # Construct source folder path (path to the obj folder)
        source_obj_folder = os.path.dirname(mask_path)
        
        # Check if mask size is valid (not exceeding 4/5 of image dimensions)
        if is_mask_size_valid(mask_width, mask_height, img_width, img_height):
            mask_areas.append({
                'mask_path': mask_path,
                'area': area,
                'scene': scene,
                'view': view,
                'obj_folder': obj_folder,
                'source_obj_folder': source_obj_folder,
                'mask_width': mask_width,
                'mask_height': mask_height,
                'img_width': img_width,
                'img_height': img_height
            })
        else:
            filtered_count += 1
    
    print(f"Filtering Statistics:")
    print(f"Total samples: {len(mask_files)}")
    print(f"Filtered samples: {filtered_count} (size too large)")
    print(f"Valid samples: {len(mask_areas)}")
    
    if len(mask_areas) == 0:
        print("No valid samples found. Exiting.")
        return
    
    # Sort by area (descending)
    mask_areas.sort(key=lambda x: x['area'], reverse=True)
    
    # Divide into large, medium, and small groups
    large_group, medium_group, small_group = divide_into_groups(mask_areas)
    
    print(f"Grouping Status:")
    print(f"Large Group (Largest Area): {len(large_group)} samples")
    print(f"Medium Group (Medium Area): {len(medium_group)} samples")
    print(f"Small Group (Smallest Area): {len(small_group)} samples")
    
    if len(large_group) > 0:
        print(f"Large Group Area Range: {large_group[-1]['area']} - {large_group[0]['area']} pixels")
    if len(medium_group) > 0:
        print(f"Medium Group Area Range: {medium_group[-1]['area']} - {medium_group[0]['area']} pixels")
    if len(small_group) > 0:
        print(f"Small Group Area Range: {small_group[-1]['area']} - {small_group[0]['area']} pixels")
    
    # Calculate selection count for each group
    large_select = int(total_select_count * large_ratio)
    medium_select = int(total_select_count * medium_ratio)
    small_select = total_select_count - large_select - medium_select  # Ensure the total count is correct
    
    print(f"\nSelection Allocation:")
    print(f"Large Group Selection: {large_select} samples ({large_ratio*100}%)")
    print(f"Medium Group Selection: {medium_select} samples ({medium_ratio*100}%)")
    print(f"Small Group Selection: {small_select} samples ({small_select/total_select_count*100:.1f}%)")
    
    # Select samples from each group
    selected_samples = []
    
    # Select from the large group
    if len(large_group) >= large_select:
        selected_samples.extend(large_group[:large_select])
    else:
        selected_samples.extend(large_group)
        print(f"Warning: Not enough samples in large group, only selected {len(large_group)} samples.")
    
    # Select from the medium group
    if len(medium_group) >= medium_select:
        selected_samples.extend(medium_group[:medium_select])
    else:
        selected_samples.extend(medium_group)
        print(f"Warning: Not enough samples in medium group, only selected {len(medium_group)} samples.")
    
    # Select from the small group
    if len(small_group) >= small_select:
        selected_samples.extend(small_group[:small_select])
    else:
        selected_samples.extend(small_group)
        print(f"Warning: Not enough samples in small group, only selected {len(small_group)} samples.")
    
    print(f"\nActually selected {len(selected_samples)} samples.")
    
    # Copy folders
    print("Starting folder copy...")
    
    for i, item in enumerate(tqdm(selected_samples, desc="Copying folders")):
        scene = item['scene']
        view = item['view']
        obj_folder = item['obj_folder']
        source_obj_folder = item['source_obj_folder']
        
        # Construct target path
        target_obj_folder = os.path.join(
            target_base,
            scene,
            "matterport_stitched_images",
            view,
            "carries_results",
            obj_folder
        )
        
        try:
            # Create parent directory for the target path
            os.makedirs(os.path.dirname(target_obj_folder), exist_ok=True)
            
            # Copy the entire obj folder
            if os.path.exists(source_obj_folder):
                if os.path.exists(target_obj_folder):
                    shutil.rmtree(target_obj_folder)  # Remove if target already exists
                shutil.copytree(source_obj_folder, target_obj_folder)
            else:
                print(f"Warning: Source folder does not exist: {source_obj_folder}")
                
        except Exception as e:
            print(f"Error copying folder {source_obj_folder} -> {target_obj_folder}: {e}")
    
    print("Done! Samples have been copied to the target directory according to the large/medium/small group ratios.")

if __name__ == "__main__":
    main()