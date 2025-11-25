import numpy as np
import os
import random

def create_skybox_npy(base_dir, train_output_path, test_output_path):
    """
    Create .npy files containing view paths. 
    The training set uses all data except the 'valid' folder, 
    and the test set uses data from the 'valid' folder.
    
    Args:
        base_dir: Root directory containing all scene folders.
        train_output_path: Output path for the training set .npy file.
        test_output_path: Output path for the test set .npy file.
    """
    # Store view paths for training and test sets
    train_view_paths = []
    test_view_paths = []
    
    # Specify the target directory
    remove_panorama_dir = base_dir
    
    # Check if the target directory exists
    if not os.path.exists(remove_panorama_dir):
        print(f"Error: Target directory {remove_panorama_dir} does not exist!")
        return
    
    # Iterate through all scene folders
    for scene_id in sorted(os.listdir(remove_panorama_dir)):
        scene_path = os.path.join(remove_panorama_dir, scene_id)
        if not os.path.isdir(scene_path):
            continue
        
        # Determine data allocation based on whether it is the 'valid' folder
        is_valid_folder = (scene_id == "valid")
        
        # Look for the 'matterport_stitched_images' directory
        stitched_images_dir = os.path.join(scene_path, "matterport_stitched_images")
        if not os.path.exists(stitched_images_dir):
            continue
            
        # Iterate through all view directories under the scene folder
        for view_dir in os.listdir(stitched_images_dir):
            view_path = os.path.join(stitched_images_dir, view_dir)
            if not os.path.isdir(view_path):
                continue

            # Look for the 'carries_results' directory
            carries_results_dir = os.path.join(view_path, "carries_results")
            if not os.path.exists(carries_results_dir) or not os.path.isdir(carries_results_dir):
                continue
                
            # Iterate through all object directories under 'carries_results'
            for obj_dir in os.listdir(carries_results_dir):
                obj_path = os.path.join(carries_results_dir, obj_dir)
                if not os.path.isdir(obj_path):
                    continue

                # Look for .txt files containing 'simple_caption'
                caption_txt_file = None
                for filename in os.listdir(obj_path):
                    # Check if the filename contains 'simple_caption' and ends with '.txt'
                    if 'simple_caption' in filename.lower() and filename.endswith('.txt'):
                        caption_txt_file = os.path.join(obj_path, filename)
                        break # Break the inner loop once the first matching file is found

                # If a matching .txt file is found
                if caption_txt_file and os.path.exists(caption_txt_file):
                    # Construct the relative path
                    rel_path = os.path.relpath(caption_txt_file, os.path.dirname(remove_panorama_dir)) # Use the path of the found txt file
                    rel_path = rel_path.replace('\\', '/')  # Ensure consistent path formatting
                    
                    # Add to training or test set depending on whether it is the 'valid' folder
                    if is_valid_folder:
                        test_view_paths.append(rel_path)
                    else:
                        train_view_paths.append(rel_path)
    
    # Print statistical information
    print(f"Found {len(train_view_paths)} editing files for the training set")
    print(f"Found {len(test_view_paths)} editing files for the test set")
    
    if len(train_view_paths) == 0 and len(test_view_paths) == 0:
        print("Warning: No matching files found! Please check the directory structure.")
        # Create empty arrays and save
        np.save(train_output_path, np.array([]))
        np.save(test_output_path, np.array([]))
        return
    
    # Convert lists to numpy arrays and save
    train_paths = np.array(train_view_paths)
    test_paths = np.array(test_view_paths)
    
    np.save(train_output_path, train_paths)
    np.save(test_output_path, test_paths)
    
    print(f"Creation complete: {len(train_paths)} edits in training set, {len(test_paths)} edits in test set")

    print("Training set shape:", train_paths.shape)
    print("Test set shape:", test_paths.shape)
    if len(train_paths) > 0:
        print("First 3 path examples from the training set:")
        for i in range(min(3, len(train_paths))):
            print(train_paths[i])
    if len(test_paths) > 0:
        print("First 3 path examples from the test set:")
        for i in range(min(3, len(test_paths))):
            print(test_paths[i])

# Usage example
if __name__ == "__main__":
    base_dir = "./data/Matterport3D/mp3d_base"  # Path to the data directory
    train_output_path = "./data/Matterport3D/mp3d_base/train.npy"  # Output path for training set
    test_output_path = "./data/Matterport3D/mp3d_base/test.npy"    # Output path for test set

    # No longer need 'test_ratio' parameter; split directly based on folder names
    create_skybox_npy(base_dir, train_output_path, test_output_path)