#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np
import json
import glob


from py360.p2e import p2e

def load_projection_params_from_json(json_path):
    """
    Load projection parameters from a JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        tuple: (center_u_deg, center_v_deg, final_hfov_deg, final_vfov_deg, erp_image_path)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    params = data['perspective_transform_params']
    center_u_deg = params['center_u_deg']
    center_v_deg = params['center_v_deg'] 
    final_hfov_deg = params['final_hfov_deg']
    final_vfov_deg = params['final_vfov_deg']
    
    # Get original ERP image path
    erp_image_path = data['original_erp_image']
    
    return center_u_deg, center_v_deg, final_hfov_deg, final_vfov_deg, erp_image_path

def process_single_image(perspective_image_path, json_path, scene, view, obj_n):
    """
    Process the projection for a single image.
    """
    print(f"\n=== Processing: Scene={scene}, View={view}, Object={obj_n} ===")
    
    # New switch: Whether to use the top/bottom parts of the original ERP to cover the final result
    use_original_top_bottom = True  # Set to True to enable, False to disable
    top_percentage = 0.13  # Top coverage percentage (0-15%)
    bottom_percentage = 0.13  # Bottom coverage percentage (85%-100%)
    
    try:
        # Parameters obtained from JSON
        center_u_deg, center_v_deg, final_hfov_deg, final_vfov_deg, original_erp_path = load_projection_params_from_json(json_path)
        
        original_hfov_from_json = final_hfov_deg * 1.2
        final_hfov_deg = final_hfov_deg * 1.2
        
        # ERP output dimensions (based on info in JSON)
        erp_height = 1024
        erp_width = 2048
        
        print(f"Perspective image path: {perspective_image_path}")
        print(f"JSON path: {json_path}")
        print(f"Original ERP path: {original_erp_path}")
        print(f"FOV Params: {original_hfov_from_json}° -> {final_hfov_deg}°")
        
        # Check if files exist
        if not os.path.exists(perspective_image_path):
            print(f"Skipping: Perspective image file not found {perspective_image_path}")
            return False
        
        if not os.path.exists(original_erp_path):
            print(f"Skipping: Original ERP image file not found {original_erp_path}")
            return False
        
        # Load perspective image
        perspective_img = cv2.imread(perspective_image_path)
        if perspective_img is None:
            print(f"Skipping: Unable to load perspective image file {perspective_image_path}")
            return False
        
        # Load original ERP image
        original_erp = cv2.imread(original_erp_path)
        if original_erp is None:
            print(f"Skipping: Unable to load original ERP image file {original_erp_path}")
            return False
        
        # If original ERP size differs from target size, adjust target size
        if original_erp.shape[:2] != (erp_height, erp_width):
            erp_height, erp_width = original_erp.shape[:2]
            print(f"Adjusting output ERP size to match original image: {erp_height}x{erp_width}")
        
        print(f"Perspective image dimensions: {perspective_img.shape}")
        print(f"Params: center_u={center_u_deg}°, center_v={center_v_deg}°")
        print(f"FOV: h_fov={final_hfov_deg}°, v_fov={final_vfov_deg}°")
        
        # Project perspective view back to ERP format
        print("Projecting perspective view back to ERP format...")
        erp_img = p2e(
            p_img=perspective_img,
            fov_deg=final_hfov_deg,
            u_deg=center_u_deg,
            v_deg=center_v_deg,
            h=erp_height,
            w=erp_width,
            mode='bilinear'
        )
        
        # Ensure output is the correct data type
        erp_img = erp_img.astype(np.uint8)
        
        # Get non-zero pixel area for overlay
        non_zero_mask = np.any(erp_img > 0, axis=2)
        
        # Overlay the projection result onto the original ERP image
        overlaid_erp = original_erp.copy()
        overlaid_erp[non_zero_mask] = erp_img[non_zero_mask]
        
        # If switch is on, cover final result with top/bottom parts of original ERP
        if use_original_top_bottom:
            top_rows = int(erp_height * top_percentage)
            bottom_start_row = int(erp_height * (1 - bottom_percentage))
            
            # Cover with top/bottom pixels from original ERP
            overlaid_erp[:top_rows, :] = original_erp[:top_rows, :]
            overlaid_erp[bottom_start_row:, :] = original_erp[bottom_start_row:, :]
        
        # Save final stitched result
        output_dir = os.path.dirname(perspective_image_path)
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = f"{output_dir}/full_image_inpainted.png"
        success = cv2.imwrite(output_path, overlaid_erp)
        
        if success:
            print(f"Successfully saved: {output_path}")
            return True
        else:
            print("Failed to save image")
            return False
            
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        return False

def main():
    # Base directory
    base_dir = "./data/Matterport3D/mp3d_hf_filter_true_dino"
    
    if not os.path.exists(base_dir):
        print(f"Error: Base directory {base_dir} not found")
        return
    
    print(f"Starting to traverse directory: {base_dir}")
    
    # Statistics variables
    total_processed = 0
    total_success = 0
    
    # Iterate through all scenes
    for scene in os.listdir(base_dir):
        scene_path = os.path.join(base_dir, scene)
        if not os.path.isdir(scene_path):
            continue
            
        print(f"\nProcessing scene: {scene}")
        
        # Iterate through all views
        stitched_images_path = os.path.join(scene_path, "matterport_stitched_images")
        if not os.path.exists(stitched_images_path):
            print(f"Skipping scene {scene}: matterport_stitched_images directory not found")
            continue
            
        for view in os.listdir(stitched_images_path):
            view_path = os.path.join(stitched_images_path, view)
            if not os.path.isdir(view_path):
                continue
                
            carries_results_path = os.path.join(view_path, "carries_results")
            if not os.path.exists(carries_results_path):
                print(f"Skipping view {scene}/{view}: carries_results directory not found")
                continue
            
            # Find all object directories
            obj_pattern = f"{view}_obj*"
            obj_dirs = glob.glob(os.path.join(carries_results_path, obj_pattern))
            
            for obj_dir in obj_dirs:
                if not os.path.isdir(obj_dir):
                    continue
                    
                # Extract object number
                obj_dirname = os.path.basename(obj_dir)
                if obj_dirname.startswith(f"{view}_obj"):
                    obj_n = obj_dirname[len(f"{view}_obj"):]
                else:
                    continue
                
                # Construct file paths
                perspective_image_path = os.path.join(obj_dir, "full_image_inpainted_perspective.png")
                json_path = os.path.join(obj_dir, f"{view}_obj{obj_n}_recaption.json")
                
                # Check if required files exist
                if not os.path.exists(json_path):
                    print(f"Skipping {scene}/{view}/obj{obj_n}: JSON file not found")
                    continue
                
                # Process single image
                total_processed += 1
                if process_single_image(perspective_image_path, json_path, scene, view, obj_n):
                    total_success += 1
    
    print(f"\n=== Processing Complete ===")
    print(f"Total processed: {total_processed}")
    print(f"Success count: {total_success}")
    print(f"Failure count: {total_processed - total_success}")
    if total_processed > 0:
        print(f"Success rate: {total_success/total_processed:.1%}")

if __name__ == "__main__":
    main()