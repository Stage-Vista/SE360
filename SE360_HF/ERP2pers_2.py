import json
import numpy as np
import cv2
import os
import sys
import glob
from pathlib import Path

# Add project root directory to Python path
# sys.path.append('/home/lab401/zhongfile/Project/SGEdit-code/Inpaint-Anything/flux')

from py360.e2p import e2p

def load_projection_params_from_json(json_path):
    """
    Load projection parameters from a JSON file.
    
    Args:
        json_path: Path to the JSON file.
        
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

def find_json_files(base_path):
    """
    Find all JSON files that meet the criteria.
    
    Args:
        base_path: Base directory path.
        
    Returns:
        list: List of JSON file paths.
    """
    json_files = []
    
    # Use glob to find all JSON files matching the pattern
    pattern = os.path.join(base_path, "**/matterport_stitched_images/**/carries_results/**/*_recaption.json")
    for json_path in glob.glob(pattern, recursive=True):
        # Verify filename format (should be [view]_obj[n]_recaption.json)
        filename = os.path.basename(json_path)
        if "_obj" in filename and filename.endswith("_recaption.json"):
            json_files.append(json_path)
    
    return json_files

def erp_to_perspective(erp_image_path, center_u_deg, center_v_deg, final_hfov_deg, final_vfov_deg, output_path, output_size=(1024, 1024)):
    """
    Project ERP image to perspective view.
    
    Args:
        erp_image_path: Path to the ERP image.
        center_u_deg: Projection center u angle (horizontal angle).
        center_v_deg: Projection center v angle (vertical angle).
        final_hfov_deg: Final horizontal field of view (FOV) in degrees.
        final_vfov_deg: Final vertical field of view (FOV) in degrees.
        output_path: Path to save the output image.
        output_size: Output image size (height, width).
    """
    # Read ERP image
    erp_img = cv2.imread(erp_image_path)
    if erp_img is None:
        raise ValueError(f"Cannot read image file: {erp_image_path}")
    
    print(f"ERP image size: {erp_img.shape}")
    print(f"Projection params: center_u={center_u_deg}°, center_v={center_v_deg}°, hfov={final_hfov_deg}°, vfov={final_vfov_deg}°")
    
    # Convert image from BGR to RGB (OpenCV default is BGR)
    erp_img_rgb = cv2.cvtColor(erp_img, cv2.COLOR_BGR2RGB)
    
    # Perform projection using e2p function
    # Note: e2p function expects a single FOV angle, using horizontal FOV here
    perspective_img = e2p(
        e_img=erp_img_rgb,
        fov_deg=final_hfov_deg,  # Use horizontal FOV
        u_deg=center_u_deg,
        v_deg=center_v_deg,
        out_hw=output_size,  # (height, width)
        in_rot_deg=0,
        mode='bilinear'
    )
    
    # Convert result back from RGB to BGR for saving
    perspective_img_bgr = cv2.cvtColor(perspective_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # Save result
    cv2.imwrite(output_path, perspective_img_bgr)
    print(f"Perspective view saved to: {output_path}")
    
    return perspective_img

def process_single_json(json_path):
    """
    Process a single JSON file.
    
    Args:
        json_path: Path to the JSON file.
        
    Returns:
        bool: True if processing was successful, False otherwise.
    """
    try:
        print(f"\nProcessing JSON file: {json_path}")
        
        # Load parameters from JSON
        center_u_deg, center_v_deg, final_hfov_deg, final_vfov_deg, erp_image_path = load_projection_params_from_json(json_path)
        
        # Expand final_hfov_deg by 1.2 times
        original_hfov = final_hfov_deg
        final_hfov_deg = final_hfov_deg * 1.2
        
        print(f"  center_u_deg: {center_u_deg}")
        print(f"  center_v_deg: {center_v_deg}")
        print(f"  Original final_hfov_deg: {original_hfov}")
        print(f"  Adjusted final_hfov_deg: {final_hfov_deg}")
        print(f"  final_vfov_deg: {final_vfov_deg}")
        print(f"  ERP Image Path: {erp_image_path}")
        
        # Check if ERP image exists
        if not os.path.exists(erp_image_path):
            print(f"  Warning: ERP image file does not exist: {erp_image_path}")
            return False
        
        # Generate output path - save as ori_perspective.png in the same folder as the JSON file
        output_path = os.path.join(os.path.dirname(json_path), "ori_perspective.png")
        
        # Execute projection
        perspective_img = erp_to_perspective(
            erp_image_path=erp_image_path,
            center_u_deg=center_u_deg,
            center_v_deg=center_v_deg,
            final_hfov_deg=final_hfov_deg,
            final_vfov_deg=final_vfov_deg,
            output_path=output_path
        )
        
        print(f"  Perspective projection completed! Output size: {perspective_img.shape}")
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function - Batch process JSON files.
    """
    # Base path
    base_path = "./data/Matterport3D/mp3d_hf_first"
    
    print(f"Start searching for JSON files...")
    print(f"Base path: {base_path}")
    
    # Find all JSON files
    json_files = find_json_files(base_path)
    
    if not json_files:
        print("No matching JSON files found")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    # Process each JSON file
    success_count = 0
    for i, json_path in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}] Processing file: {os.path.relpath(json_path, base_path)}")
        
        if process_single_json(json_path):
            success_count += 1
        else:
            print(f"  Processing failed")
    
    print(f"\nBatch processing completed!")
    print(f"Successfully processed: {success_count}/{len(json_files)} files")

if __name__ == "__main__":
    main()