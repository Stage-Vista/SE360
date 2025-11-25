import os
import json
import shutil
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
from skimage.metrics import structural_similarity as ssim
import cv2
from concurrent.futures import ThreadPoolExecutor
import threading

class SSIMFilter:
    def __init__(self):
        """
        Initialize SSIM Filter
        """
        print("Initializing SSIM Filter")
        # Thread-safe counters
        self.lock = threading.Lock()
        self.total_processed = 0
        self.total_copied_true = 0
        self.total_copied_false = 0
        self.total_border_modified = 0  # Number of objects with modified borders
    
    def create_border_mask(self, image_shape, border_percentage=0.1):
        """
        Create a mask for the border region (border_percentage for top, bottom, left, right)
        
        Args:
            image_shape: Image shape (height, width)
            border_percentage: Percentage of border region (default 0.1 means 10%)
            
        Returns:
            Binary mask of the border region
        """
        height, width = image_shape
        border_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Calculate border width
        border_h = int(height * border_percentage)
        border_w = int(width * border_percentage)
        
        # Top border
        border_mask[0:border_h, :] = 255
        # Bottom border
        border_mask[height-border_h:height, :] = 255
        # Left border
        border_mask[:, 0:border_w] = 255
        # Right border
        border_mask[:, width-border_w:width] = 255
        
        return border_mask
    
    def compute_image_ssim_similarity(self, image1_path, image2_path, mask_path=None, check_border=False):
        """
        Compute SSIM similarity between two images (within the mask region).
        
        Args:
            image1_path: Path to the first image file
            image2_path: Path to the second image file
            mask_path: Path to the mask image file; if provided, SSIM is calculated only within the mask region
            check_border: If True, also compute SSIM for border region (10% edge area)
            
        Returns:
            SSIM similarity value (between 0-1, where 1 means identical)
            If check_border=True, returns tuple (main_ssim, border_ssim)
        """
        try:
            # Load images
            img1 = cv2.imread(image1_path)
            img2 = cv2.imread(image2_path)
            
            if img1 is None or img2 is None:
                print(f"Cannot load images: {image1_path} or {image2_path}")
                return 0.0
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # Ensure image dimensions are the same
            if gray1.shape != gray2.shape:
                # Resize images to the same dimensions
                height = min(gray1.shape[0], gray2.shape[0])
                width = min(gray1.shape[1], gray2.shape[1])
                gray1 = cv2.resize(gray1, (width, height))
                gray2 = cv2.resize(gray2, (width, height))
            
            # If a mask is provided, calculate SSIM only within the mask region
            if mask_path and os.path.exists(mask_path):
                # Load mask image
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"Cannot load mask image: {mask_path}")
                    return 0.0
                
                # Resize mask to match image dimensions
                if mask.shape != gray1.shape:
                    mask = cv2.resize(mask, (gray1.shape[1], gray1.shape[0]))
                
                # Convert mask to binary image (0 or 255)
                _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                
                # Calculate mask area (for display info)
                mask_area = np.sum(mask_binary > 0)
                total_area = mask_binary.shape[0] * mask_binary.shape[1]
                
                # Extract pixels within the mask region
                mask_indices = mask_binary > 0
                pixels1 = gray1[mask_indices]
                pixels2 = gray2[mask_indices]
                
                # Reshape 1D pixel array to a shape suitable for SSIM calculation
                # Attempt to create a near-square region
                num_pixels = len(pixels1)
                side_length = int(np.sqrt(num_pixels))
                
                # If mask region is too small, use minimum viable size
                if side_length < 7:
                    # For very small mask regions, use simple mean difference as similarity metric
                    mean1 = np.mean(pixels1)
                    mean2 = np.mean(pixels2)
                    # Convert mean difference to a 0-1 similarity value
                    diff = abs(mean1 - mean2) / 255.0
                    similarity = max(0.0, 1.0 - diff)
                    print(f"Note: Small mask region ({num_pixels} pixels), using mean difference for similarity: {similarity:.3f}")
                    return similarity
                
                # Truncate pixels to form a rectangle
                pixels_to_use = side_length * side_length
                pixels1_rect = pixels1[:pixels_to_use].reshape(side_length, side_length)
                pixels2_rect = pixels2[:pixels_to_use].reshape(side_length, side_length)
                
                # Compute SSIM within the mask region
                ssim_value = ssim(pixels1_rect, pixels2_rect, data_range=255)
                print(f"Mask region info: {mask_area}/{total_area} pixels ({mask_area/total_area*100:.1f}%)")
                
            else:
                # If no mask, compute SSIM for the whole image
                ssim_value = ssim(gray1, gray2, data_range=255)
            
            # If checking border region is required
            if check_border:
                border_mask = self.create_border_mask(gray1.shape, border_percentage=0.1)
                border_indices = border_mask > 0
                border_pixels1 = gray1[border_indices]
                border_pixels2 = gray2[border_indices]
                
                # Reshape border pixels for SSIM calculation
                num_border_pixels = len(border_pixels1)
                side_length = int(np.sqrt(num_border_pixels))
                
                if side_length < 7:
                    # Border region too small, using mean difference
                    mean1 = np.mean(border_pixels1)
                    mean2 = np.mean(border_pixels2)
                    diff = abs(mean1 - mean2) / 255.0
                    border_ssim_value = max(0.0, 1.0 - diff)
                    print(f"Border region too small ({num_border_pixels} pixels), using mean difference: {border_ssim_value:.3f}")
                else:
                    pixels_to_use = side_length * side_length
                    border_pixels1_rect = border_pixels1[:pixels_to_use].reshape(side_length, side_length)
                    border_pixels2_rect = border_pixels2[:pixels_to_use].reshape(side_length, side_length)
                    border_ssim_value = ssim(border_pixels1_rect, border_pixels2_rect, data_range=255)
                    print(f"Border SSIM: {border_ssim_value:.3f} (Border pixels: {num_border_pixels})")
                
                return ssim_value, border_ssim_value
            
            return ssim_value
            
        except Exception as e:
            print(f"Error computing SSIM similarity - Image 1: {image1_path}, Image 2: {image2_path}, Mask: {mask_path}, Error: {e}")
            if check_border:
                return 0.0, 0.0
            return 0.0
    
    def process_single_scene(self, scene_dir, target_dir_true, target_dir_false, threshold, border_threshold, pbar, require_perspective=False):
        """
        Process a single scene directory (for multi-threading).
        
        Args:
            scene_dir: Path to scene directory
            target_dir_true: Target directory path (below threshold)
            target_dir_false: Target directory path (above/equal threshold)
            threshold: SSIM similarity threshold for mask region
            border_threshold: SSIM threshold for border region
            pbar: Progress bar object
            
        Returns:
            (processed_count, copied_true_count, copied_false_count, border_modified_count)
        """
        scene_name = os.path.basename(scene_dir)
        scene_processed = 0
        scene_copied_true = 0
        scene_copied_false = 0
        scene_border_modified = 0
        
        # Look for matterport_stitched_images directory
        stitched_images_dir = os.path.join(scene_dir, "matterport_stitched_images")
        if not os.path.exists(stitched_images_dir):
            print(f"Skipping scene {scene_name}: matterport_stitched_images directory not found")
            pbar.update(1)
            return scene_processed, scene_copied_true, scene_copied_false, scene_border_modified
        
        # Look for all view directories
        view_pattern = os.path.join(stitched_images_dir, "*")
        view_dirs = glob.glob(view_pattern)
        view_dirs = [d for d in view_dirs if os.path.isdir(d)]

        for view_dir in view_dirs:
            view_name = os.path.basename(view_dir)
            
            # Look for carries_results directory
            carries_results_dir = os.path.join(view_dir, "carries_results")
            if not os.path.exists(carries_results_dir):
                continue
            
            # Process each obj folder
            processed, copied_true, copied_false, border_modified = self.process_carries_results(
                carries_results_dir, 
                scene_name, 
                view_name,
                target_dir_true,
                target_dir_false,
                threshold,
                border_threshold,
                require_perspective
            )
            
            scene_processed += processed
            scene_copied_true += copied_true
            scene_copied_false += copied_false
            scene_border_modified += border_modified
        
        # Thread-safe statistics update
        with self.lock:
            self.total_processed += scene_processed
            self.total_copied_true += scene_copied_true
            self.total_copied_false += scene_copied_false
            self.total_border_modified += scene_border_modified
        
        pbar.update(1)
        return scene_processed, scene_copied_true, scene_copied_false, scene_border_modified
    
    def process_directory(self, source_dir, target_dir_true, target_dir_false, threshold=0.86, border_threshold=0.95, num_threads=4, require_perspective=False):
        """
        Process the entire directory structure (multi-threaded version).
        
        Args:
            source_dir: Source directory path
            target_dir_true: Target directory path (objects below threshold)
            target_dir_false: Target directory path (objects above/equal to threshold)
            threshold: SSIM similarity threshold for mask region (default: 0.86)
            border_threshold: SSIM threshold for border region (default: 0.95)
            num_threads: Number of threads
        """
        # Create target directories
        os.makedirs(target_dir_true, exist_ok=True)
        os.makedirs(target_dir_false, exist_ok=True)
        
        # Find all scene directories
        scene_pattern = os.path.join(source_dir, "*")
        scene_dirs = glob.glob(scene_pattern)
        scene_dirs = [d for d in scene_dirs if os.path.isdir(d)]
        
        print(f"Found {len(scene_dirs)} scene directories")
        print(f"Processing using {num_threads} threads")
        
        # Reset counters
        self.total_processed = 0
        self.total_copied_true = 0
        self.total_copied_false = 0
        self.total_border_modified = 0
        
        # Process scenes using thread pool
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            with tqdm(total=len(scene_dirs), desc="Processing scenes") as pbar:
                # Submit all tasks
                futures = []
                for scene_dir in scene_dirs:
                    future = executor.submit(
                        self.process_single_scene,
                        scene_dir,
                        target_dir_true,
                        target_dir_false,
                        threshold,
                        border_threshold,
                        pbar,
                        require_perspective
                    )
                    futures.append(future)
                
                # Wait for all tasks to complete
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error processing scene: {e}")
                        import traceback
                        traceback.print_exc()
        
        print(f"\n{'='*80}")
        print(f"Processing complete!")
        print(f"{'='*80}")
        print(f"Total processed: {self.total_processed} objects")
        print(f"\nClassification Results:")
        print(f"  ✅ filter_true (Success - Mask changed & Border unmodified): {self.total_copied_true} objects ({self.total_copied_true/self.total_processed*100:.1f}%)" if self.total_processed > 0 else "  ✅ filter_true: 0 objects")
        print(f"  ❌ filter_false (Failure - Mask unchanged or Border modified): {self.total_copied_false} objects ({self.total_copied_false/self.total_processed*100:.1f}%)" if self.total_processed > 0 else "  ❌ filter_false: 0 objects")
        if self.total_processed > 0:
            print(f"\nBorder Modification Statistics:")
            print(f"  Objects with border modifications (Border SSIM < {border_threshold}): {self.total_border_modified} objects ({self.total_border_modified/self.total_processed*100:.1f}%)")
        print(f"{'='*80}")
    
    def process_carries_results(self, carries_results_dir, scene_name, view_name, target_dir_true, target_dir_false, threshold, border_threshold, require_perspective=False):
        """
        Process a single carries_results directory.
        
        Args:
            carries_results_dir: Path to carries_results directory
            scene_name: Scene name
            view_name: View name
            target_dir_true: Target root directory (below threshold)
            target_dir_false: Target root directory (above/equal threshold)
            threshold: SSIM similarity threshold for mask region
            border_threshold: SSIM threshold for border region
            
        Returns:
            (processed_count, copied_true_count, copied_false_count, border_modified_count)
        """
        processed_count = 0
        copied_true_count = 0
        copied_false_count = 0
        border_modified_count = 0
        
        # Find all obj folders - use view_name instead of scene_name
        obj_pattern = os.path.join(carries_results_dir, f"{view_name}_obj*")
        
        obj_dirs = glob.glob(obj_pattern)
        obj_dirs = [d for d in obj_dirs if os.path.isdir(d)]
        print(f"Found {len(obj_dirs)} objects")
        
        for obj_dir in obj_dirs:
            obj_name = os.path.basename(obj_dir)
            
            # Construct file paths
            inpainted_image_path = os.path.join(obj_dir, "full_image_inpainted.png")  # Inpainted ERP image
            inpainted_perspective_path = os.path.join(obj_dir, "full_image_inpainted_perspective.png")  # Inpainted perspective image
            mask_path = os.path.join(obj_dir, "full_bbox_mask.png")  # Mask file
            json_path = os.path.join(obj_dir, f"{obj_name}_recaption.json")
            
            # Check if basic files exist
            if not os.path.exists(inpainted_image_path):
                print(f"Skipping {obj_name}: inpainted image file does not exist - {inpainted_image_path}")
                continue
                
            if not os.path.exists(mask_path):
                print(f"Skipping {obj_name}: mask file does not exist")
                continue
                
            if not os.path.exists(json_path):
                print(f"Skipping {obj_name}: JSON file does not exist")
                continue
            
            # Check optional files and record status
            has_perspective = os.path.exists(inpainted_perspective_path)
            
            # If perspective view is required but not found, skip
            if require_perspective and not has_perspective:
                print(f"Skipping {obj_name}: Perspective view required but not found - {inpainted_perspective_path}")
                continue
            
            try:
                # Read JSON file to get original_category and original image path
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                original_category = data.get('original_category', '')
                original_erp_image = data.get('original_erp_image', '')
                
                # Check if original image path exists
                if not original_erp_image:
                    print(f"Skipping {obj_name}: original_erp_image field not found in JSON")
                    continue
                
                # If it's a relative path, convert to absolute path
                if not os.path.isabs(original_erp_image):
                    # Assuming original_erp_image is relative to a base directory
                    # Here assuming it is relative to the dataset root
                    print(f"Warning {obj_name}: original_erp_image is a relative path: {original_erp_image}")
                
                original_image_path = original_erp_image
                
                # Check if original image file exists
                if not os.path.exists(original_image_path):
                    print(f"Skipping {obj_name}: Original ERP image does not exist - {original_image_path}")
                    continue
                
                # Calculate SSIM similarity (within mask region) and border SSIM
                result = self.compute_image_ssim_similarity(original_image_path, inpainted_image_path, mask_path, check_border=True)
                
                # Unpack results
                if isinstance(result, tuple):
                    ssim_similarity, border_ssim = result
                else:
                    ssim_similarity = result
                    border_ssim = None
                    
                processed_count += 1
                
                # Safe print to avoid formatting errors caused by special characters in category
                border_ssim_str = f"{border_ssim:.3f}" if border_ssim is not None else "N/A"
                print(f"{obj_name}: Mask region SSIM = {ssim_similarity:.3f}, Border SSIM = {border_ssim_str}, Category = ", end='')
                print(original_category)
                print(f"  Original Image: {original_image_path}")
                print(f"  Inpainted Image: {inpainted_image_path}")
                perspective_status = '✅' if has_perspective else '❌'
                perspective_info = inpainted_perspective_path if has_perspective else '(Not Found)'
                print(f"  Contains Perspective: {perspective_status} {perspective_info}")
                
                # Determine if border is modified (Border SSIM below threshold indicates significant modification)
                border_modified = border_ssim is not None and border_ssim < border_threshold
                if border_modified:
                    border_ssim_val = f"{border_ssim:.3f}"
                    print(f"  ⚠️ Warning: Border region modified (Border SSIM: {border_ssim_val} < {border_threshold})")
                    border_modified_count += 1
                
                # Classification Logic:
                # filter_true (Success): mask_ssim < threshold AND border_ssim >= border_threshold (or no border check)
                # filter_false (Failure): mask_ssim >= threshold OR border_ssim < border_threshold
                is_mask_pass = ssim_similarity < threshold
                is_border_pass = (border_ssim is None) or (border_ssim >= border_threshold)
                
                if is_mask_pass and is_border_pass:
                    # Mask region modified successfully AND border region unmodified, classified as success
                    target_obj_dir = os.path.join(
                        target_dir_true, 
                        scene_name, 
                        "matterport_stitched_images", 
                        view_name, 
                        "carries_results",
                        obj_name
                    )
                    copied_true_count += 1
                    print(f"  ✅ -> Copied to filter_true (Success): {target_obj_dir}")
                else:
                    # Mask region unchanged OR border region modified, classified as failure
                    target_obj_dir = os.path.join(
                        target_dir_false, 
                        scene_name, 
                        "matterport_stitched_images", 
                        view_name, 
                        "carries_results",
                        obj_name
                    )
                    copied_false_count += 1
                    fail_reason = []
                    if not is_mask_pass:
                        fail_reason.append(f"Mask SSIM >= {threshold}")
                    if not is_border_pass:
                        fail_reason.append(f"Border SSIM < {border_threshold}")
                    print(f"  ❌ -> Copied to filter_false (Failure): {target_obj_dir}")
                    print(f"     Failure reason: {' AND '.join(fail_reason)}")
                
                # Create target directory structure and copy folder
                os.makedirs(os.path.dirname(target_obj_dir), exist_ok=True)
                
                # Copy the entire obj folder
                if os.path.exists(target_obj_dir):
                    shutil.rmtree(target_obj_dir)
                shutil.copytree(obj_dir, target_obj_dir)
                
                # Save SSIM info to JSON file
                target_json_path = os.path.join(target_obj_dir, f"{obj_name}_recaption.json")
                try:
                    with open(target_json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    # Add SSIM info and classification results
                    json_data['mask_ssim'] = float(ssim_similarity)
                    if border_ssim is not None:
                        json_data['border_ssim'] = float(border_ssim)
                        json_data['border_modified'] = bool(border_modified)
                    json_data['filter_result'] = 'true' if is_mask_pass and is_border_pass else 'false'
                    json_data['filter_reason'] = {
                        'mask_pass': bool(is_mask_pass),
                        'border_pass': bool(is_border_pass)
                    }
                    
                    with open(target_json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, ensure_ascii=False, indent=2)
                        
                except Exception as e:
                    print(f"  Warning: Failed to save SSIM info to JSON - {e}")
                    
            except Exception as e:
                print(f"Error processing {obj_name}: {e}")
                continue
        
        return processed_count, copied_true_count, copied_false_count, border_modified_count

def main():
    parser = argparse.ArgumentParser(
        description='Filter image similarity using SSIM (Mask + Border check, Multi-threaded version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Classification Logic:
  filter_true (Success):  Mask SSIM < threshold AND Border SSIM >= border_threshold
  filter_false (Failure): Mask SSIM >= threshold OR Border SSIM < border_threshold
  
Note:
  - Success: Significant change in inpainted region, AND border region has not been modified.
  - Failure: Almost no change in inpainted region, OR border region has been modified.
        '''
    )
    parser.add_argument('--source_dir', type=str, 
                        default='./data/Matterport3D/mp3d_hf',
                        help='Source directory path')
    parser.add_argument('--target_dir_true', type=str,
                        default='./data/Matterport3D/mp3d_hf_filter_true_ssim',
                        help='Target directory for successful inpainting (Mask changed AND Border intact)')
    parser.add_argument('--target_dir_false', type=str,
                        default='./data/Matterport3D/mp3d_hf_filter_false_ssim',
                        help='Target directory for failed inpainting (Mask unchanged OR Border modified)')
    parser.add_argument('--threshold', type=float, default=0.86,
                        help='SSIM threshold for mask region - lower means more change required (default: 0.86)')
    parser.add_argument('--border_threshold', type=float, default=0.98,
                        help='SSIM threshold for border region - lower means more border modification allowed (default: 0.98)')
    parser.add_argument('--num_threads', type=int, default=20,
                        help='Number of threads, recommended to be 1-2 times the number of CPU cores')
    parser.add_argument('--require_perspective', default=True, action='store_true',
                        help='Whether to require full_image_inpainted_perspective.png file existence to process the object')
    
    args = parser.parse_args()
    
    print("=== SSIM Filter (Multi-threaded version - Based on mask region + Border check, ERP images) ===")
    print(f"Source Directory: {args.source_dir}")
    print(f"Target Directory (Success - filter_true): {args.target_dir_true}")
    print(f"Target Directory (Failure - filter_false): {args.target_dir_false}")
    print(f"Mask SSIM Threshold: {args.threshold}")
    print(f"Border SSIM Threshold: {args.border_threshold}")
    print(f"Thread Count: {args.num_threads}")
    print(f"Require Perspective: {'Yes' if args.require_perspective else 'No'}")
    print("\nClassification Logic:")
    print(f"  ✅ filter_true (Success): Mask SSIM < {args.threshold} AND Border SSIM >= {args.border_threshold}")
    print(f"  ❌ filter_false (Failure): Mask SSIM >= {args.threshold} OR Border SSIM < {args.border_threshold}")
    print("\nNote:")
    print("  - Using original image path from original_erp_image in JSON")
    print("  - Calculating SSIM similarity with full_image_inpainted.png within the mask region")
    print("  - Also checking border region (10% edge area) to detect boundary modifications")
    print("  - Supports mask regions of any size (small regions use mean difference algorithm)")
    
    # Check if source directory exists
    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory does not exist - {args.source_dir}")
        return
    
    # Initialize SSIM filter
    ssim_filter = SSIMFilter()
    
    # Start processing
    ssim_filter.process_directory(args.source_dir, args.target_dir_true, args.target_dir_false, args.threshold, args.border_threshold, args.num_threads, args.require_perspective)

if __name__ == "__main__":
    main()