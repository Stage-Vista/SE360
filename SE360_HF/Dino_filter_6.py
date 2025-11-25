import os
import json
import shutil
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import cv2
from concurrent.futures import ThreadPoolExecutor
import threading
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm

class DinoFilter:
    def __init__(self):
        """
        Initialize DINOv2 filter
        """
        print("Initializing DINOv2 filter")
        # Thread-safe counters
        self.lock = threading.Lock()
        self.total_processed = 0
        self.total_copied_true = 0
        self.total_copied_false_in = 0
        self.total_copied_false_out = 0
        
        # Initialize DINOv2 model
        print("Loading DINOv2 model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load DINOv2 model
        self.model = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((518, 518)),  # Input size recommended by DINOv2
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("DINOv2 model loaded successfully")
    
    def compute_image_dino_similarity_dual(self, image1_path, image2_path, mask_path):
        """
        Calculate feature similarity between two images using DINOv2 
        (Calculate similarity separately for inside the mask and the border area)
        
        Args:
            image1_path: Path to the first image file
            image2_path: Path to the second image file
            mask_path: Path to the mask image file
            
        Returns:
            (mask_inside_similarity, border_similarity): Cosine similarity values for inside-mask and border areas 
            (between 0-1, where 1 means completely identical)
        """
        try:
            # Load images
            img1 = cv2.imread(image1_path)
            img2 = cv2.imread(image2_path)
            
            if img1 is None or img2 is None:
                print(f"Cannot load images: {image1_path} or {image2_path}")
                return 0.0, 0.0
            
            # Convert to RGB format
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            # Load mask image
            if not os.path.exists(mask_path):
                print(f"Mask file does not exist: {mask_path}")
                return 0.0, 0.0
                
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Cannot load mask image: {mask_path}")
                return 0.0, 0.0
            
            # Resize mask to match image dimensions
            if mask.shape[:2] != img1_rgb.shape[:2]:
                mask = cv2.resize(mask, (img1_rgb.shape[1], img1_rgb.shape[0]))
            
            # Convert mask to binary image (0 or 255)
            _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # Calculate mask area (for display info)
            mask_area = np.sum(mask_binary > 0)
            total_area = mask_binary.shape[0] * mask_binary.shape[1]
            print(f"Mask area info: {mask_area}/{total_area} pixels ({mask_area/total_area*100:.1f}%)")
            
            # Create 3-channel version of mask
            mask_3ch = np.stack([mask_binary, mask_binary, mask_binary], axis=2)
            
            # Create border mask (10% area at image edges)
            h, w = img1_rgb.shape[:2]
            border_h = int(h * 0.1)  # Border height (10%)
            border_w = int(w * 0.1)  # Border width (10%)
            
            # Create border mask
            border_mask = np.zeros((h, w), dtype=np.uint8)
            # Top border
            border_mask[0:border_h, :] = 255
            # Bottom border
            border_mask[h-border_h:h, :] = 255
            # Left border
            border_mask[:, 0:border_w] = 255
            # Right border
            border_mask[:, w-border_w:w] = 255
            
            # Create 3-channel version of border mask
            border_mask_3ch = np.stack([border_mask, border_mask, border_mask], axis=2)
            
            print(f"Border area info: Border width={border_w}px({border_w/w*100:.1f}%), Border height={border_h}px({border_h/h*100:.1f}%)")
            
            # 1. Calculate similarity inside the mask area
            img1_masked_inside = np.where(mask_3ch > 0, img1_rgb, 0)
            img2_masked_inside = np.where(mask_3ch > 0, img2_rgb, 0)
            
            # Convert to PIL images
            img1_inside_pil = Image.fromarray(img1_masked_inside.astype(np.uint8))
            img2_inside_pil = Image.fromarray(img2_masked_inside.astype(np.uint8))
            
            # Preprocess images
            img1_inside_tensor = self.transform(img1_inside_pil).unsqueeze(0).to(self.device)
            img2_inside_tensor = self.transform(img2_inside_pil).unsqueeze(0).to(self.device)
            
            # 2. Calculate similarity in the border area
            img1_masked_border = np.where(border_mask_3ch > 0, img1_rgb, 0)
            img2_masked_border = np.where(border_mask_3ch > 0, img2_rgb, 0)
            
            # Convert to PIL images
            img1_border_pil = Image.fromarray(img1_masked_border.astype(np.uint8))
            img2_border_pil = Image.fromarray(img2_masked_border.astype(np.uint8))
            
            # Preprocess images
            img1_border_tensor = self.transform(img1_border_pil).unsqueeze(0).to(self.device)
            img2_border_tensor = self.transform(img2_border_pil).unsqueeze(0).to(self.device)
            
            # Extract features and calculate similarity
            with torch.no_grad():
                print(f"    Extracting DINOv2 features (mask inside and border area)...")
                
                # Features inside mask
                features1_inside = self.model.forward_features(img1_inside_tensor)
                features2_inside = self.model.forward_features(img2_inside_tensor)
                feat1_inside = features1_inside[:, 0, :]  # [CLS] token
                feat2_inside = features2_inside[:, 0, :]  # [CLS] token
                similarity_inside = F.cosine_similarity(feat1_inside, feat2_inside, dim=1).item()
                similarity_inside = (similarity_inside + 1) / 2  # Convert to 0-1 range
                
                # Features in border area
                features1_border = self.model.forward_features(img1_border_tensor)
                features2_border = self.model.forward_features(img2_border_tensor)
                feat1_border = features1_border[:, 0, :]  # [CLS] token
                feat2_border = features2_border[:, 0, :]  # [CLS] token
                similarity_border = F.cosine_similarity(feat1_border, feat2_border, dim=1).item()
                similarity_border = (similarity_border + 1) / 2  # Convert to 0-1 range
            
            return similarity_inside, similarity_border
            
        except Exception as e:
            print(f"Error computing DINOv2 dual similarity - Image 1: {image1_path}, Image 2: {image2_path}, mask: {mask_path}, Error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, 0.0
    
    def process_single_scene(self, scene_dir, target_dir_true, target_dir_false_in, target_dir_false_out, mask_inside_threshold, mask_outside_threshold, pbar, require_perspective=False):
        """
        Process a single scene directory (for multi-threading)
        
        Args:
            scene_dir: Path to the scene directory
            target_dir_true: Target directory path (good quality repairs)
            target_dir_false_in: Target directory path (inside mask too similar)
            target_dir_false_out: Target directory path (border area too dissimilar)
            mask_inside_threshold: Inside-mask similarity threshold (values above this are considered false)
            mask_outside_threshold: Border area similarity threshold (values below this are considered false)
            pbar: Progress bar object
            
        Returns:
            (processed_count, copied_true_count, copied_false_in_count, copied_false_out_count)
        """
        scene_name = os.path.basename(scene_dir)
        scene_processed = 0
        scene_copied_true = 0
        scene_copied_false_in = 0
        scene_copied_false_out = 0
        
        # Update progress bar description
        pbar.set_description(f"Processing scene: {scene_name}")
        
        # Find matterport_stitched_images directory
        stitched_images_dir = os.path.join(scene_dir, "matterport_stitched_images")
        if not os.path.exists(stitched_images_dir):
            print(f"Skipping scene {scene_name}: matterport_stitched_images directory not found")
            pbar.update(1)
            return scene_processed, scene_copied_true, scene_copied_false_in, scene_copied_false_out
        
        # Find all view directories
        view_pattern = os.path.join(stitched_images_dir, "*")
        view_dirs = glob.glob(view_pattern)
        view_dirs = [d for d in view_dirs if os.path.isdir(d)]

        for view_dir in view_dirs:
            view_name = os.path.basename(view_dir)
            
            # Find carries_results directory
            carries_results_dir = os.path.join(view_dir, "carries_results")
            if not os.path.exists(carries_results_dir):
                continue
            
            # Process each obj folder
            processed, copied_true, copied_false_in, copied_false_out = self.process_carries_results(
                carries_results_dir, 
                scene_name, 
                view_name,
                target_dir_true,
                target_dir_false_in,
                target_dir_false_out,
                mask_inside_threshold,
                mask_outside_threshold,
                require_perspective
            )
            
            scene_processed += processed
            scene_copied_true += copied_true
            scene_copied_false_in += copied_false_in
            scene_copied_false_out += copied_false_out
        
        # Thread-safe statistics update
        with self.lock:
            self.total_processed += scene_processed
            self.total_copied_true += scene_copied_true
            self.total_copied_false_in += scene_copied_false_in
            self.total_copied_false_out += scene_copied_false_out
        
        # Update progress bar with current statistics
        pbar.set_postfix({
            'processed': self.total_processed,
            'true': self.total_copied_true,
            'false_in': self.total_copied_false_in,
            'false_out': self.total_copied_false_out
        })
        pbar.update(1)
        return scene_processed, scene_copied_true, scene_copied_false_in, scene_copied_false_out
    
    def process_directory(self, source_dir, target_dir_true, target_dir_false_in, target_dir_false_out, mask_inside_threshold=0.9, mask_outside_threshold=0.8, num_threads=4, require_perspective=False):
        """
        Process the entire directory structure (multi-threaded version)
        
        Args:
            source_dir: Source directory path
            target_dir_true: Target directory path (good quality repair objects)
            target_dir_false_in: Target directory path (inside mask too similar)
            target_dir_false_out: Target directory path (border area too dissimilar)
            mask_inside_threshold: Inside-mask similarity threshold (values above this are considered false)
            mask_outside_threshold: Border area similarity threshold (values below this are considered false)
            num_threads: Number of threads
        """
        # Create target directories
        os.makedirs(target_dir_true, exist_ok=True)
        os.makedirs(target_dir_false_in, exist_ok=True)
        os.makedirs(target_dir_false_out, exist_ok=True)
        
        # Find all scene directories
        scene_pattern = os.path.join(source_dir, "*")
        scene_dirs = glob.glob(scene_pattern)
        scene_dirs = [d for d in scene_dirs if os.path.isdir(d)]
        
        print(f"Found {len(scene_dirs)} scene directories")
        print(f"Processing with {num_threads} threads")
        
        # Reset counters
        self.total_processed = 0
        self.total_copied_true = 0
        self.total_copied_false_in = 0
        self.total_copied_false_out = 0
        
        # Process scenes using thread pool
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            with tqdm(total=len(scene_dirs), desc="Overall Progress", position=0, 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}') as pbar:
                # Submit all tasks
                futures = []
                for scene_dir in scene_dirs:
                    future = executor.submit(
                        self.process_single_scene,
                        scene_dir,
                        target_dir_true,
                        target_dir_false_in,
                        target_dir_false_out,
                        mask_inside_threshold,
                        mask_outside_threshold,
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
        
        print(f"\nProcessing complete!")
        print(f"Total processed: {self.total_processed} objects")
        print(f"Copied to dino_filter_true (Good quality repairs): {self.total_copied_true} objects")
        print(f"Copied to dino_filter_false_in (Inside mask too similar): {self.total_copied_false_in} objects")
        print(f"Copied to dino_filter_false_out (Border area too dissimilar): {self.total_copied_false_out} objects")
        print(f"Criteria: Inside-mask similarity <= {mask_inside_threshold} AND Border area similarity >= {mask_outside_threshold}")
    
    def process_carries_results(self, carries_results_dir, scene_name, view_name, target_dir_true, target_dir_false_in, target_dir_false_out, mask_inside_threshold, mask_outside_threshold, require_perspective=False):
        """
        Process a single carries_results directory
        
        Args:
            carries_results_dir: carries_results directory path
            scene_name: Scene name
            view_name: View name
            target_dir_true: Target root directory (Good quality repairs)
            target_dir_false_in: Target root directory (Inside mask too similar)
            target_dir_false_out: Target root directory (Border area too dissimilar)
            mask_inside_threshold: Inside-mask similarity threshold (values above this are considered false)
            mask_outside_threshold: Border area similarity threshold (values below this are considered false)
            
        Returns:
            (processed_count, copied_true_count, copied_false_in_count, copied_false_out_count)
        """
        processed_count = 0
        copied_true_count = 0
        copied_false_in_count = 0
        copied_false_out_count = 0
        
        # Find all obj folders - using view_name instead of scene_name
        obj_pattern = os.path.join(carries_results_dir, f"{view_name}_obj*")
        
        obj_dirs = glob.glob(obj_pattern)
        obj_dirs = [d for d in obj_dirs if os.path.isdir(d)]
        
        if len(obj_dirs) == 0:
            return processed_count, copied_true_count, copied_false_in_count, copied_false_out_count
        
        # Create object-level progress bar
        obj_pbar = tqdm(obj_dirs, desc=f"Processing {scene_name}/{view_name}", 
                       leave=False, position=1, 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for obj_dir in obj_pbar:
            obj_name = os.path.basename(obj_dir)
            
            # Build file paths
            inpainted_perspective_path = os.path.join(obj_dir, "full_image_inpainted_perspective.png")  # Inpainted perspective image
            ori_perspective_path = os.path.join(obj_dir, "ori_perspective.png")  # Original perspective image
            mask_perspective_path = os.path.join(obj_dir, "full_bbox_mask_perspective.png")  # Perspective image mask file
            json_path = os.path.join(obj_dir, f"{obj_name}_recaption.json")
            
            # Check if basic files exist
            if not os.path.exists(inpainted_perspective_path):
                print(f"Skipping {obj_name}: Inpainted perspective image does not exist - {inpainted_perspective_path}")
                continue
                
            if not os.path.exists(ori_perspective_path):
                print(f"Skipping {obj_name}: Original perspective image does not exist - {ori_perspective_path}")
                continue
                
            if not os.path.exists(mask_perspective_path):
                print(f"Skipping {obj_name}: Perspective image mask file does not exist - {mask_perspective_path}")
                continue
                
            if not os.path.exists(json_path):
                print(f"Skipping {obj_name}: JSON file does not exist")
                continue
            
            try:
                # Read JSON file to get original_category
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                original_category = data.get('original_category', '')
                
                # Update object progress bar description
                obj_pbar.set_description(f"Processing {scene_name}/{view_name} - {obj_name}")
                
                # Calculate DINOv2 feature similarity (separately for inside mask and border area)
                mask_inside_similarity, border_similarity = self.compute_image_dino_similarity_dual(
                    ori_perspective_path, inpainted_perspective_path, mask_perspective_path
                )
                processed_count += 1
                
                # Update object progress bar postfix info
                obj_pbar.set_postfix({
                    'in': f'{mask_inside_similarity:.3f}',
                    'border': f'{border_similarity:.3f}',
                    'category': original_category[:8] + '...' if len(original_category) > 8 else original_category
                })
                
                print(f"{obj_name}: Inside-mask Sim = {mask_inside_similarity:.3f}, Border Sim = {border_similarity:.3f}, Category = {original_category}")
                print(f"  Original Perspective: {ori_perspective_path}")
                print(f"  Inpainted Perspective: {inpainted_perspective_path}")
                print(f"  Perspective Mask: {mask_perspective_path}")
                
                # Decide which folder to copy to based on dual thresholds
                # Good quality repair: inside-mask similarity shouldn't be too high (implies repair effect) 
                # and border similarity shouldn't be too low (implies good quality).
                is_good_quality = (mask_inside_similarity <= mask_inside_threshold and 
                                 border_similarity >= mask_outside_threshold)
                
                is_mask_inside_too_similar = mask_inside_similarity > mask_inside_threshold
                is_border_too_different = border_similarity < mask_outside_threshold
                
                if is_good_quality:
                    # Good quality repair, copy to dino_filter_true folder
                    target_obj_dir = os.path.join(
                        target_dir_true, 
                        scene_name, 
                        "matterport_stitched_images", 
                        view_name, 
                        "carries_results",
                        obj_name
                    )
                    copied_true_count += 1
                    print(f"  -> Good quality repair, copied to dino_filter_true: {target_obj_dir}")
                elif is_mask_inside_too_similar and is_border_too_different:
                    # Both issues present, prioritizing mask-inside issue (repair effect not obvious)
                    target_obj_dir = os.path.join(
                        target_dir_false_in, 
                        scene_name, 
                        "matterport_stitched_images", 
                        view_name, 
                        "carries_results",
                        obj_name
                    )
                    copied_false_in_count += 1
                    print(f"  -> Inside mask too similar AND border area too dissimilar, copied to dino_filter_false_in: {target_obj_dir}")
                    print(f"     Reason: Inside-mask Sim({mask_inside_similarity:.3f})>{mask_inside_threshold} AND Border Sim({border_similarity:.3f})<{mask_outside_threshold}")
                elif is_mask_inside_too_similar:
                    # Only mask-inside issue (repair effect not obvious)
                    target_obj_dir = os.path.join(
                        target_dir_false_in, 
                        scene_name, 
                        "matterport_stitched_images", 
                        view_name, 
                        "carries_results",
                        obj_name
                    )
                    copied_false_in_count += 1
                    print(f"  -> Inside mask too similar, copied to dino_filter_false_in: {target_obj_dir}")
                    print(f"     Reason: Inside-mask Sim({mask_inside_similarity:.3f})>{mask_inside_threshold}")
                elif is_border_too_different:
                    # Only border area issue (poor repair quality)
                    target_obj_dir = os.path.join(
                        target_dir_false_out, 
                        scene_name, 
                        "matterport_stitched_images", 
                        view_name, 
                        "carries_results",
                        obj_name
                    )
                    copied_false_out_count += 1
                    print(f"  -> Border area too dissimilar, copied to dino_filter_false_out: {target_obj_dir}")
                    print(f"     Reason: Border Sim({border_similarity:.3f})<{mask_outside_threshold}")
                else:
                    # This case theoretically shouldn't happen
                    print(f"  -> Unknown case, skipping: Inside={mask_inside_similarity:.3f}, Border={border_similarity:.3f}")
                    continue
                
                # Create target directory structure and copy folder
                os.makedirs(os.path.dirname(target_obj_dir), exist_ok=True)
                
                # Copy entire obj folder
                if os.path.exists(target_obj_dir):
                    shutil.rmtree(target_obj_dir)
                shutil.copytree(obj_dir, target_obj_dir)
                    
            except Exception as e:
                print(f"Error processing {obj_name}: {e}")
                continue
        
        # Close object-level progress bar
        obj_pbar.close()
        
        return processed_count, copied_true_count, copied_false_in_count, copied_false_out_count

def main():
    parser = argparse.ArgumentParser(description='Filter image inpainting quality using DINOv2 dual thresholds, calculating similarity separately for inside-mask and border areas of perspective images (Multi-threaded version)')
    parser.add_argument('--source_dir', type=str, 
                        default='./data/Matterport3D/mp3d_hf_filter_true_ssim',
                        help='Source directory path')
    parser.add_argument('--target_dir_true', type=str,
                        default='./data/Matterport3D/mp3d_hf_filter_true_dino',
                        help='Target directory path (Good quality repair objects)')
    parser.add_argument('--target_dir_false_in', type=str,
                        default='./data/Matterport3D/mp3d_hf_filter_false_in_dino',
                        help='Target directory path (Objects with mask-inside area too similar)')
    parser.add_argument('--target_dir_false_out', type=str,
                        default='./data/Matterport3D/mp3d_hf_filter_false_out_dino',
                        help='Target directory path (Objects with border area too dissimilar)')
    parser.add_argument('--mask_inside_threshold', type=float, default=0.9,
                        help='Inside-mask similarity threshold (values above this are considered insufficient repair effect)')
    parser.add_argument('--mask_outside_threshold', type=float, default=0.8,
                        help='Border area similarity threshold (values below this are considered poor repair quality)')
    parser.add_argument('--num_threads', type=int, default=15,
                        help='Number of threads, suggested to be 1-2 times the number of CPU cores')
    parser.add_argument('--require_perspective', default=True, action='store_true',
                        help='Whether to require perspective image files to exist before processing (currently defaults to requiring ori_perspective.png, etc.)')
    
    args = parser.parse_args()
    
    print("=== DINOv2 Dual Threshold Filter (Multi-threaded - Based on Perspective Image Mask Inside/Border) ===")
    print(f"Source Directory: {args.source_dir}")
    print(f"Target Directory (Good repairs): {args.target_dir_true}")
    print(f"Target Directory (Mask-inside too similar): {args.target_dir_false_in}")
    print(f"Target Directory (Border too dissimilar): {args.target_dir_false_out}")
    print(f"Inside-mask Similarity Threshold: {args.mask_inside_threshold} (Values > this considered insufficient repair)")
    print(f"Border Similarity Threshold: {args.mask_outside_threshold} (Values < this considered poor quality)")
    print(f"Number of Threads: {args.num_threads}")
    print(f"Require Perspective Images: {'Yes' if args.require_perspective else 'No'}")
    print("Note: Using ori_perspective.png as original perspective image")
    print("Note: Calculating DINOv2 feature similarity with full_image_inpainted_perspective.png separately for inside-mask and border areas")
    print("Note: Border area is defined as the 10% width area around the image edges")
    print("Note: Using cosine similarity to calculate DINOv2 feature similarity")
    print("Criteria: Inside-mask Sim <= mask_inside_threshold AND Border Sim >= mask_outside_threshold is considered a Good Repair")
    
    # Check if source directory exists
    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory does not exist - {args.source_dir}")
        return
    
    # Initialize DINOv2 filter
    dino_filter = DinoFilter()
    
    # Start processing
    dino_filter.process_directory(args.source_dir, args.target_dir_true, args.target_dir_false_in, args.target_dir_false_out, args.mask_inside_threshold, args.mask_outside_threshold, args.num_threads, args.require_perspective)

if __name__ == "__main__":
    main()