import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.Segment.seg_utils import predict_with_grounding_dino_caption, initialize_ground_sam_models
import cv2
from utils import load_img_to_array
import torch
from torch.utils.data import Dataset
import glob
import json
from pathlib import Path
import time
import argparse

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Base data path
base_data_path = "./data/Matterport3D/mp3d_base"
# Image file base path
image_base_path = "./data/Matterport3D/mp3d_skybox"


# Initialize models (only needs to be done once)
grounding_dino_model, sam_predictor = initialize_ground_sam_models(device)


# Modify bbox visualization function
def visualize_bbox(image, detections):
    img_with_boxes = image.copy()
    
    for item in detections:
        # Get bbox coordinates
        bbox = item['bbox']  # This is a list containing 4 coordinate values
        x1, y1, x2, y2 = map(int, bbox)  # Convert coordinates to integers
        
        # Get label and score
        label = item['label']
        score = item['score']
        
        # Draw bounding box
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label and confidence (show only the first 20 characters to avoid overly long text)
        short_label = label[:20] + "..." if len(label) > 20 else label
        text = f"{short_label}: {score:.2f}"
        cv2.putText(img_with_boxes, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_with_boxes

# Create custom dataset class
class SceneViewDataset(Dataset):
    def __init__(self, base_data_path, image_base_path, regenerate_existing=False):
        self.base_data_path = base_data_path
        self.image_base_path = image_base_path
        self.regenerate_existing = regenerate_existing
        self.tasks = self.collect_tasks()
        
    def collect_tasks(self):
        tasks = []
        # Iterate through all scenes
        scene_paths = glob.glob(os.path.join(self.base_data_path, "*"))
        for scene_path in scene_paths:
            scene_id = os.path.basename(scene_path)
            
            # Get all viewpoint folders under this scene
            view_paths = glob.glob(os.path.join(scene_path, "matterport_stitched_images", "*"))
            
            for view_path in view_paths:
                view_id = os.path.basename(view_path)
                
                # Check if object description JSON file exists
                json_file_path = os.path.join(view_path, "description", "objects.json")
                if not os.path.exists(json_file_path):
                    continue
                    
                # Point to the image file at the new location
                image_path = os.path.join(self.image_base_path, scene_id, "matterport_stitched_images", f"{view_id}.png")
                if not os.path.exists(image_path):
                    continue
                
                # Create save directory for the viewpoint
                view_save_dir = os.path.join(self.base_data_path, scene_id, "matterport_stitched_images", view_id)
                bbox_output_path = os.path.join(view_save_dir, "bbox_dino.json")
                output_img_path = os.path.join(view_save_dir, "detection_result_dino.jpg")
                
                # If result files already exist, decide whether to skip based on regenerate_existing
                if not self.regenerate_existing and os.path.exists(bbox_output_path) and os.path.exists(output_img_path):
                    continue
                
                # Add to task list
                tasks.append((scene_id, view_id, json_file_path, image_path, bbox_output_path, output_img_path))
        
        return tasks
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        scene_id, view_id, json_file_path, image_path, bbox_output_path, output_img_path = self.tasks[idx]
        
        # Do not actually load the image here; only return paths and info to avoid excessive memory usage
        return {
            'scene_id': scene_id,
            'view_id': view_id,
            'json_file_path': json_file_path,
            'image_path': image_path,
            'bbox_output_path': bbox_output_path,
            'output_img_path': output_img_path
        }

# Function to process a single item
def process_single_item(item, grounding_dino_model, sam_predictor):
    scene_id = item['scene_id']
    view_id = item['view_id']
    json_file_path = item['json_file_path']
    image_path = item['image_path']
    bbox_output_path = item['bbox_output_path']
    output_img_path = item['output_img_path']

    print(f"\nProcessing scene: {scene_id}, View: {view_id}")

    # Load image
    try:
        image_rgb = load_img_to_array(image_path)
    except Exception as e:
        print(f"Cannot load image {image_path}: {e}")
        return False

    # Save original image dimensions
    original_height, original_width = image_rgb.shape[:2]
    resized_image = image_rgb # Use original image directly for detection

    # Load object information
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            objects_data = json.load(f)
        if not objects_data:
             print("Warning: JSON file is empty or no objects found")
             return False
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return False

    final_detections = [] # Store the highest confidence bbox that passes filtering for each description

    # Iterate through each object description in JSON
    for obj_info in objects_data:
        if "description" not in obj_info or not obj_info["description"]:
            print("Warning: Found an entry with missing or empty 'description', skipping")
            continue

        description = obj_info["description"]
        category = obj_info.get("category") # Get category

        print(f"\nDetecting description: '{description}'")

        # Detect for current description
        try:
            current_detections = predict_with_grounding_dino_caption(
                grounding_dino_model,
                resized_image, # Use original image
                [description], # Pass list containing a single description
                box_threshold=0.3,
                text_threshold=0.25
            )
        except Exception as e:
            print(f"Error detecting description '{description}': {e}")
            continue # Skip this description

        if not current_detections:
            print(f"  - No bounding boxes detected for '{description}'")
            continue

        # Find the detection result with highest confidence
        best_detection = max(current_detections, key=lambda x: x['score'])
        print(f"  - Found highest confidence result for '{description}': score={best_detection['score']:.2f}")

        # --- Apply filtering conditions ---
        bbox = best_detection["bbox"]
        x1, y1, x2, y2 = bbox

        # Check if bbox touches image boundaries (leaving a 30-pixel margin)
        if x1 <= 30 or y1 <= 30 or x2 >= original_width - 30 or y2 >= original_height - 30:
            print(f"  - Bbox touches boundary, skipping")
            continue

        # Calculate bbox width and height
        width = x2 - x1
        height = y2 - y1

        # Check if the longest side of the bbox exceeds 3/5 of the image height
        max_side_length = max(width, height)
        max_allowed_length = original_height * 3 / 5
        if max_side_length > max_allowed_length:
            print(f"  - Bbox longest side too long ({max_side_length:.2f} > {max_allowed_length:.2f}), skipping")
            continue

        # Calculate total image area
        image_area = original_width * original_height
        # Calculate bbox area and percentage of total image area
        bbox_area = width * height
        area_percentage = (bbox_area / image_area) * 100

        # Check if bbox area is within allowed range (0.3% ~ 40%)
        if area_percentage < 0.3:
            print(f"  - Bbox area too small ({area_percentage:.2f}%), skipping")
            continue
        elif area_percentage > 40:
            print(f"  - Bbox area too large ({area_percentage:.2f}%), skipping")
            continue

        # --- Filter passed ---
        print(f"  - Bbox passed all checks, keeping")
        # Add category to best_detection for later use
        best_detection["category"] = category
        # label is already the description, no need to modify
        final_detections.append(best_detection)

    # --- Loop ends ---

    # Check if there are any valid detection results
    if not final_detections:
        print("\nProcessing complete, but no valid detection results found")
        # Even if no results, might need to decide whether to create an empty file or marker based on requirements
        # Choosing not to create any file here and returning False
        return False

    print(f"\nFinal number of valid bounding boxes kept: {len(final_detections)}")

    # Execute visualization (using final_detections)
    img_with_boxes = visualize_bbox(resized_image, final_detections)

    # Resize result back to original size (kept just in case, though we didn't scale)
    img_with_boxes_original_size = cv2.resize(img_with_boxes, (original_width, original_height))

    # Save result image
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    # Convert image from RGB to BGR before saving
    img_to_save_bgr = cv2.cvtColor(img_with_boxes_original_size, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_img_path, img_to_save_bgr) # Save BGR image
    print(f"Detection result image saved to {output_img_path}")

    # Create dictionary to save results
    bbox_results = {
        "image_path": image_path,
        "detections": []
    }

    # Add each detection result
    for detection in final_detections:
        result = {
            "description": detection["label"], # label is description
            "category": detection.get("category"), # Get category from previously added field
            "bbox": detection["bbox"],
            "score": float(detection["score"])
        }
        bbox_results["detections"].append(result)

    # Save to file
    with open(bbox_output_path, 'w', encoding='utf-8') as f:
        json.dump(bbox_results, f, indent=2, ensure_ascii=False)
    print(f"Bounding box info saved to: {bbox_output_path}")

    return True

# Sequential process function
def process_all_tasks(regenerate_existing=False):
    print("Initializing models...")
    # Initialize models (only needs to be done once)
    grounding_dino_model, sam_predictor = initialize_ground_sam_models(device)
    
    print("Creating dataset...")
    dataset = SceneViewDataset(base_data_path, image_base_path, regenerate_existing)
    print(f"Collected {len(dataset)} processing tasks in total")
    
    if len(dataset) == 0:
        print("No tasks to process, program exiting")
        return
    
    # Record start time
    start_time = time.time()
    
    # Process each task sequentially (no DataLoader needed for sequential processing)
    completed_tasks = 0
    total_tasks = len(dataset)
    
    for idx in range(total_tasks):
        item = dataset[idx]
        
        success = process_single_item(item, grounding_dino_model, sam_predictor)
        if success:
            completed_tasks += 1
        print(f"Progress [{idx+1}/{total_tasks}]: Scene {item['scene_id']}, View {item['view_id']} - {'Success' if success else 'Skipped'}")
    
    # Calculate total elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nAll tasks processed. Completed {completed_tasks}/{total_tasks} tasks. Elapsed time: {elapsed_time:.2f} seconds")

# Main program start
if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Object detection using Grounding DINO (Sequential Processing)')
    parser.add_argument('--regenerate', default=False, action='store_true', help='Regenerate existing result files (skips by default)')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    regenerate_existing = args.regenerate
    
    print(f"Starting sequential processing")
    print(f"Regenerate existing results: {regenerate_existing}")
    
    process_all_tasks(regenerate_existing)