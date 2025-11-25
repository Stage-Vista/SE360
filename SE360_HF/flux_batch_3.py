import requests
import base64
import json
import os
import time
import glob
from datetime import datetime
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class TaskInfo:
    """Data class for task information"""
    image_path: str
    task_id: str
    polling_url: str
    original_category: str
    start_time: float
    status: str = "submitted"  # submitted, processing, completed, failed
    result_url: Optional[str] = None
    error_message: Optional[str] = None

def find_image_files(base_path):
    """
    Find all matching ori_perspective.png files
    
    Args:
        base_path: Base path to search in
        
    Returns:
        list: List of image file paths
    """
    image_files = []
    
    # Use glob to find all image files matching the pattern
    pattern = os.path.join(base_path, "**/matterport_stitched_images/**/carries_results/**/ori_perspective.png")
    
    for image_path in glob.glob(pattern, recursive=True):
        image_files.append(image_path)
    
    return image_files

def get_json_path_from_image_path(image_path):
    """
    Get the corresponding JSON file path based on the image path
    
    Args:
        image_path: Image file path
        
    Returns:
        str: JSON file path
    """
    # Get the directory where the image file is located
    image_dir = os.path.dirname(image_path)
    
    # Extract the [view]_obj[n] part from the directory name
    dir_name = os.path.basename(image_dir)
    
    # The JSON filename should be [view]_obj[n]_recaption.json
    json_filename = f"{dir_name}_recaption.json"
    json_path = os.path.join(image_dir, json_filename)
    
    return json_path

def load_original_category_from_json(json_path):
    """
    Load original_category from the JSON file
    
    Args:
        json_path: JSON file path
        
    Returns:
        str: original_category value
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_category = data.get('original_category', '')
        return original_category
    except Exception as e:
        print(f"  Failed to read JSON file: {e}")
        return None

def encode_image_to_base64(image_path):
    """
    Encode image to base64
    
    Args:
        image_path: Image file path
        
    Returns:
        str: base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def submit_single_task(image_path: str, api_key: str) -> Optional[TaskInfo]:
    """
    Submit a single task to the API
    
    Args:
        image_path: Image file path
        api_key: API key
        
    Returns:
        TaskInfo: Task information object, or None if failed
    """
    try:
        print(f"Submitting task: {os.path.basename(image_path)}")
        
        # Get corresponding JSON file path
        json_path = get_json_path_from_image_path(image_path)
        
        if not os.path.exists(json_path):
            print(f"  JSON file does not exist: {json_path}")
            return None
        
        # Read original_category
        original_category = load_original_category_from_json(json_path)
        
        if not original_category:
            print(f"  Cannot read original_category")
            return None
        
        # Construct prompt - Handle special case for 'table'
        if "table" in original_category.lower() or "dining table" in original_category.lower():
            prompt = f"remove the {original_category} in the center of the image, if there are chairs, remove them too"
            print(f"  Handling special case for table: {prompt}")
        else:
            prompt = f"remove the {original_category} in the center of the image"
        
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"  Image file does not exist: {image_path}")
            return None
        
        # Encode image
        input_image_base64 = encode_image_to_base64(image_path)
        
        # API Call
        url = "https://api.bfl.ai/v1/flux-kontext-pro"
        
        payload = {
            "prompt": prompt,
            "input_image": input_image_base64,
            "seed": 42,
            "width": 1024,
            "height": 1024,
            "steps": 50,  
            "aspect_ratio": "1:1",
            "output_format": "png",
            "prompt_upsampling": False,
            "safety_tolerance": 1,
            "interval": 2
        }
        
        headers = {
            "x-key": api_key,
            "Content-Type": "application/json"
        }
        
        response = requests.request("POST", url, json=payload, headers=headers)
        
        if response.status_code == 200:
            response_data = response.json()
            
            # Check if polling_url exists
            if 'polling_url' in response_data and 'id' in response_data:
                task_info = TaskInfo(
                    image_path=image_path,
                    task_id=response_data['id'],
                    polling_url=response_data['polling_url'],
                    original_category=original_category,
                    start_time=time.time()
                )
                print(f"  ✅ Task submitted successfully, ID: {task_info.task_id}")
                return task_info
            else:
                print(f"  ❌ Polling URL not found in response")
                return None
                
        else:
            print(f"  ❌ API call failed, status code: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"  ❌ Error submitting task: {e}")
        return None

def check_task_status(task: TaskInfo, api_key: str) -> TaskInfo:
    """
    Check the status of a single task
    
    Args:
        task: Task information object
        api_key: API key
        
    Returns:
        TaskInfo: Updated task information object
    """
    try:
        poll_response = requests.get(task.polling_url, headers={"x-key": api_key})
        
        if poll_response.status_code == 200:
            poll_data = poll_response.json()
            
            # Check task status
            if poll_data.get('status') == 'Ready' and 'result' in poll_data:
                result = poll_data['result']
                
                # Get image URL
                if 'sample' in result:
                    task.status = "completed"
                    task.result_url = result['sample']
                else:
                    task.status = "failed"
                    task.error_message = "Image URL not found in response"
            
            # Check for errors
            elif 'error' in poll_data:
                task.status = "failed"
                task.error_message = poll_data['error']
            
            # If still processing
            else:
                task.status = "processing"
                
        else:
            task.status = "failed"
            task.error_message = f"Polling failed, status code: {poll_response.status_code}"
    
    except Exception as e:
        task.status = "failed"
        task.error_message = f"Error during polling: {e}"
    
    return task

def download_and_save_result(task: TaskInfo) -> bool:
    """
    Download and save task result
    
    Args:
        task: Completed task information object
        
    Returns:
        bool: Whether save was successful
    """
    try:
        if not task.result_url:
            return False
        
        # Download image
        img_response = requests.get(task.result_url)
        
        if img_response.status_code == 200:
            # Save to the directory where the image file is located, named full_image_inpainted.png
            output_path = os.path.join(os.path.dirname(task.image_path), "full_image_inpainted_perspective.png")
            
            # Save image
            with open(output_path, "wb") as f:
                f.write(img_response.content)
            
            print(f"  ✅ Image saved: {os.path.basename(task.image_path)}")
            return True
        else:
            print(f"  ❌ Failed to download image: {os.path.basename(task.image_path)}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error saving result: {e}")
        return False

def manage_concurrent_tasks(image_files: List[str], api_key: str, max_concurrent: int = 24):
    """
    Manage concurrent task processing
    
    Args:
        image_files: List of image file paths
        api_key: API key  
        max_concurrent: Maximum concurrent tasks
    """
    active_tasks: Dict[str, TaskInfo] = {}  # task_id -> TaskInfo
    pending_files = image_files.copy()
    completed_count = 0
    failed_count = 0
    skipped_count = 0
    
    print(f"Starting concurrent processing, max concurrent: {max_concurrent}")
    print(f"Total files to process: {len(image_files)}")
    
    while pending_files or active_tasks:
        # 1. Submit new tasks (until concurrency limit is reached)
        while len(active_tasks) < max_concurrent and pending_files:
            image_path = pending_files.pop(0)
            
            # First check if output file already exists
            output_path = os.path.join(os.path.dirname(image_path), "full_image_inpainted_perspective.png")
            if os.path.exists(output_path):
                skipped_count += 1
                print(f"  ⏭️ Output file already exists, skipping: {os.path.basename(image_path)}")
                continue
            
            task = submit_single_task(image_path, api_key)
            
            if task:
                active_tasks[task.task_id] = task
            else:
                failed_count += 1
                print(f"  ❌ Submission failed: {os.path.basename(image_path)}")
        
        if not active_tasks:
            break
        
        # 2. Check status of all active tasks
        print(f"\nChecking progress... Active tasks: {len(active_tasks)}, Waiting to submit: {len(pending_files)}")
        
        completed_tasks = []
        
        # Use thread pool to check task status in parallel
        with ThreadPoolExecutor(max_workers=min(len(active_tasks), 10)) as executor:
            # Submit all status check tasks
            future_to_task = {
                executor.submit(check_task_status, task, api_key): task_id 
                for task_id, task in active_tasks.items()
            }
            
            # Process results
            for future in as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    updated_task = future.result()
                    active_tasks[task_id] = updated_task
                    
                    if updated_task.status == "completed":
                        completed_tasks.append(task_id)
                    elif updated_task.status == "failed":
                        completed_tasks.append(task_id)
                        print(f"  ❌ Task failed: {os.path.basename(updated_task.image_path)} - {updated_task.error_message}")
                    
                except Exception as exc:
                    print(f"  ❌ Error checking task status: {exc}")
                    completed_tasks.append(task_id)
        
        # 3. Process completed tasks
        for task_id in completed_tasks:
            task = active_tasks.pop(task_id)
            
            if task.status == "completed":
                if download_and_save_result(task):
                    completed_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1
            
            # Show current progress
            total_processed = completed_count + failed_count + skipped_count
            print(f"  Progress: {total_processed}/{len(image_files)} (Success: {completed_count}, Failed: {failed_count}, Skipped: {skipped_count})")
        
        # 4. Check for timed-out tasks
        current_time = time.time()
        timeout_tasks = []
        
        for task_id, task in active_tasks.items():
            if current_time - task.start_time > 300:  # 5 minute timeout
                timeout_tasks.append(task_id)
        
        for task_id in timeout_tasks:
            task = active_tasks.pop(task_id)
            failed_count += 1
            print(f"  ⚠️ Task timed out: {os.path.basename(task.image_path)}")
        
        # 5. Wait a while before checking again
        if active_tasks:
            time.sleep(2)  # Check every 2 seconds
    
    print(f"\nBatch processing completed!")
    print(f"Successfully processed: {completed_count}/{len(image_files)} files")
    print(f"Failed: {failed_count}/{len(image_files)} files")
    print(f"Skipped: {skipped_count}/{len(image_files)} files")

def main():
    """
    Main function - Batch concurrent image processing
    """
    # Base path
    base_path = "./data/Matterport3D/mp3d_hf_first"
    
    # API Key
    api_key = "Your API key"  # Please replace with your actual API key
    
    # Maximum concurrent tasks
    max_concurrent = 10
    
    print(f"Starting search for image files...")
    print(f"Base path: {base_path}")
    
    # Find all image files
    image_files = find_image_files(base_path)
    
    if not image_files:
        print("No matching image files found")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # Process all images concurrently
    manage_concurrent_tasks(image_files, api_key, max_concurrent)

if __name__ == "__main__":
    main()