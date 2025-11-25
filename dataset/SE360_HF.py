import os
import numpy as np
from glob import glob
from .PanoDataset import PanoDataset, PanoDataModule

class SE360_HF_Dataset(PanoDataset):
    def __init__(self, config, mode='train'):
        # First call parent constructor, but temporarily don't set result_dir
        original_result_dir = config.get('result_dir')
        self.test_function = config.get('test_function', 'add')
        if original_result_dir is not None:
            config = config.copy()
            config['result_dir'] = None
        
        super().__init__(config, mode)
        
        # If there's result_dir, perform custom filtering logic
        if original_result_dir is not None:
            self.result_dir = original_result_dir
            results = self.scan_results(self.result_dir)
            assert results, f"No results found in {self.result_dir}, forgot to set environment variable WANDB_RUN_ID?"
            
            # Perform flexible matching: match directory names based on hash part in view_id
            results_set = set(results)
            new_data = []
            for d in self.data:
                view_id = d['view_id']
                # Extract hash part from view_id (remove pano_ prefix)
                if view_id.startswith('pano_'):
                    hash_part = view_id[5:]  # Remove 'pano_' prefix
                    # Check if there are directories containing this hash
                    matching_dirs = [r for r in results_set if hash_part in r]
                    if matching_dirs:
                        # Update data item, record matching directory name
                        d['result_dir_name'] = matching_dirs[0]
                        new_data.append(d)
                else:
                    # If view_id is not in pano_ format, directly check if it matches
                    if view_id in results_set:
                        d['result_dir_name'] = view_id
                        new_data.append(d)
            
            if len(new_data) != len(self.data):
                print(f"WARNING: {len(self.data)-len(new_data)} views are missing in results folder {self.result_dir} for {self.mode} set.")
                self.data = list(new_data)
                self.data.sort()
        else:
            self.result_dir = None

    def load_split(self, mode):
        split_file = 'train.npy' if mode == 'train' else 'test.npy'
        split_dir = os.path.join(self.inpaint_data_dir, split_file)

        if os.path.exists(split_dir) and (mode == 'train' or mode == 'val'):
            new_data = []
            data = np.load(split_dir)
            if mode == 'train':
                for d in data:
                    _, scene_id,_, view_id, _, object_id, label = d.split('/') 
                    data_type = 'mp3d'
                    new_data.append({
                        'data_type': data_type,
                        'scene_id': scene_id,
                        'view_id': view_id,
                        'object_id': object_id,
                        'label': label,
                        'function': 'add',
                        'ref_img': [],
                        'ref_prompt': [],
                        
                    })
                    if not self.use_ref:
                        new_data.append({
                            'data_type': data_type,
                            'scene_id': scene_id,
                            'view_id': view_id,
                            'object_id': object_id,
                            'label': label,
                            'function': 'remove',
                            'ref_img': [],
                            'ref_prompt': [],
                        })
            elif mode == 'val':
                for d in data:
                    _, scene_id,_, view_id, _, object_id, label = d.split('/') 
                    data_type = 'mp3d'
                    if 'remove' in object_id:
                        function = 'remove'
                    else:
                        function = 'add'
                    if self.use_ref:
                        ref_img = "ref1.jpg"
                        ref_prompt = "ref1_instruction.txt"
                    else:
                        ref_img = []
                        ref_prompt = []
                    new_data.append({
                        'data_type': data_type,
                        'scene_id': scene_id,
                        'view_id': view_id,
                        'object_id': object_id,
                        'label': label,
                        'function': function,
                        'ref_img': ref_img,
                        'ref_prompt': ref_prompt,
                    })
        elif mode == 'predict' or mode == 'test':
            new_data = []
            prompts= []
            print(f"Scanning {self.predict_data_dir}...")
            for scene_id in sorted(os.listdir(self.predict_data_dir)):
                scene_path = os.path.join(self.predict_data_dir, scene_id)
                if not os.path.isdir(scene_path):
                    continue
                    # Traverse all viewpoint directories under scene folder
                for view_dir in os.listdir(scene_path):
                    view_path = os.path.join(scene_path, view_dir)
                    if os.path.isdir(view_path):
                        # Check if parent directory contains inpainted.png and original_input.png
                        parent_dir = os.path.dirname(view_path)
                        parent_files = os.listdir(parent_dir) if os.path.exists(parent_dir) else []
                        
                        # if "inpainted.png" in parent_files and "original_input.png" in parent_files and "objects.txt" in parent_files:
                            # Build relative path: remove_panorama/scene/view/subpath/inpainted.png
                        # Dynamically detect image format (png or jpg)
                        img_extensions = ['.png', '.jpg', '.jpeg']
                        inpaint_path = None
                        for ext in img_extensions:
                            potential_path = os.path.join(view_path, f"full_image_inpainted{ext}")
                            if os.path.exists(potential_path):
                                inpaint_path = potential_path
                                break
                        
                        # If no image file of any format is found, skip
                        if inpaint_path is None:
                            continue
                            
                        rel_path = os.path.relpath(inpaint_path, os.path.dirname(self.predict_data_dir))
                        rel_path = rel_path.replace('\\', '/')  # Ensure consistent path format
                        prompts.append(rel_path)

            for d in prompts:
                _, scene_id, view_id, img_name = d.split('/')
                ref_img_list = []
                instruction_list = []
                json_name = f"{view_id}_recaption.json"
                test_pers_prompt = f"{view_id}_description.txt"
                ori_erp_name = f"{view_id}_erp.jpg"
                if self.mode == 'predict':
                    ori_erp_name = img_name
                for file in os.listdir(os.path.join(self.predict_data_dir, scene_id, view_id)):
                    if file.startswith('ref') and (file.endswith('.png') or file.endswith('.jpg')):
                        ref_img_list.append(file)
                    if file.endswith('.txt') and file.startswith('caption'):
                        instruction_list.append(file)
                if len(ref_img_list) > 0 and self.use_ref:
                        for ref_img in ref_img_list:
                            ref_prompt = ref_img.split('.')[0] + '_instruction.txt'
                            new_data.append({
                                'scene_id': scene_id,
                                'view_id': view_id,
                                'img_name': img_name,
                                'ori_erp_name': ori_erp_name,
                                'instruction': [],
                                'ref_img': ref_img,
                                'ref_prompt': ref_prompt,
                                'json': json_name,
                                'test_pers_prompt': test_pers_prompt,
                                'function': 'add'
                            })
                else:
                    for instruction in instruction_list:
                        new_data.append({
                            'scene_id': scene_id,
                            'view_id': view_id,
                            'img_name': img_name,
                            'ori_erp_name': ori_erp_name,
                            'instruction': instruction,
                            'ref_img': [],
                            'ref_prompt': [],
                            'json': json_name,
                            'test_pers_prompt': test_pers_prompt,
                            'function': self.test_function
                        })
        else:
            raise FileNotFoundError(f"Cannot find split file: {split_dir}")

        # Limit dataset size
        # if mode == 'train':
        #     print(f"[Debug] Original training samples: {len(new_data)}")
        #     new_data = new_data[:200]  # Use only first 200 training samples
        #     print(f"[Debug] Using only first 200 samples for training")
        # elif mode in ['val', 'test']:
        #     print(f"[Debug] Original {mode} samples: {len(new_data)}")
        #     new_data = new_data  # Validation and test sets use all samples
        #     print(f"[Debug] Using all samples for {mode}")
        
        return new_data #9820 perspective views

    def scan_results(self, result_dir):
        # Scan all subdirectories in result_dir
        results = glob(os.path.join(result_dir, '*/'))
        parsed_results = []
        for r in results:
            # Directly use directory name as identifier, no special parsing
            dir_name = r.split('/')[-2]
            parsed_results.append(dir_name)
        return parsed_results

    def get_data(self, idx):
        data = self.data[idx].copy()
        if self.mode == 'predict' or self.mode == 'test':
            scene_id, view_id, img_name, ori_erp_name, instruction, ref_img, ref_prompt, json_name, test_pers_prompt = data['scene_id'], data['view_id'], data['img_name'], data['ori_erp_name'], data['instruction'], data['ref_img'], data['ref_prompt'], data['json'], data['test_pers_prompt']
        else:
            scene_id, view_id, object_id, simple_label, ref_img, ref_prompt = data['scene_id'], data['view_id'], data['object_id'], data['label'], data['ref_img'], data['ref_prompt']
            label = '_'.join(simple_label.split('_')[1:])
            if 'absolute' in label:
                data['rotation_type'] = 'absolute'
            elif 'relative' in label:
                data['rotation_type'] = 'relative'
        if self.mode == 'predict' and self.config['repeat_predict'] > 1:
            data['pano_id'] = f"{scene_id}_{view_id}_{data['repeat_id']:06d}"
        else:
            data['pano_id'] = f"{scene_id}_{view_id}"

        if self.mode != 'predict' and self.mode != 'test':
            # Path to each panoramic image
            data['pano_path'] = os.path.join(self.data_dir, scene_id, 'matterport_stitched_images',f"{view_id}.png") 

            # Dynamically detect image format (png or jpg)
            img_extensions = ['.png', '.jpg', '.jpeg']
            remove_pano_path = None
            for ext in img_extensions:
                potential_path = os.path.join(self.inpaint_data_dir, scene_id, 'matterport_stitched_images', view_id, 'carries_results', object_id, f'full_image_inpainted{ext}')
                if os.path.exists(potential_path):
                    remove_pano_path = potential_path
                    break
            
            data['remove_pano_path'] = remove_pano_path
            if self.mode == 'val':
                data['pano_path'] = data['remove_pano_path']

        if self.mode == 'predict' or self.mode == 'test':
            data['remove_pano_path'] = os.path.join(self.predict_data_dir, scene_id, view_id, img_name)
            data['pano_path'] = os.path.join(self.predict_data_dir, scene_id, view_id, ori_erp_name)


        if self.mode == 'train' and self.use_cubemap_prompt==False and self.only_pano==False:
            prompt = []
            # for i in range(8):
            #     degree = i * 45
            #     # Path to each perspective image description
            #     prompt_path = os.path.join(self.data_dir, scene_id, 'blip3', f"{view_id}_{degree}.txt")
            #     prompt.append(self.load_prompt(prompt_path))
            for i in range(20):
                # Path to each perspective image description
                prompt_path = os.path.join(self.data_dir, scene_id, 'Blip_pers', view_id,f"{view_id}_{i:02d}.txt")
                prompt.append(self.load_prompt(prompt_path))
            data['prompt'] = prompt
        # elif self.mode == 'train' and self.use_cubemap_prompt==True:
        #     prompt = []
        #     for i in range(10):
        #         # Path to each perspective image description
        #         prompt_path = os.path.join(self.eight_data_dir, scene_id, 'qwen_descriptions_8pers', view_id,f"{view_id}_{i:02d}.txt")
        #         prompt.append(self.load_prompt(prompt_path))
        #     data['prompt'] = prompt
        elif self.mode == 'test':
            data['prompt'] = [''] * 10
        elif self.mode == 'val' and self.use_cubemap_prompt==True:
            data['prompt'] = [''] * 10
        # elif self.mode == 'predict' and self.use_cubemap_prompt==True:
        #     prompt = []
        #     for i in range(10):
        #         # Path to each perspective image description
        #         prompt_path = os.path.join(self.eight_data_dir, scene_id, 'qwen_descriptions_8pers', view_id,f"{view_id}_{i:02d}.txt")
        #         prompt.append(self.load_prompt(prompt_path))
        #     data['prompt'] = prompt
        else:
            data['prompt'] = [''] * 10
        
        if self.mode == 'train':
            data['pano_simple_prompt_path'] = os.path.join(self.inpaint_data_dir, scene_id, 'matterport_stitched_images', view_id, 'carries_results', object_id, simple_label) #
            data['pano_mask_path'] = os.path.join(self.inpaint_data_dir, scene_id, 'matterport_stitched_images', view_id, 'carries_results', object_id, 'full_bbox_mask.png') #
            data['pano_prompt_path'] = os.path.join(self.inpaint_data_dir, scene_id, 'matterport_stitched_images', view_id, 'carries_results', object_id, label) #
            data['json_path'] = os.path.join(self.inpaint_data_dir, scene_id, 'matterport_stitched_images', view_id, 'carries_results', object_id, f'{object_id}_recaption.json')
        elif self.mode == 'predict' or self.mode == 'test':
            data['pano_mask_path'] = os.path.join(self.predict_data_dir, scene_id, view_id, "full_bbox_mask.png") #
            if len(ref_img) > 0:
                data['ref_img_path'] = os.path.join(self.predict_data_dir, scene_id, view_id, ref_img)
                data['pano_prompt_path'] = os.path.join(self.predict_data_dir, scene_id, view_id, ref_prompt)
                data['ref_pano_prompt_path'] = os.path.join(self.predict_data_dir, scene_id, view_id, ref_prompt)
            else:
                # If there's result_dir, try to find corresponding txt file from result_dir
                if self.result_dir is not None:
                    # Rebuild complete directory name
                    result_dir_name = data.get('result_dir_name')
                    if result_dir_name:
                        result_scene_dir = os.path.join(self.result_dir, result_dir_name)
                        if os.path.exists(result_scene_dir):
                            # Find corresponding txt file based on instruction filename
                            instruction_base = instruction.replace('.txt', '')  # Remove .txt extension
                            target_txt_file = os.path.join(result_scene_dir, f"*{instruction_base}*.txt")
                            matched_txt_files = glob(target_txt_file)
                            
                            if matched_txt_files:
                                # Use matched txt file
                                data['pano_prompt_path'] = matched_txt_files[0]
                            else:
                                # If no matching txt file found, try to find any txt file
                                txt_files = glob(os.path.join(result_scene_dir, '*.txt'))
                                if txt_files:
                                    data['pano_prompt_path'] = txt_files[0]
                                else:
                                    # If no txt file found, use instruction as default
                                    data['pano_prompt_path'] = os.path.join(self.predict_data_dir, scene_id, view_id, instruction)
                        else:
                            data['pano_prompt_path'] = os.path.join(self.predict_data_dir, scene_id, view_id, instruction)
                    else:
                        data['pano_prompt_path'] = os.path.join(self.predict_data_dir, scene_id, view_id, instruction) #
                    data['ref_img_path'] = []
                    data['json_path'] = os.path.join(self.predict_data_dir, scene_id, view_id, json_name)
                    data['test_pers_prompt_path'] = os.path.join(self.predict_data_dir, scene_id, view_id, test_pers_prompt)
                else:
                    data['pano_prompt_path'] = os.path.join(self.predict_data_dir, scene_id, view_id, instruction) #
                    data['ref_img_path'] = []
                    data['json_path'] = os.path.join(self.predict_data_dir, scene_id, view_id, json_name)
                    data['test_pers_prompt_path'] = os.path.join(self.predict_data_dir, scene_id, view_id, test_pers_prompt)
        else:
            data['pano_simple_prompt_path'] = os.path.join(self.inpaint_data_dir, scene_id, 'matterport_stitched_images', view_id, 'carries_results', object_id, simple_label) 
            data['pano_mask_path'] = os.path.join(self.inpaint_data_dir, scene_id, 'matterport_stitched_images', view_id, 'carries_results', object_id, 'full_bbox_mask.png') #
            data['pano_prompt_path'] = os.path.join(self.inpaint_data_dir, scene_id, 'matterport_stitched_images', view_id, 'carries_results', object_id, label)
            data['json_path'] = os.path.join(self.inpaint_data_dir, scene_id, 'matterport_stitched_images', view_id, 'carries_results', object_id, f'{object_id}_recaption.json')
            if data['ref_img'] and data['ref_prompt'] != []:
                data['ref_img_path'] = os.path.join(self.inpaint_data_dir, scene_id, 'matterport_stitched_images', view_id, 'carries_results', object_id, ref_img)
                data['ref_pano_prompt_path'] = os.path.join(self.inpaint_data_dir, scene_id, 'matterport_stitched_images', view_id, 'carries_results', object_id, ref_prompt)        

        # Process predicted images in result_dir
        if self.result_dir is not None:
            if self.mode == 'predict' or self.mode == 'test':
                # Use stored result_dir_name
                result_dir_name = data.get('result_dir_name')
                if result_dir_name:
                    result_scene_dir = os.path.join(self.result_dir, result_dir_name)
                    if os.path.exists(result_scene_dir):
                        # Match corresponding png file based on instruction
                        instruction_base = instruction.replace('.txt', '')  # Remove .txt extension
                        # Try to find png files containing instruction_base
                        png_files = glob(os.path.join(result_scene_dir, '*.png'))
                        matched_png_files = [f for f in png_files if instruction_base in os.path.basename(f)]
                        
                        if matched_png_files:
                            # Use matched png file
                            data['pano_pred_path'] = matched_png_files[0]
                        else:
                            # If no matching png file found, prioritize files containing instruction
                            instruction_files = [f for f in png_files if 'instruction' in os.path.basename(f)]
                            if instruction_files:
                                data['pano_pred_path'] = instruction_files[0]
                            else:
                                data['pano_pred_path'] = png_files[0] if png_files else os.path.join(self.result_dir, data['pano_id'], 'pano.png')
                    else:
                        # If directory doesn't exist, use default path
                        data['pano_pred_path'] = os.path.join(self.result_dir, data['pano_id'], 'pano.png')
                else:
                    # If no result_dir_name, use default path
                    data['pano_pred_path'] = os.path.join(self.result_dir, data['pano_id'], 'pano.png')
            else:
                # For other modes, use original logic
                data['pano_pred_path'] = os.path.join(self.result_dir, data['pano_id'], 'pano.png')
        return data


class SE360_HF(PanoDataModule):
    def __init__(
            self,
            data_dir: str = 'data/Matterport3D/mp3d_skybox',
            inpaint_data_dir: str = 'data/Matterport3D/SE360_HF',
            s3d_data_dir: str = None,
            s3d_inpaint_data_dir: str = None,
            predict_data_dir: str = 'data/Matterport3D/test_final_without_artifacts',
            *args,
            **kwargs
            ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.dataset_cls = SE360_HF_Dataset
