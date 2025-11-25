
import torch.nn.functional as F
import numpy as np
import torch
from PIL import Image
import cv2

def load_torch_image(numpy_image):
    numpy_image = np.transpose(numpy_image, (2, 0, 1))
    image = torch.from_numpy(numpy_image)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    return image

def load_image(image_path, device):
    from torchvision.io import read_image
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image

def bbox2layouts(target_bboxes, reverse, phrases, out_dir="./", visualize=False, return_type = 'pt', sp_sz=64):
    layouts_ = []
    fg_mask = np.zeros((sp_sz,sp_sz), dtype=bool)
    bboxes = target_bboxes[::-1] if reverse else target_bboxes
    phrases = phrases[::-1] if reverse else phrases
    for bbox, name in zip(bboxes, phrases):
        blend_mask = np.zeros((sp_sz,sp_sz), dtype=bool)
        x1, y1, x2, y2 = list(map(lambda x: int(x*sp_sz), bbox))
        blend_mask[y1:y2, x1:x2] = 1
        blend_mask = blend_mask^(blend_mask&fg_mask)
        if visualize:
            Image.fromarray(blend_mask).save(f"{out_dir}/mask_{name}.png")
        layouts_.append(blend_mask)
        fg_mask = fg_mask | blend_mask
    
    layouts_ = layouts_[::-1] if reverse else layouts_
    if return_type == 'pt':
        layouts = [torch.FloatTensor(l).unsqueeze(0).unsqueeze(0).cuda() for l in layouts_]
        layouts = F.interpolate(torch.cat(layouts),(sp_sz,sp_sz),mode='nearest')
        return layouts
    elif return_type == 'np':
        return layouts_
    
def parse_text(tokenizer, prompt, detail_arr, phases):
    '''
    Input: 
    prompt: "a photo of <person-0>."
    detail_arr: ["He has dark hair."]
    
    Output:
    prompt: "a photo of <person-0>. He has dark hair."
    idx_to_alter: [4, ]
    supr_idxs: [[4, 6, 7, 8, 9, 10], ]
    '''
    
    prompt_arr = [prompt] + detail_arr
    tokens_arr = [[tokenizer.decode(i) for i in tokenizer.encode(p)] for p in prompt_arr]
    phases_arr = [[tokenizer.decode(i) for i in tokenizer.encode(p)][1:-1] for p in phases]
    
    prompt = " ".join(prompt_arr)
    prompt_tokens = [tokenizer.decode(i) for i in tokenizer.encode(prompt)]
    # print(dict(zip(range(len(prompt_tokens)), prompt_tokens)))
    
    prompt_tokens_split = []
    st = 1
    for i in range(len(tokens_arr)):
        ed = st + len(tokens_arr[i]) - 2 # <SOT>, <EOT>
        prompt_tokens_split.append((st, ed))
        st = ed
        
    key_desc_range, details_desc_range_arr = prompt_tokens_split[0], prompt_tokens_split[1:]
    # get supr_idxs from details
    supr_idxs = [list(range(*details_desc_range)) for details_desc_range in details_desc_range_arr]
    
    # get idxs_to_alter from key_desc_range
    idxs_to_alter = []
    for encode_phases in phases_arr:
        for i in range(*key_desc_range):
            if prompt_tokens[i:i+len(encode_phases)] == encode_phases:
                idxs_to_alter.append(list(range(i,i+len(encode_phases))))
                break
    
    for idx, idx_arr in zip(idxs_to_alter, supr_idxs):
        idx_arr.extend(idx)
    
    return prompt, supr_idxs

def region_prompts(tokenizer, detail_arr, phrases):
    '''
    Input: 
    phrases: ["<cat-1>", "<mug-0>"]
    detail_arr: ["It has a soft grey coat with subtle stripes and bright yellow-green eyes.",
                    "It is a red mug with white snowflake patterns."]
    
    Output:
    regional_prompts: 
        ["a <cat-1>. It has a soft grey coat with subtle stripes and bright yellow-green eyes.",
        "a <mug-0>. It is a red mug with white snowflake patterns."]
    supr_idxs: [[1, (detailed description idxs for cat)],
                [1, (detailed description idxs for mug)]]
    
    '''
    regional_prompts = []
    supr_idxs_list = []
    for detail, phrase in zip(detail_arr, phrases):
        rg_prompt = f"a {phrase}."
        # print(phrase, detail)
        rg_prompt, rg_supr_idxs = parse_text(tokenizer, rg_prompt, [detail], [phrase])
        regional_prompts.append(rg_prompt)
        supr_idxs_list.append(rg_supr_idxs)
    return regional_prompts, supr_idxs_list

def enlarge_bbox(bbox_arr, alpha=1.3):
    update_bbox = []
    if alpha > 1:
        for bbox in bbox_arr:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            new_width = width * alpha
            new_height = height * alpha
            x1_new = x1 - (new_width - width) / 2
            y1_new = y1 - (new_height - height) / 2
            x2_new = min(1, x1_new + new_width)
            y2_new = min(1, y1_new + new_height)
            x1_new = max(0, x1_new)
            y1_new = max(0, y1_new)
            update_bbox.append([x1_new, y1_new, x2_new, y2_new])
    else:
        update_bbox = bbox_arr
    update_bbox = np.array(update_bbox)*512
    return update_bbox

def generate_unique_color(index, total):
    """ Generate a unique color for each mask using HSV color space. """
    import colorsys
    hue = index / total
    rgb_color = colorsys.hsv_to_rgb(hue, 1, 1)  # Full saturation and value for vivid colors
    return tuple(int(c * 255) for c in rgb_color)

def overlay_masks_detectron_style(image, masks):
    """
    Overlay multiple masks on the image in Detectron style with unique colors.

    Parameters:
    image (PIL.Image): The image to overlay the masks on.
    masks (list of PIL.Image or list of np.ndarray): The masks to overlay. Can be PIL Images or NumPy arrays.

    Returns:
    PIL.Image: The image with the masks overlaid in Detectron style.
    """
    # Convert PIL Image to numpy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)

    total_masks = len(masks)
    
    # Process each mask
    for i, mask in enumerate(masks):
        # Convert PIL Image to numpy array if necessary
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
            
        if mask.dtype == "bool":
            mask = (255 * mask).astype(np.uint8)

        # Resize mask to match image size
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Generate a unique color for the mask
        color = generate_unique_color(i, total_masks)

        # Create a colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask_resized == 255] = color  # Assuming mask is binary

        # Blend the mask into the image
        image = cv2.addWeighted(image, 1.0, colored_mask, 0.4, 0)

        # Draw mask edges for better clarity
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, 2)

    # Convert back to PIL Image
    overlaid_image = Image.fromarray(image)

    return overlaid_image

def dilate_mask(mask, dilate_kernel, target_type="PIL"):
    if isinstance(mask, Image.Image):
        mask = np.array(mask).astype(float)
    if dilate_kernel > 0.0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(dilate_kernel), int(dilate_kernel)))
        mask = cv2.dilate(mask, kernel)
    if target_type == "PIL":
        mask = Image.fromarray((mask * 255).astype(np.uint8))
    return mask

def erase_on_image_level(image, mask):
    inpainted_image = cv2.inpaint(image, mask.astype(np.uint8), inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted_image
