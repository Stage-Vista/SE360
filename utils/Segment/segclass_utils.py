import numpy as np
from PIL import Image
import torch
import torchvision
import cv2
import os
import json
from utils.basic_utils import overlay_masks_detectron_style
from segment_anything import sam_model_registry, SamPredictor

def get_points(binary_mask):
    y_indices, x_indices = np.where(binary_mask > 0)

    # Calculate x_min and x_max
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    x_center = (x_min + x_max) // 2  # Use integer division for center line
    y_center = (y_min + y_max) // 2 
    center_point = [x_center, y_center]

    width = x_max - x_min
    height = y_max - y_min
    if width > height:
        # Filter points where x equals x_center
        center_x_mask = x_indices == x_center
        y_values_at_center = y_indices[center_x_mask]

        # Find the lowest and highest y values at the center line
        y_lowest = y_values_at_center.min()
        y_highest = y_values_at_center.max()

        # Output the coordinates
        lowest_point = [x_center, y_lowest]
        highest_point = [x_center, y_highest]
    else:
        # Filter points where y equals y_center
        center_y_mask = y_indices == y_center
        x_values_at_center = x_indices[center_y_mask]
        
        # Find the lowest and highest x values at the center line
        x_lowest = x_values_at_center.min()
        x_highest = x_values_at_center.max()
        # Output the coordinates
        lowest_point = [x_lowest, y_center]
        highest_point = [x_highest, y_center]

    points = np.array([lowest_point, center_point, highest_point])
    return points

def segment(sam_predictor, image, xyxy):
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, _, _ = sam_predictor.predict(
            box=box,
            multimask_output=False
        )
        refined_mask = masks[0]
        points = get_points(refined_mask)
        refined_mask2, _, _ = sam_predictor.predict(
            point_coords=points,
            point_labels=np.ones(len(points)),
            mask_input=None,
            multimask_output=False,
        )
        refined_mask = np.logical_or(refined_mask, refined_mask2[0])
        result_masks.append(refined_mask)
    return np.array(result_masks)

def ground_segment(grounding_dino_model, sam_predictor, input_image, objects_to_seg, box_threshold, text_threshold):
    NMS_THRESHOLD = 0.8
    
    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=input_image,
        classes=objects_to_seg,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    
    # 检查并打印原始检测结果
    print("原始检测结果数量:", len(detections.xyxy))
    print("原始class_id:", detections.class_id)
    
    # NMS post process
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()
    
    print("NMS后保留的索引:", nms_idx)
    
    # 确保class_id的类型与索引兼容
    if isinstance(detections.class_id, list):
        detections.class_id = np.array(detections.class_id)
    
    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]
    
    print("NMS后的class_id:", detections.class_id)

    # Prompting SAM with detected boxes
    def sam_segment(sam_predictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    # convert detections to masks
    detections.mask = sam_segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )
    
    # 创建实例层级的结果

    instance_results = []
    for i in range(len(detections.xyxy)):
        xyxy = detections.xyxy[i]
        confidence = detections.confidence[i]
        class_id = detections.class_id[i]
        mask = detections.mask[i]
        
        label_name = objects_to_seg[class_id]
        print('label_name',label_name)
        # 为每个实例计算归一化边界框
        bbox = mask_to_normalized_bbox(mask)
        instance_results.append({
            "label": label_name,
            "mask": mask,
            "bbox": bbox,
            "confidence": float(confidence)
        })
    
    # 按置信度排序实例（从高到低）
    instance_results = sorted(instance_results, key=lambda x: x["confidence"], reverse=True)
    
    # 按类别组织结果
    class_masks = []
    class_bboxes = []
    for name in objects_to_seg:
        # 对于语义级别的兼容性，仍然创建每个类别的合并掩码
        # for inst in instance_results:
        #     print(inst['label'])
        instances = [inst for inst in instance_results if inst["label"] == name]
        if instances:
            # 合并相同类别的所有掩码（为了向后兼容）
            combined_mask = np.any(np.stack([inst["mask"] for inst in instances]), axis=0)
            class_masks.append(combined_mask)
            # 使用合并掩码的边界框
            combined_bbox = mask_to_normalized_bbox(combined_mask)
            class_bboxes.append(combined_bbox)
        else:
            class_masks.append(np.zeros((input_image.shape[0], input_image.shape[1]), dtype=bool))
            class_bboxes.append([0, 0, 0, 0])
    
    return class_masks, class_bboxes, instance_results

def mask_to_normalized_bbox(mask):
    # Find the rows and columns where the mask is non-zero
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # Find the minimum and maximum row and column indices
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Normalize the coordinates by the size of the mask
    height, width = mask.shape
    x1, x2 = cmin / width, cmax / width
    y1, y2 = rmin / height, rmax / height
    return (x1, y1, x2, y2)

def initialize_sam_model(device):
    # Segment-Anything checkpoint
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = "checkpoints/sam_vit_h_4b8939.pth"

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device)
    sam_predictor = SamPredictor(sam)
    return sam_predictor

def initialize_ground_sam_models(device, load_sam=True):
    from groundingdino.util.inference import Model
    from segment_anything import sam_model_registry, SamPredictor
    
    # GroundingDINO配置和检查点 - 使用相对于项目的正确路径
    GROUNDING_DINO_CONFIG_PATH = "utils/Segment/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "./checkpoints/groundingdino_swint_ogc.pth"

    # Segment-Anything检查点
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = "./checkpoints/sam_vit_h_4b8939.pth"

    # 构建GroundingDINO推理模型
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    # 如果不需要加载SAM模型，直接返回
    if not load_sam:
        return grounding_dino_model, None
    
    # 否则继续加载SAM模型
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    
    return grounding_dino_model, sam_predictor

def predict_with_grounding_dino(grounding_dino_model, image, objects_to_seg, box_threshold=0.25, text_threshold=0.25):
    """
    仅使用Grounding DINO模型进行物体检测
    
    Args:
        grounding_dino_model: 已加载的Grounding DINO模型
        image: RGB格式的输入图像
        objects_to_seg: 要检测的物体列表
        box_threshold: 边界框置信度阈值
        text_threshold: 文本匹配置信度阈值
        
    Returns:
        检测结果列表，每个元素包含边界框和置信度
    """
    # 使用Grounding DINO进行预测
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=objects_to_seg,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    
    # 提取结果
    results = []
    for i in range(len(detections.xyxy)):
        results.append({
            "bbox": detections.xyxy[i].tolist(),
            "score": float(detections.confidence[i]),
            "label": objects_to_seg[detections.class_id[i]]
        })
    
    return results


    