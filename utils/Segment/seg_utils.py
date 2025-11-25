import numpy as np
from PIL import Image
import torch
import torchvision
import cv2
import os
import json
from utils.basic_utils import overlay_masks_detectron_style
from segment_anything import sam_model_registry, SamPredictor

def get_center_point(binary_mask):
    y_indices, x_indices = np.where(binary_mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        # 如果掩码为空，返回空数组
        return np.array([])
        
    # 计算边界框
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    
    # 计算几何中心
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    
    # 确保中心点在掩码内部
    # 如果计算的中心点不在掩码内，找到最近的掩码点
    if binary_mask[y_center, x_center] == 0:
        # 计算所有掩码点到几何中心的距离
        distances = np.sqrt((x_indices - x_center)**2 + (y_indices - y_center)**2)
        # 找到最近的点
        nearest_idx = np.argmin(distances)
        x_center = x_indices[nearest_idx]
        y_center = y_indices[nearest_idx]
    
    # 只返回中心点
    center_point = np.array([[x_center, y_center]])
    return center_point

def get_5_points(binary_mask):
    y_indices, x_indices = np.where(binary_mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        # 如果掩码为空，返回空数组
        return np.array([])
        
    # 计算边界框
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    
    # 计算几何中心
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    
    # 确保中心点在掩码内部
    # 如果计算的中心点不在掩码内，找到最近的掩码点
    if binary_mask[y_center, x_center] == 0:
        # 计算所有掩码点到几何中心的距离
        distances = np.sqrt((x_indices - x_center)**2 + (y_indices - y_center)**2)
        # 找到最近的点
        nearest_idx = np.argmin(distances)
        x_center = x_indices[nearest_idx]
        y_center = y_indices[nearest_idx]
    
    center_point = [x_center, y_center]
    
    # 获取水平方向的点
    center_y_mask = y_indices == y_center
    x_values_at_center = x_indices[center_y_mask]
    if len(x_values_at_center) > 0:
        x_left = x_values_at_center.min()
        x_right = x_values_at_center.max()
        left_point = [x_left, y_center]
        right_point = [x_right, y_center]
    else:
        # 如果水平中心线上没有点，选择最近的行
        dist_to_center = np.abs(y_indices - y_center)
        nearest_y = y_indices[np.argmin(dist_to_center)]
        nearest_y_mask = y_indices == nearest_y
        x_values = x_indices[nearest_y_mask]
        x_left = x_values.min()
        x_right = x_values.max()
        left_point = [x_left, nearest_y]
        right_point = [x_right, nearest_y]
    
    # 获取垂直方向的点
    center_x_mask = x_indices == x_center
    y_values_at_center = y_indices[center_x_mask]
    if len(y_values_at_center) > 0:
        y_top = y_values_at_center.min()
        y_bottom = y_values_at_center.max()
        top_point = [x_center, y_top]
        bottom_point = [x_center, y_bottom]
    else:
        # 如果垂直中心线上没有点，选择最近的列
        dist_to_center = np.abs(x_indices - x_center)
        nearest_x = x_indices[np.argmin(dist_to_center)]
        nearest_x_mask = x_indices == nearest_x
        y_values = y_indices[nearest_x_mask]
        y_top = y_values.min()
        y_bottom = y_values.max()
        top_point = [nearest_x, y_top]
        bottom_point = [nearest_x, y_bottom]
    
    # 返回5个点：左、上、中心、右、下
    points = np.array([left_point, top_point, center_point, right_point, bottom_point])
    return points

def get_3_points(binary_mask):
    y_indices, x_indices = np.where(binary_mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        # 如果掩码为空，返回空数组
        return np.array([])
        
    # 计算边界框
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    
    # 计算宽度和高度
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    
    # 计算几何中心
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    
    # 确保中心点在掩码内部
    # 如果计算的中心点不在掩码内，找到最近的掩码点
    if binary_mask[y_center, x_center] == 0:
        # 计算所有掩码点到几何中心的距离
        distances = np.sqrt((x_indices - x_center)**2 + (y_indices - y_center)**2)
        # 找到最近的点
        nearest_idx = np.argmin(distances)
        x_center = x_indices[nearest_idx]
        y_center = y_indices[nearest_idx]
    
    center_point = [x_center, y_center]
    
    # 根据宽高比决定返回水平方向还是垂直方向的点
    if width > height:  # 如果宽度大于高度，返回左中右
        # 获取水平方向的点
        center_y_mask = y_indices == y_center
        x_values_at_center = x_indices[center_y_mask]
        if len(x_values_at_center) > 0:
            x_left = x_values_at_center.min()
            x_right = x_values_at_center.max()
            left_point = [x_left, y_center]
            right_point = [x_right, y_center]
        else:
            # 如果水平中心线上没有点，选择最近的行
            dist_to_center = np.abs(y_indices - y_center)
            nearest_y = y_indices[np.argmin(dist_to_center)]
            nearest_y_mask = y_indices == nearest_y
            x_values = x_indices[nearest_y_mask]
            x_left = x_values.min()
            x_right = x_values.max()
            left_point = [x_left, nearest_y]
            right_point = [x_right, nearest_y]
        
        # 返回左中右三点
        points = np.array([left_point, center_point, right_point])
    else:  # 如果高度大于或等于宽度，返回上中下
        # 获取垂直方向的点
        center_x_mask = x_indices == x_center
        y_values_at_center = y_indices[center_x_mask]
        if len(y_values_at_center) > 0:
            y_top = y_values_at_center.min()
            y_bottom = y_values_at_center.max()
            top_point = [x_center, y_top]
            bottom_point = [x_center, y_bottom]
        else:
            # 如果垂直中心线上没有点，选择最近的列
            dist_to_center = np.abs(x_indices - x_center)
            nearest_x = x_indices[np.argmin(dist_to_center)]
            nearest_x_mask = x_indices == nearest_x
            y_values = y_indices[nearest_x_mask]
            y_top = y_values.min()
            y_bottom = y_values.max()
            top_point = [nearest_x, y_top]
            bottom_point = [nearest_x, y_bottom]
        
        # 返回上中下三点
        points = np.array([top_point, center_point, bottom_point])
    
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
        points = get_center_point(refined_mask)
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
    
    # 初始化空的检测结果列表
    all_xyxy = []
    all_confidence = []
    all_class_id = []
    
    # 单独检测每个对象
    for idx, obj in enumerate(objects_to_seg):
        # 使用predict_with_caption代替predict_with_classes
        detections, phrases = grounding_dino_model.predict_with_caption(
            image=input_image,
            caption=obj,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        # 如果检测到对象，添加到结果列表
        if len(detections.xyxy) > 0:
            all_xyxy.append(detections.xyxy)
            all_confidence.append(detections.confidence)
            # 为所有检测到的实例分配相同的类别ID
            all_class_id.extend([idx] * len(detections.xyxy))
    
    # 检查是否有检测结果
    if not all_xyxy:
        print("未检测到任何对象")
        # 返回空结果
        empty_masks = [np.zeros((input_image.shape[0], input_image.shape[1]), dtype=bool) for _ in objects_to_seg]
        empty_bboxes = [[0, 0, 0, 0] for _ in objects_to_seg]
        return empty_masks, empty_bboxes, []
    
    # 合并所有检测结果
    all_xyxy = np.vstack(all_xyxy) if all_xyxy else np.array([])
    all_confidence = np.concatenate(all_confidence) if all_confidence else np.array([])
    all_class_id = np.array(all_class_id)
    
    # 创建detections对象以保持代码兼容性
    class SimpleDetections:
        def __init__(self, xyxy, confidence, class_id):
            self.xyxy = xyxy
            self.confidence = confidence
            self.class_id = class_id
            self.mask = None
    
    detections = SimpleDetections(all_xyxy, all_confidence, all_class_id)
    
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

def predict_with_grounding_dino_caption(grounding_dino_model, image, objects_to_seg, box_threshold=0.25, text_threshold=0.25):
    NMS_THRESHOLD = 0.8
    # 初始化空的检测结果列表
    all_xyxy = []
    all_confidence = []
    all_class_id = []
    
    # 单独检测每个对象
    for idx, obj in enumerate(objects_to_seg):
        # 使用predict_with_caption代替predict_with_classes
        detections, phrases = grounding_dino_model.predict_with_caption(
            image=image,
            caption=obj,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        # 如果检测到对象，添加到结果列表
        if len(detections.xyxy) > 0:
            all_xyxy.append(detections.xyxy)
            all_confidence.append(detections.confidence)
            # 为所有检测到的实例分配相同的类别ID
            all_class_id.extend([idx] * len(detections.xyxy))
    
    # 检查是否有检测结果
    if not all_xyxy:
        print("未检测到任何对象")
        # 返回空结果
        return []  # 统一返回空列表
    
    # 合并所有检测结果
    all_xyxy = np.vstack(all_xyxy) if all_xyxy else np.array([])
    all_confidence = np.concatenate(all_confidence) if all_confidence else np.array([])
    all_class_id = np.array(all_class_id)
        # 创建detections对象以保持代码兼容性
    class SimpleDetections:
        def __init__(self, xyxy, confidence, class_id):
            self.xyxy = xyxy
            self.confidence = confidence
            self.class_id = class_id
            self.mask = None
    
    detections = SimpleDetections(all_xyxy, all_confidence, all_class_id)
    
    # 检查并打印原始检测结果
    # print("原始检测结果数量:", len(detections.xyxy))
    # print("原始class_id:", detections.class_id)
    
    # NMS post process
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()
    
    # print("NMS后保留的索引:", nms_idx)
    
    # 确保class_id的类型与索引兼容
    if isinstance(detections.class_id, list):
        detections.class_id = np.array(detections.class_id)
    
    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]
    
    # print("NMS后的class_id:", detections.class_id)
    results = []
    for i in range(len(detections.xyxy)):
        results.append({
            "bbox": detections.xyxy[i].tolist(),
            "score": float(detections.confidence[i]),
            "label": objects_to_seg[detections.class_id[i]]
        })

    return results  # 返回列表

def get_9_points(binary_mask):
    y_indices, x_indices = np.where(binary_mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        # 如果掩码为空，返回空数组
        return np.array([])
        
    # 计算边界框
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    
    # 计算几何中心
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    
    # 确保中心点在掩码内部
    # 如果计算的中心点不在掩码内，找到最近的掩码点
    if binary_mask[y_center, x_center] == 0:
        # 计算所有掩码点到几何中心的距离
        distances = np.sqrt((x_indices - x_center)**2 + (y_indices - y_center)**2)
        # 找到最近的点
        nearest_idx = np.argmin(distances)
        x_center = x_indices[nearest_idx]
        y_center = y_indices[nearest_idx]
    
    center_point = [x_center, y_center]
    
    # 获取水平方向的点
    center_y_mask = y_indices == y_center
    x_values_at_center = x_indices[center_y_mask]
    if len(x_values_at_center) > 0:
        x_left = x_values_at_center.min()
        x_right = x_values_at_center.max()
        left_point = [x_left, y_center]
        right_point = [x_right, y_center]
    else:
        # 如果水平中心线上没有点，选择最近的行
        dist_to_center = np.abs(y_indices - y_center)
        nearest_y = y_indices[np.argmin(dist_to_center)]
        nearest_y_mask = y_indices == nearest_y
        x_values = x_indices[nearest_y_mask]
        x_left = x_values.min()
        x_right = x_values.max()
        left_point = [x_left, nearest_y]
        right_point = [x_right, nearest_y]
    
    # 获取垂直方向的点
    center_x_mask = x_indices == x_center
    y_values_at_center = y_indices[center_x_mask]
    if len(y_values_at_center) > 0:
        y_top = y_values_at_center.min()
        y_bottom = y_values_at_center.max()
        top_point = [x_center, y_top]
        bottom_point = [x_center, y_bottom]
    else:
        # 如果垂直中心线上没有点，选择最近的列
        dist_to_center = np.abs(x_indices - x_center)
        nearest_x = x_indices[np.argmin(dist_to_center)]
        nearest_x_mask = x_indices == nearest_x
        y_values = y_indices[nearest_x_mask]
        y_top = y_values.min()
        y_bottom = y_values.max()
        top_point = [nearest_x, y_top]
        bottom_point = [nearest_x, y_bottom]
    
    # 添加四个角点
    # 将坐标点分成四个象限
    top_left_mask = (x_indices < x_center) & (y_indices < y_center)
    top_right_mask = (x_indices >= x_center) & (y_indices < y_center)
    bottom_left_mask = (x_indices < x_center) & (y_indices >= y_center)
    bottom_right_mask = (x_indices >= x_center) & (y_indices >= y_center)
    
    # 左上角点：在左上象限找到离左上角(x_min, y_min)最近的点
    if np.any(top_left_mask):
        tl_x, tl_y = x_indices[top_left_mask], y_indices[top_left_mask]
        tl_distances = np.sqrt((tl_x - x_min)**2 + (tl_y - y_min)**2)
        tl_idx = np.argmin(tl_distances)
        top_left_point = [tl_x[tl_idx], tl_y[tl_idx]]
    else:
        # 如果该象限没有点，取最近的点
        distances = np.sqrt((x_indices - x_min)**2 + (y_indices - y_min)**2)
        idx = np.argmin(distances)
        top_left_point = [x_indices[idx], y_indices[idx]]
    
    # 右上角点：在右上象限找到离右上角(x_max, y_min)最近的点
    if np.any(top_right_mask):
        tr_x, tr_y = x_indices[top_right_mask], y_indices[top_right_mask]
        tr_distances = np.sqrt((tr_x - x_max)**2 + (tr_y - y_min)**2)
        tr_idx = np.argmin(tr_distances)
        top_right_point = [tr_x[tr_idx], tr_y[tr_idx]]
    else:
        # 如果该象限没有点，取最近的点
        distances = np.sqrt((x_indices - x_max)**2 + (y_indices - y_min)**2)
        idx = np.argmin(distances)
        top_right_point = [x_indices[idx], y_indices[idx]]
    
    # 左下角点：在左下象限找到离左下角(x_min, y_max)最近的点
    if np.any(bottom_left_mask):
        bl_x, bl_y = x_indices[bottom_left_mask], y_indices[bottom_left_mask]
        bl_distances = np.sqrt((bl_x - x_min)**2 + (bl_y - y_max)**2)
        bl_idx = np.argmin(bl_distances)
        bottom_left_point = [bl_x[bl_idx], bl_y[bl_idx]]
    else:
        # 如果该象限没有点，取最近的点
        distances = np.sqrt((x_indices - x_min)**2 + (y_indices - y_max)**2)
        idx = np.argmin(distances)
        bottom_left_point = [x_indices[idx], y_indices[idx]]
    
    # 右下角点：在右下象限找到离右下角(x_max, y_max)最近的点
    if np.any(bottom_right_mask):
        br_x, br_y = x_indices[bottom_right_mask], y_indices[bottom_right_mask]
        br_distances = np.sqrt((br_x - x_max)**2 + (br_y - y_max)**2)
        br_idx = np.argmin(br_distances)
        bottom_right_point = [br_x[br_idx], br_y[br_idx]]
    else:
        # 如果该象限没有点，取最近的点
        distances = np.sqrt((x_indices - x_max)**2 + (y_indices - y_max)**2)
        idx = np.argmin(distances)
        bottom_right_point = [x_indices[idx], y_indices[idx]]
    
    # 返回9个点：左上、上、右上、左、中心、右、左下、下、右下
    points = np.array([
        top_left_point, top_point, top_right_point,
        left_point, center_point, right_point,
        bottom_left_point, bottom_point, bottom_right_point
    ])
    return points