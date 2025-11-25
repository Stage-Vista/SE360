import os
import shutil
import sys

def check_and_copy_skybox_image(base_dir, skybox_base_dir, target_dir):
    """
    遍历 base_dir 目录结构 ([scene]/matterport_stitched_images/[view]/...)
    检查 'full_image_inpainted.png' 文件是否存在于 carries_results/[view]_obj[n]/ 中。

    如果某个 [view] 下的所有 [view]_obj[n] 都不包含该文件，
    则尝试将对应的 skybox_base_dir/[scene]/matterport_stitched_images/[view].png 文件
    拷贝到 target_dir/[scene]/[view].png。

    Args:
        base_dir (str): 用于检查条件的目录路径 (e.g., .../mp3d_inpaint_all)
        skybox_base_dir (str): 包含源 skybox 图像的基础目录路径 (e.g., .../mp3d_skybox)
        target_dir (str): 拷贝文件的目标根目录路径 (e.g., .../testset)
    """
    check_target_filename = "full_image_inpainted.png"
    copied_count = 0
    skipped_count = 0
    error_count = 0
    source_missing_count = 0
    condition_met_count = 0 # Count how many views meet the copy condition

    print(f"开始扫描条件目录: {base_dir}")
    print(f"拷贝源基础目录 (Skybox): {skybox_base_dir}")
    print(f"目标拷贝目录: {target_dir}")
    print("-" * 30)

    # 1. 遍历 [scene] 文件夹 (在 base_dir 中)
    for scene_name in os.listdir(base_dir):
        scene_path_for_check = os.path.join(base_dir, scene_name)
        if not os.path.isdir(scene_path_for_check):
            continue

        stitched_images_path_for_check = os.path.join(scene_path_for_check, "matterport_stitched_images")
        if not os.path.isdir(stitched_images_path_for_check):
            continue

        # 2. 遍历 [view] 文件夹 (在 base_dir 中)
        for view_name in os.listdir(stitched_images_path_for_check):
            view_path_for_check = os.path.join(stitched_images_path_for_check, view_name)
            if not os.path.isdir(view_path_for_check):
                continue

            # --- 开始检查条件 ---
            carries_results_path = os.path.join(view_path_for_check, "carries_results")
            should_copy = False # Flag to indicate if copying is needed
            reason = ""         # Reason for copying

            if not os.path.isdir(carries_results_path):
                should_copy = True
                reason = "无 carries_results 目录"
            else:
                found_check_file_in_view = False
                obj_folders_exist = False
                for item_name in os.listdir(carries_results_path):
                    item_path = os.path.join(carries_results_path, item_name)
                    if os.path.isdir(item_path) and item_name.startswith(f"{view_name}_obj"):
                        obj_folders_exist = True
                        check_file_path = os.path.join(item_path, check_target_filename)
                        if os.path.isfile(check_file_path):
                            found_check_file_in_view = True
                            break

                if obj_folders_exist and not found_check_file_in_view:
                    should_copy = True
                    reason = f"{check_target_filename} 未在任何 obj 文件夹中找到"
                elif not obj_folders_exist and os.path.isdir(carries_results_path):
                    should_copy = True
                    reason = "carries_results 存在但为空或无 obj 文件夹"
            # --- 条件检查结束 ---


            # --- 执行拷贝操作 (如果 should_copy 为 True) ---
            if should_copy:
                condition_met_count += 1

                # --- !!! 修改点在这里 !!! ---
                # 构建正确的源文件路径，包含 scene 和 matterport_stitched_images
                source_file_name = f"{view_name}.png"
                source_file_path = os.path.join(
                    skybox_base_dir,             # /.../mp3d_skybox
                    scene_name,                  # [scene]
                    "matterport_stitched_images", # matterport_stitched_images
                    source_file_name             # [view].png
                )
                # --- 修改结束 ---

                # 构建目标目录路径和目标文件路径
                dest_scene_dir = os.path.join(target_dir, scene_name)
                dest_file_path = os.path.join(dest_scene_dir, source_file_name) # 目标文件名与源文件名相同

                # 检查源文件是否存在
                if not os.path.isfile(source_file_path):
                    print(f"警告：源文件未找到，无法拷贝: {source_file_path} (原因: {reason})")
                    source_missing_count += 1
                    continue # 处理下一个 view

                # 检查目标文件是否已存在
                if os.path.exists(dest_file_path):
                    print(f"警告：目标文件已存在，跳过拷贝: {dest_file_path} (原因: {reason})")
                    skipped_count += 1
                    continue # 处理下一个 view

                # 创建目标场景目录 (如果不存在)
                try:
                    os.makedirs(dest_scene_dir, exist_ok=True)
                except OSError as e:
                    print(f"错误：无法创建目标场景目录 {dest_scene_dir}: {e}", file=sys.stderr)
                    error_count += 1
                    continue # 处理下一个 view

                # 执行文件拷贝
                try:
                    print(f"拷贝文件 ({reason}): {source_file_path} -> {dest_file_path}")
                    shutil.copy2(source_file_path, dest_file_path) # copy2 保留元数据
                    copied_count += 1
                except Exception as e:
                    print(f"错误：拷贝文件时出错 {source_file_path} -> {dest_file_path}: {e}", file=sys.stderr)
                    error_count += 1

    print("-" * 30)
    print("扫描与拷贝完成。")
    print(f"满足拷贝条件的视图数量: {condition_met_count}")
    print(f"成功拷贝的文件数量: {copied_count}")
    print(f"因目标已存在而跳过的数量: {skipped_count}")
    print(f"因源文件缺失而跳过的数量: {source_missing_count}")
    print(f"发生错误数量: {error_count}")

# --- 配置路径 ---
# 用于检查条件的目录
base_directory = "/home/lab401/zhongfile/Project/PanFusion/data/Matterport3D/version2/mp3d_inpaint_all"
# 包含源 skybox 图像的基础目录路径 (注意变量名修改)
skybox_base_directory = "/home/lab401/zhongfile/Project/PanFusion/data/Matterport3D/mp3d_skybox"
# 拷贝文件的目标根目录
target_root_directory = "/home/lab401/zhongfile/Project/PanFusion/data/Matterport3D/testset"

# --- 执行检查和拷贝 ---
if __name__ == "__main__":
    if not os.path.isdir(base_directory):
        print(f"错误：基础条件目录 {base_directory} 不存在或不是一个目录。", file=sys.stderr)
    elif not os.path.isdir(skybox_base_directory):
        print(f"错误：Skybox 源基础目录 {skybox_base_directory} 不存在或不是一个目录。", file=sys.stderr)
    else:
        check_and_copy_skybox_image(base_directory, skybox_base_directory, target_root_directory)