import numpy as np
from scipy.ndimage import map_coordinates
from . import utils

def p2e(p_img, fov_deg, u_deg, v_deg, h, w, in_rot_deg=0, mode='bilinear'):
    """
    将透视图投影回全景图。

    参数:
        p_img: 输入的透视图图像 (numpy ndarray, H, W, C 或 H, W)
        fov_deg: 透视图的视场角 (度)，可以是单个标量或 (水平FOV, 垂直FOV) 元组
        u_deg: 水平视角中心 (度)，范围 [-180, 180]
        v_deg: 垂直视角中心 (度)，范围 [-90, 90]
        h: 输出全景图的高度
        w: 输出全景图的宽度
        in_rot_deg: 输入透视图的滚动旋转角度 (度)，默认为 0
        mode: 插值模式 ('bilinear' 或 'nearest')，默认为 'bilinear'

    返回:
        输出的全景图 (numpy ndarray, h, w, C 或 h, w)
    """
    assert len(p_img.shape) in (2, 3)
    p_h, p_w = p_img.shape[:2]

    # 处理FOV输入
    try:
        # 假设是 (h_fov, v_fov)
        h_fov_deg, v_fov_deg = fov_deg
    except TypeError:
        # 假设是单个标量，水平和垂直FOV相同
        h_fov_deg, v_fov_deg = fov_deg, fov_deg

    h_fov = h_fov_deg * np.pi / 180.0
    v_fov = v_fov_deg * np.pi / 180.0

    # 角度转弧度 (注意 u 的符号与 e2p 中的定义相反，因为这里是相机本身的角度)
    u_rad = -u_deg * np.pi / 180.0
    v_rad = v_deg * np.pi / 180.0
    in_rot_rad = in_rot_deg * np.pi / 180.0

    # 设置插值阶数
    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode: {}'.format(mode))

    # 1. 生成输出全景图的 (u, v) 坐标网格
    uv = utils.equirect_uvgrid(h, w)

    # 2. 将 (u, v) 转换为单位球面上的 3D 坐标 (世界坐标系)
    xyz_eq = utils.uv2unitxyz(uv) # Shape (h, w, 3)

    # 3. 计算逆旋转矩阵 (将世界坐标旋转到相机坐标系)
    # 与 e2p 中 xyzpers 的旋转顺序相反且角度取反
    # e2p: out.dot(Rx).dot(Ry).dot(Ri)
    # p2e: xyz_eq.dot(Ri_inv).dot(Ry_inv).dot(Rx_inv)
    # R_inv = R.T
    Rx = utils.rotation_matrix(v_rad, [1, 0, 0]) # Pitch
    Ry = utils.rotation_matrix(u_rad, [0, 1, 0]) # Yaw

    # 计算滚动旋转轴 (与 e2p 中 Ri 的轴相同)
    # 这是相机 Z 轴在应用 Pitch 和 Yaw 后的方向
    roll_axis = np.array([0., 0., 1.]).dot(Rx).dot(Ry)
    Ri = utils.rotation_matrix(in_rot_rad, roll_axis) # Roll

    # 组合旋转矩阵并求逆 (转置)
    R = Rx @ Ry @ Ri
    R_inv = R.T

    # 4. 将全景图 3D 坐标旋转到透视相机的坐标系
    # (h, w, 3) -> (h*w, 3) @ (3, 3) -> (h*w, 3) -> (h, w, 3)
    xyz_pers = xyz_eq.reshape(-1, 3) @ R_inv
    xyz_pers = xyz_pers.reshape(h, w, 3)

    # 5. 投影到相机图像平面，计算归一化坐标和有效性掩码
    x_pers, y_pers, z_pers = np.split(xyz_pers, 3, axis=-1)

    # 掩码：只考虑相机前方 (z > epsilon) 的点
    mask_valid_z = (z_pers > 1e-8)[..., 0]

    # 为防止除零错误，将无效点的 z 设为 1 (这些点最终会被掩码排除)
    z_pers[~mask_valid_z] = 1.0

    # 归一化坐标 (x/z, y/z)
    x_norm = x_pers / z_pers
    y_norm = y_pers / z_pers

    # 6. 将归一化坐标转换为透视图中的像素坐标
    # 计算透视图中 x, y 坐标的最大范围 (基于 FOV)
    x_max = np.tan(h_fov / 2.0)
    y_max = np.tan(v_fov / 2.0)

    # 转换到像素坐标系 (中心为 (p_w/2 - 0.5, p_h/2 - 0.5))
    # x: [-x_max, x_max] -> [0, p_w-1]
    # y: [-y_max, y_max] -> [p_h-1, 0] (注意 y 轴反转)
    coor_x = (x_norm / (2 * x_max) + 0.5) * p_w - 0.5
    coor_y = (-y_norm / (2 * y_max) + 0.5) * p_h - 0.5 # y 轴反转

    # 组合坐标给 map_coordinates (需要 (dim, ...))
    coords = np.stack([coor_y[..., 0], coor_x[..., 0]], axis=0) # Shape (2, h, w)

    # 7. 采样透视图
    # 初始化输出图像 (用 0 填充背景)
    if len(p_img.shape) == 3:
        output_img = np.zeros((h, w, p_img.shape[2]), dtype=p_img.dtype)
        # 对每个通道进行采样
        for i in range(p_img.shape[2]):
            sampled_values = map_coordinates(p_img[..., i], coords,
                                             order=order, mode='constant', cval=0.0)
            # 应用掩码，只填充有效的像素
            output_img[mask_valid_z, i] = sampled_values[mask_valid_z]
    else: # 单通道
        output_img = np.zeros((h, w), dtype=p_img.dtype)
        sampled_values = map_coordinates(p_img, coords,
                                         order=order, mode='constant', cval=0.0)
        # 应用掩码
        output_img[mask_valid_z] = sampled_values[mask_valid_z]

    return output_img

# 可以在这里添加一个示例用法
if __name__ == '__main__':
    # 示例：创建一个虚拟透视图并将其投影回全景图
    import cv2 # 需要 opencv-python

    # a. 先从全景图生成一个透视图 (使用 e2p)
    # 假设有一个全景图 e_img_path
    # e_img = cv2.imread(e_img_path)
    # h_e, w_e = e_img.shape[:2]
    # 如果没有真实全景图，创建一个棋盘格代替
    h_e, w_e = 512, 1024
    e_img = np.kron([[0, 255] * (w_e//128), [255, 0] * (w_e//128)] * (h_e//128), np.ones((64, 64))).astype(np.uint8)
    e_img = cv2.cvtColor(e_img, cv2.COLOR_GRAY2BGR) # 转为 3 通道

    fov_deg = (80, 80) # 水平、垂直视场角
    u_deg = 45       # 水平视角
    v_deg = -20      # 垂直视角
    p_h, p_w = 400, 500 # 透视图尺寸

    # 导入 e2p (假设在同一目录下或已安装)
    try:
        from .e2p import e2p
    except ImportError:
        # 如果直接运行此文件，尝试从当前目录导入
        from e2p import e2p

    print("Generating perspective view...")
    pers_img = e2p(e_img, fov_deg, u_deg, v_deg, (p_h, p_w), mode='bilinear')
    pers_img = pers_img.astype(np.uint8)
    # cv2.imwrite("temp_perspective.png", pers_img)
    print("Perspective view generated.")

    # b. 使用 p2e 将透视图投影回全景图
    print("Projecting perspective back to equirectangular...")
    # 使用与生成时相同的参数
    reproj_e_img = p2e(pers_img, fov_deg, u_deg, v_deg, h_e, w_e, mode='bilinear')
    reproj_e_img = reproj_e_img.astype(np.uint8)
    print("Projection finished.")

    # c. 显示结果 (可选)
    # 创建一个对比图：左边是原始全景图的一部分，右边是重新投影的全景图
    combined_img = np.zeros_like(e_img)
    # 将重新投影的部分（非黑色区域）叠加到 combined_img 上
    mask_reproj = np.any(reproj_e_img > 0, axis=2) # 找到非黑像素
    combined_img[mask_reproj] = reproj_e_img[mask_reproj]

    # 为了对比，可以在另一半显示原始图像
    # (这里简单地将原始图像放在 combined_img 上，重投影会覆盖一部分)
    # 更清晰的对比可能是并排显示或只显示重投影结果
    # cv2.imshow("Original Equirectangular", e_img)
    cv2.imshow("Reprojected Equirectangular (Overlay)", combined_img)
    cv2.imshow("Generated Perspective", pers_img)
    print("Showing images. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()