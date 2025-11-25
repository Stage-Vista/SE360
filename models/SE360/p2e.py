import numpy as np
from scipy.ndimage import map_coordinates
from . import utils_360 as utils

def p2e(p_img, fov_deg, u_deg, v_deg, h, w, in_rot_deg=0, mode='bilinear'):

    assert len(p_img.shape) in (2, 3)
    p_h, p_w = p_img.shape[:2]

    # process FOV input
    try:
        # assume (h_fov, v_fov)
        h_fov_deg, v_fov_deg = fov_deg
    except TypeError:
        # assume single scalar, horizontal and vertical FOV are the same
        h_fov_deg, v_fov_deg = fov_deg, fov_deg

    h_fov = h_fov_deg * np.pi / 180.0
    v_fov = v_fov_deg * np.pi / 180.0

    # convert angle to radian (note u's sign is opposite to e2p's definition, because here is camera's angle)
    u_rad = -u_deg * np.pi / 180.0
    v_rad = v_deg * np.pi / 180.0
    in_rot_rad = in_rot_deg * np.pi / 180.0

    # Set the interpolation order
    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode: {}'.format(mode))

    # 1. generate (u, v) grid for output panorama
    uv = utils.equirect_uvgrid(h, w)

    # 2. convert (u, v) to 3D coordinates on unit sphere (world coordinate system)
    xyz_eq = utils.uv2unitxyz(uv) # Shape (h, w, 3)

    # 3. calculate inverse rotation matrix (rotate world coordinates to camera coordinate system)
    # opposite rotation order and angle to xyzpers in e2p
    # e2p: out.dot(Rx).dot(Ry).dot(Ri)
    # p2e: xyz_eq.dot(Ri_inv).dot(Ry_inv).dot(Rx_inv)
    # R_inv = R.T
    Rx = utils.rotation_matrix(v_rad, [1, 0, 0]) # Pitch
    Ry = utils.rotation_matrix(u_rad, [0, 1, 0]) # Yaw

    # calculate roll rotation axis (same as Ri in e2p)
    # this is the direction of camera Z axis after applying Pitch and Yaw
    roll_axis = np.array([0., 0., 1.]).dot(Rx).dot(Ry)
    Ri = utils.rotation_matrix(in_rot_rad, roll_axis) # Roll

    # combine rotation matrices and take inverse (transpose)
    R = Rx @ Ry @ Ri
    R_inv = R.T

    # 4. rotate panorama 3D coordinates to camera coordinate system
    # (h, w, 3) -> (h*w, 3) @ (3, 3) -> (h*w, 3) -> (h, w, 3)
    xyz_pers = xyz_eq.reshape(-1, 3) @ R_inv
    xyz_pers = xyz_pers.reshape(h, w, 3)

    # 5. project to camera image plane, calculate normalized coordinates and validity mask
    x_pers, y_pers, z_pers = np.split(xyz_pers, 3, axis=-1)

    # mask: only consider points in front of camera (z > epsilon)
    mask_valid_z = (z_pers > 1e-8)[..., 0]

    # set invalid points' z to 1 to avoid division by zero error (these points will be excluded by mask)
    z_pers[~mask_valid_z] = 1.0

    # normalize coordinates (x/z, y/z)
    x_norm = x_pers / z_pers
    y_norm = y_pers / z_pers

    # 6. convert normalized coordinates to pixel coordinates in perspective view
    # calculate maximum range of x, y coordinates in perspective view (based on FOV)
    x_max = np.tan(h_fov / 2.0)
    y_max = np.tan(v_fov / 2.0)

    # convert to pixel coordinates (center at (p_w/2 - 0.5, p_h/2 - 0.5))
    # x: [-x_max, x_max] -> [0, p_w-1]
    # y: [-y_max, y_max] -> [p_h-1, 0] (note y axis is reversed)
    coor_x = (x_norm / (2 * x_max) + 0.5) * p_w - 0.5
    coor_y = (-y_norm / (2 * y_max) + 0.5) * p_h - 0.5 # y axis is reversed

    # combine coordinates for map_coordinates (need (dim, ...))
    coords = np.stack([coor_y[..., 0], coor_x[..., 0]], axis=0) # Shape (2, h, w)

    # 7. sample perspective view
    # initialize output image (fill background with 0)
    if len(p_img.shape) == 3:
        output_img = np.zeros((h, w, p_img.shape[2]), dtype=p_img.dtype)
        # sample each channel
        for i in range(p_img.shape[2]):
            sampled_values = map_coordinates(p_img[..., i], coords,
                                             order=order, mode='constant', cval=0.0)
            # apply mask, only fill valid pixels
            output_img[mask_valid_z, i] = sampled_values[mask_valid_z]
    else: # single channel
        output_img = np.zeros((h, w), dtype=p_img.dtype)
        sampled_values = map_coordinates(p_img, coords,
                                         order=order, mode='constant', cval=0.0)
        # apply mask, only fill valid pixels
        output_img[mask_valid_z] = sampled_values[mask_valid_z]

    return output_img

# can add an example usage here
if __name__ == '__main__':
    # example: create a virtual perspective view and project it back to panorama
    import cv2 # need opencv-python

    # a. first generate a perspective view from panorama (using e2p)
    # assume there is a panorama e_img_path
    # e_img = cv2.imread(e_img_path)
    # h_e, w_e = e_img.shape[:2]
    # if there is no real panorama, create a checkerboard instead
    h_e, w_e = 512, 1024
    e_img = np.kron([[0, 255] * (w_e//128), [255, 0] * (w_e//128)] * (h_e//128), np.ones((64, 64))).astype(np.uint8)
    e_img = cv2.cvtColor(e_img, cv2.COLOR_GRAY2BGR) # convert to 3 channels

    fov_deg = (80, 80) # horizontal, vertical FOV
    u_deg = 45       # horizontal angle
    v_deg = -20      # vertical angle
    p_h, p_w = 400, 500 # perspective view size

    # import e2p (assume in the same directory or installed)
    try:
        from .e2p import e2p
    except ImportError:
        # if running this file directly, try to import from current directory
        from e2p import e2p

    print("Generating perspective view...")
    pers_img = e2p(e_img, fov_deg, u_deg, v_deg, (p_h, p_w), mode='bilinear')
    pers_img = pers_img.astype(np.uint8)
    # cv2.imwrite("temp_perspective.png", pers_img)
    print("Perspective view generated.")

    # b. use p2e to project perspective view back to panorama
    print("Projecting perspective back to equirectangular...")
    # use the same parameters as when generating
    reproj_e_img = p2e(pers_img, fov_deg, u_deg, v_deg, h_e, w_e, mode='bilinear')
    reproj_e_img = reproj_e_img.astype(np.uint8)
    print("Projection finished.")

    # c. display results (optional)
    # create a comparison image: left is part of original panorama, right is reprojected panorama
    combined_img = np.zeros_like(e_img)
    # add reprojected part (non-black pixels) to combined_img
    mask_reproj = np.any(reproj_e_img > 0, axis=2) # find non-black pixels
    combined_img[mask_reproj] = reproj_e_img[mask_reproj]

    # for comparison, show original image on the other half
    # (here simply put original image on combined_img, reproject will cover part of it)
    # clearer comparison might be side-by-side display or only show reprojected result
    # cv2.imshow("Original Equirectangular", e_img)
    cv2.imshow("Reprojected Equirectangular (Overlay)", combined_img)
    cv2.imshow("Generated Perspective", pers_img)
    print("Showing images. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()