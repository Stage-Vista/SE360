import numpy as np
from scipy.ndimage import map_coordinates
from . import utils

def p2e(p_img, fov_deg, u_deg, v_deg, h, w, in_rot_deg=0, mode='bilinear'):
    """
    Project a perspective view back to an equirectangular image.

    Args:
        p_img: Input perspective image (numpy ndarray, H, W, C or H, W).
        fov_deg: Field of view of the perspective view (degrees). Can be a scalar or a tuple (horizontal FOV, vertical FOV).
        u_deg: Horizontal viewing center (degrees), range [-180, 180].
        v_deg: Vertical viewing center (degrees), range [-90, 90].
        h: Height of the output equirectangular image.
        w: Width of the output equirectangular image.
        in_rot_deg: Roll rotation angle of the input perspective view (degrees), default is 0.
        mode: Interpolation mode ('bilinear' or 'nearest'), default is 'bilinear'.

    Returns:
        Output equirectangular image (numpy ndarray, h, w, C or h, w).
    """
    assert len(p_img.shape) in (2, 3)
    p_h, p_w = p_img.shape[:2]

    # Handle FOV input
    try:
        # Assume it is (h_fov, v_fov)
        h_fov_deg, v_fov_deg = fov_deg
    except TypeError:
        # Assume it is a single scalar, horizontal and vertical FOV are the same
        h_fov_deg, v_fov_deg = fov_deg, fov_deg

    h_fov = h_fov_deg * np.pi / 180.0
    v_fov = v_fov_deg * np.pi / 180.0

    # Convert angles to radians (Note: sign of u is opposite to definition in e2p, because this is the camera's angle)
    u_rad = -u_deg * np.pi / 180.0
    v_rad = v_deg * np.pi / 180.0
    in_rot_rad = in_rot_deg * np.pi / 180.0

    # Set interpolation order
    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode: {}'.format(mode))

    # 1. Generate (u, v) coordinate grid for output equirectangular image
    uv = utils.equirect_uvgrid(h, w)

    # 2. Convert (u, v) to 3D coordinates on unit sphere (World Coordinates)
    xyz_eq = utils.uv2unitxyz(uv) # Shape (h, w, 3)

    # 3. Calculate inverse rotation matrix (Rotate world coordinates to camera coordinate system)
    # Rotation order is opposite to xyzpers in e2p and angles are negated
    # e2p: out.dot(Rx).dot(Ry).dot(Ri)
    # p2e: xyz_eq.dot(Ri_inv).dot(Ry_inv).dot(Rx_inv)
    # R_inv = R.T
    Rx = utils.rotation_matrix(v_rad, [1, 0, 0]) # Pitch
    Ry = utils.rotation_matrix(u_rad, [0, 1, 0]) # Yaw

    # Calculate roll rotation axis (Same axis as Ri in e2p)
    # This is the direction of the Camera Z-axis after applying Pitch and Yaw
    roll_axis = np.array([0., 0., 1.]).dot(Rx).dot(Ry)
    Ri = utils.rotation_matrix(in_rot_rad, roll_axis) # Roll

    # Combine rotation matrices and invert (transpose)
    R = Rx @ Ry @ Ri
    R_inv = R.T

    # 4. Rotate equirectangular 3D coordinates to perspective camera coordinate system
    # (h, w, 3) -> (h*w, 3) @ (3, 3) -> (h*w, 3) -> (h, w, 3)
    xyz_pers = xyz_eq.reshape(-1, 3) @ R_inv
    xyz_pers = xyz_pers.reshape(h, w, 3)

    # 5. Project to camera image plane, calculate normalized coordinates and validity mask
    x_pers, y_pers, z_pers = np.split(xyz_pers, 3, axis=-1)

    # Mask: Only consider points in front of the camera (z > epsilon)
    mask_valid_z = (z_pers > 1e-8)[..., 0]

    # To prevent division by zero errors, set z of invalid points to 1 (these points will be masked out anyway)
    z_pers[~mask_valid_z] = 1.0

    # Normalized coordinates (x/z, y/z)
    x_norm = x_pers / z_pers
    y_norm = y_pers / z_pers

    # 6. Convert normalized coordinates to pixel coordinates in perspective view
    # Calculate max range of x, y coordinates in perspective view (based on FOV)
    x_max = np.tan(h_fov / 2.0)
    y_max = np.tan(v_fov / 2.0)

    # Convert to pixel coordinate system (center at (p_w/2 - 0.5, p_h/2 - 0.5))
    # x: [-x_max, x_max] -> [0, p_w-1]
    # y: [-y_max, y_max] -> [p_h-1, 0] (Note: y-axis is inverted)
    coor_x = (x_norm / (2 * x_max) + 0.5) * p_w - 0.5
    coor_y = (-y_norm / (2 * y_max) + 0.5) * p_h - 0.5 # y-axis inversion

    # Stack coordinates for map_coordinates (requires (dim, ...))
    coords = np.stack([coor_y[..., 0], coor_x[..., 0]], axis=0) # Shape (2, h, w)

    # 7. Sample perspective view
    # Initialize output image (fill background with 0)
    if len(p_img.shape) == 3:
        output_img = np.zeros((h, w, p_img.shape[2]), dtype=p_img.dtype)
        # Sample each channel
        for i in range(p_img.shape[2]):
            sampled_values = map_coordinates(p_img[..., i], coords,
                                             order=order, mode='constant', cval=0.0)
            # Apply mask, only fill valid pixels
            output_img[mask_valid_z, i] = sampled_values[mask_valid_z]
    else: # Single channel
        output_img = np.zeros((h, w), dtype=p_img.dtype)
        sampled_values = map_coordinates(p_img, coords,
                                         order=order, mode='constant', cval=0.0)
        # Apply mask
        output_img[mask_valid_z] = sampled_values[mask_valid_z]

    return output_img

# Example usage can be added here
if __name__ == '__main__':
    # Example: Create a dummy perspective view and project it back to equirectangular
    import cv2 # requires opencv-python

    # a. First generate a perspective view from an equirectangular image (using e2p)
    # Assume there is an equirectangular image e_img_path
    # e_img = cv2.imread(e_img_path)
    # h_e, w_e = e_img.shape[:2]
    # If no real equirectangular image, create a checkerboard instead
    h_e, w_e = 512, 1024
    e_img = np.kron([[0, 255] * (w_e//128), [255, 0] * (w_e//128)] * (h_e//128), np.ones((64, 64))).astype(np.uint8)
    e_img = cv2.cvtColor(e_img, cv2.COLOR_GRAY2BGR) # Convert to 3 channels

    fov_deg = (80, 80) # Horizontal, vertical FOV
    u_deg = 45       # Horizontal viewing angle
    v_deg = -20      # Vertical viewing angle
    p_h, p_w = 400, 500 # Perspective view dimensions

    # Import e2p (Assume it's in the same directory or installed)
    try:
        from .e2p import e2p
    except ImportError:
        # If running this file directly, try importing from current directory
        from e2p import e2p

    print("Generating perspective view...")
    pers_img = e2p(e_img, fov_deg, u_deg, v_deg, (p_h, p_w), mode='bilinear')
    pers_img = pers_img.astype(np.uint8)
    # cv2.imwrite("temp_perspective.png", pers_img)
    print("Perspective view generated.")

    # b. Use p2e to project perspective view back to equirectangular
    print("Projecting perspective back to equirectangular...")
    # Use the same parameters as used for generation
    reproj_e_img = p2e(pers_img, fov_deg, u_deg, v_deg, h_e, w_e, mode='bilinear')
    reproj_e_img = reproj_e_img.astype(np.uint8)
    print("Projection finished.")

    # c. Show results (optional)
    # Create a comparison image: left is part of original, right is reprojected
    combined_img = np.zeros_like(e_img)
    # Overlay the reprojected part (non-black area) onto combined_img
    mask_reproj = np.any(reproj_e_img > 0, axis=2) # Find non-black pixels
    combined_img[mask_reproj] = reproj_e_img[mask_reproj]

    # For comparison, you could display the original image on the other parts
    # (Here we simply overlay on combined_img, reprojection will cover some parts)
    # A clearer comparison might be side-by-side or showing just the reprojection
    # cv2.imshow("Original Equirectangular", e_img)
    cv2.imshow("Reprojected Equirectangular (Overlay)", combined_img)
    cv2.imshow("Generated Perspective", pers_img)
    print("Showing images. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()