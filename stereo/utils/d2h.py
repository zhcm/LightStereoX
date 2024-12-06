import numpy as np


def parse_camera_pose(file_path):
    """
    解析相机位姿文件，返回位移向量和四元数。

    Args:
        file_path (str): 相机位姿文件路径

    Returns:
        camera_position (np.ndarray): 位移向量
        camera_orientation (quaternion.quaternion): 四元数表示的旋转
    """
    # 读取变换矩阵
    matrix = np.loadtxt(file_path)

    # 提取平移向量（变换矩阵的最后一列）
    position = matrix[:3, 3]
    # position[1], position[2] = position[2], position[1]
    # 提取旋转矩阵（前3x3部分）
    rotation_matrix = matrix[:3, :3]

    # 将旋转矩阵转换为四元数
    # camera_orientation = quaternion.from_rotation_matrix(rotation_matrix.T)

    return position, rotation_matrix


def depth_to_height(depth_image, hfov, camera_position, camera_orientation):
    """
    Converts depth image to a height map using camera parameters.

    Args:
        depth_image (np.ndarray): The input depth image.
        hfov (float): Horizontal field of view in degrees.
        camera_position (np.ndarray): The global position of the camera.
        camera_orientation (quaternion.quaternion): The camera's quaternion orientation.

    Returns:
        np.ndarray: Global height map derived from depth image.
    """
    img_height, img_width = depth_image.shape
    focal_length_px = img_width / (2 * np.tan(np.radians(hfov / 2)))

    i_idx, j_idx = np.indices((img_height, img_width))
    x_prime = (j_idx - img_width / 2)
    y_prime = (i_idx - img_height / 2)

    x_local = x_prime * depth_image / focal_length_px
    y_local = y_prime * depth_image / focal_length_px
    z_local = depth_image

    local_points = np.stack((x_local, -z_local, -y_local), axis=-1)
    global_points = local_to_global(camera_position, camera_orientation, local_points)

    return global_points[:, :, 2]  # Return height map


def local_to_global(position, rotation_matrix, local_point):
    """
    Transforms a local coordinate point to global coordinates based on position and rotation matrix.

    Args:
        position (np.ndarray): The global position (3D vector).
        rotation_matrix (np.ndarray): The 3x3 rotation matrix.
        local_point (np.ndarray): The point in local coordinates (3D).

    Returns:
        np.ndarray: Transformed global coordinates.
    """
    # Rotate the local points using the rotation matrix
    rotated_point = np.dot(local_point, rotation_matrix)

    # Add the global position to the rotated points
    global_point = rotated_point + position

    return global_point
