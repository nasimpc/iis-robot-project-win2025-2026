"""
Module 4: Perception
Implements object detection, table plane detection using RANSAC, and PCA for pose estimation.

This module processes RGB-D sensor data to:
- Detect objects and obstacles based on predefined attributes (color, size)
- Identify the table plane using RANSAC
- Use PCA to find optimal pose for avoidance and grasping
"""

import numpy as np
import cv2
from sklearn.decomposition import PCA
from typing import Tuple, List, Dict, Optional
import pybullet as p


def rgb_to_point_cloud(rgb, depth, mask, robot_id=None, width=320, height=240, fov=60):
    """
    Convert RGB-D image to 3D point cloud in world coordinates.
    
    Args:
        rgb: RGB image (H x W x 4)
        depth: Depth buffer (H x W)
        mask: Segmentation mask (H x W)
        robot_id: PyBullet robot ID for coordinate transformation
        width: Image width
        height: Image height
        fov: Field of view in degrees
        
    Returns:
        points: Nx3 array of 3D points in WORLD coordinates
        colors: Nx3 array of RGB colors
        masks: Nx1 array of object IDs
    """
    # Convert depth from normalized [0, 1] to actual distance
    # PyBullet depth buffer: depth = far * near / (far - (far - near) * depth_buffer)
    far = 10.0
    near = 0.1
    
    # Reshape depth to 2D if needed
    depth_2d = np.array(depth).reshape(height, width)
    depth_meters = far * near / (far - (far - near) * depth_2d)
    
    # Camera intrinsic parameters
    f = height / (2 * np.tan(np.deg2rad(fov) / 2))
    cx = width / 2.0
    cy = height / 2.0
    
    # Create pixel grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert to 3D points (camera coordinates)
    # Camera frame: x=right, y=down, z=forward
    z_cam = depth_meters
    x_cam = (u - cx) * z_cam / f
    y_cam = -(v - cy) * z_cam / f  # Negative because image y-axis points down
    
    # Stack into point cloud (camera frame)
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1).reshape(-1, 3)
    
    # Transform from camera frame to world frame
    if robot_id is not None:
        # Get robot pose
        robot_pos, robot_orn = p.getBasePositionAndOrientation(robot_id)
        
        # Get rotation matrix from quaternion
        rot_matrix = np.array(p.getMatrixFromQuaternion(robot_orn)).reshape(3, 3)
        
        # Transform each point: p_world = R * p_cam + t
        # Camera is looking forward (+X in world when robot yaw=0)
        # Need to rotate camera frame to match world frame
        # Camera: x=right, y=up (after flip), z=forward
        # World: x=forward, y=left, z=up (standard robotics)
        # Correction matrix: swap and flip axes
        cam_to_robot = np.array([
            [0, 0, 1],   # camera z (forward) -> robot x
            [-1, 0, 0],  # camera -x (left) -> robot y  
            [0, 1, 0]    # camera y (up) -> robot z (FIXED: was -1)
        ])
        
        # Transform points
        points_robot = points_cam @ cam_to_robot.T
        points_world = points_robot @ rot_matrix.T + np.array(robot_pos)
        
        points = points_world
    else:
        # No transformation, return camera coordinates
        points = points_cam
    
    # Extract colors
    rgb_array = np.array(rgb).reshape(height, width, 4)
    colors = rgb_array[:, :, :3].reshape(-1, 3) / 255.0  # Normalize to [0, 1]
    
    # Extract masks
    mask_array = np.array(mask).reshape(-1)
    
    # Filter out invalid points (too far or too close)
    valid_mask = (depth_meters.reshape(-1) > near) & (depth_meters.reshape(-1) < far)
    
    return points[valid_mask], colors[valid_mask], mask_array[valid_mask]
    colors = rgb_array[:, :, :3].reshape(-1, 3) / 255.0  # Normalize to [0, 1]
    
    # Extract masks
    mask_array = np.array(mask).reshape(-1)
    
    # Filter out invalid points (too far or too close)
    valid_mask = (depth_meters.reshape(-1) > near) & (depth_meters.reshape(-1) < far)
    
    return points[valid_mask], colors[valid_mask], mask_array[valid_mask]


def detect_objects_by_color(points, colors, target_color, color_tolerance=0.3):
    """
    Detect objects in point cloud based on color matching.
    
    Args:
        points: Nx3 array of 3D points
        colors: Nx3 array of RGB colors (normalized [0, 1])
        target_color: [R, G, B] target color (normalized [0, 1])
        color_tolerance: Maximum Euclidean distance in RGB space for matching
        
    Returns:
        detected_points: Mx3 array of points matching the target color
        centroid: [x, y, z] centroid of detected points (or None if not found)
    """
    target_color = np.array(target_color)
    
    # Calculate color distance
    color_diff = np.linalg.norm(colors - target_color, axis=1)
    
    # Filter points by color
    color_mask = color_diff < color_tolerance
    detected_points = points[color_mask]
    
    if len(detected_points) == 0:
        return None, None
    
    # Calculate centroid
    centroid = np.mean(detected_points, axis=0)
    
    return detected_points, centroid


def ransac_plane_detection(points, distance_threshold=0.05, num_iterations=100, min_inliers=100):
    """
    Detect a plane in 3D point cloud using RANSAC algorithm.
    Useful for detecting the table surface.
    
    Args:
        points: Nx3 array of 3D points
        distance_threshold: Maximum distance for a point to be considered an inlier
        num_iterations: Number of RANSAC iterations
        min_inliers: Minimum number of inliers required
        
    Returns:
        plane_model: [a, b, c, d] plane equation coefficients (ax + by + cz + d = 0)
        inliers: Boolean mask of inlier points
        inlier_points: Mx3 array of inlier points
    """
    if len(points) < 3:
        return None, None, None
    
    best_plane = None
    best_inliers = None
    max_inlier_count = 0
    
    for _ in range(num_iterations):
        # Randomly sample 3 points
        sample_indices = np.random.choice(len(points), 3, replace=False)
        sample_points = points[sample_indices]
        
        # Calculate plane equation from 3 points
        p1, p2, p3 = sample_points
        
        # Two vectors in the plane
        v1 = p2 - p1
        v2 = p3 - p1
        
        # Normal vector (cross product)
        normal = np.cross(v1, v2)
        
        # Normalize
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        
        normal = normal / norm
        
        # Plane equation: ax + by + cz + d = 0
        a, b, c = normal
        d = -np.dot(normal, p1)
        
        # Calculate distance of all points to the plane
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
        
        # Find inliers
        inliers = distances < distance_threshold
        inlier_count = np.sum(inliers)
        
        # Update best plane if this one has more inliers
        if inlier_count > max_inlier_count:
            max_inlier_count = inlier_count
            best_plane = [a, b, c, d]
            best_inliers = inliers
    
    if best_plane is None or max_inlier_count < min_inliers:
        return None, None, None
    
    inlier_points = points[best_inliers]
    
    return best_plane, best_inliers, inlier_points


def estimate_pose_pca(points):
    """
    Estimate the orientation of an object using Principal Component Analysis (PCA).
    This finds the main axes of the point cloud.
    
    Args:
        points: Nx3 array of 3D points
        
    Returns:
        centroid: [x, y, z] center of the object
        principal_axes: 3x3 matrix where each row is a principal axis
                       (sorted by variance: largest to smallest)
        dimensions: [length, width, height] approximate object dimensions along principal axes
    """
    if len(points) < 3:
        return None, None, None
    
    # Calculate centroid
    centroid = np.mean(points, axis=0)
    
    # Center the points
    centered_points = points - centroid
    
    # Perform PCA
    pca = PCA(n_components=3)
    pca.fit(centered_points)
    
    # Principal axes (eigenvectors)
    principal_axes = pca.components_
    
    # Project points onto principal axes to find dimensions
    projected = centered_points @ principal_axes.T
    dimensions = np.max(projected, axis=0) - np.min(projected, axis=0)
    
    return centroid, principal_axes, dimensions


def detect_table(points, colors, expected_height=0.625, height_tolerance=0.15, brown_color=[0.5, 0.3, 0.1]):
    """
    Detect the table using both color filtering and RANSAC plane detection.
    
    Args:
        points: Nx3 array of 3D points
        colors: Nx3 array of RGB colors
        expected_height: Expected table height in meters
        height_tolerance: Tolerance for height filtering
        brown_color: RGB color of the table
        
    Returns:
        table_plane: [a, b, c, d] plane equation
        table_center: [x, y, z] centroid of the table
        table_points: Mx3 array of table surface points
    """
    # Filter points by approximate height (z-coordinate)
    height_mask = np.abs(points[:, 2] - expected_height) < height_tolerance
    candidate_points = points[height_mask]
    candidate_colors = colors[height_mask]
    
    if len(candidate_points) == 0:
        return None, None, None
    
    # Apply RANSAC to find the dominant plane at this height
    table_plane, inliers, table_points = ransac_plane_detection(
        candidate_points,
        distance_threshold=0.03,
        num_iterations=200,
        min_inliers=50
    )
    
    if table_plane is None:
        return None, None, None
    
    # Calculate table center
    table_center = np.mean(table_points, axis=0)
    
    return table_plane, table_center, table_points


def detect_target_object(points, colors, target_color=[1.0, 0.0, 0.0], color_tolerance=0.5):
    """
    Detect the red target object (cylinder) on the table.
    
    Args:
        points: Nx3 array of 3D points
        colors: Nx3 array of RGB colors
        target_color: RGB color of target (red by default)
        color_tolerance: Color matching tolerance (increased for better detection)
        
    Returns:
        target_center: [x, y, z] centroid
        target_points: Mx3 array of target object points
        target_pose: Principal axes and dimensions from PCA
    """
    # More sophisticated red detection - look for high red channel with low green/blue
    red_channel = colors[:, 0]
    green_channel = colors[:, 1]
    blue_channel = colors[:, 2]
    
    # Red objects have: high R, low G, low B
    red_score = red_channel - 0.5 * (green_channel + blue_channel)
    red_mask = (red_channel > 0.4) & (red_score > 0.3)
    
    # Also use traditional color distance as backup
    target_color_arr = np.array(target_color)
    color_diff = np.linalg.norm(colors - target_color_arr, axis=1)
    color_mask = color_diff < color_tolerance
    
    # Combine both methods
    combined_mask = red_mask | color_mask
    red_ish_points = np.sum(combined_mask)
    
    # Debug: Print color statistics occasionally
    if np.random.rand() < 0.02:  # 2% chance to print (more frequent feedback)
        print(f"[TargetDetect] Total points: {len(points)}, Red-ish points: {red_ish_points}, Tolerance: {color_tolerance}")
    
    # Get target points using combined mask
    target_points = points[combined_mask]
    
    if len(target_points) < 5:  # Need at least 5 points (lowered for small target)
        return None, None, None
    
    # Calculate centroid
    target_center = np.mean(target_points, axis=0)
    
    # Estimate pose using PCA
    if len(target_points) >= 3:  # PCA needs at least 3 points
        centroid, principal_axes, dimensions = estimate_pose_pca(target_points)
    else:
        centroid = target_center
        principal_axes = np.eye(3)
        dimensions = np.array([0.08, 0.08, 0.12])  # Approximate cylinder dimensions
    
    target_pose = {
        'centroid': centroid,
        'principal_axes': principal_axes,
        'dimensions': dimensions
    }
    
    return target_center, target_points, target_pose


def detect_obstacles(points, colors, obstacle_colors=None, color_tolerance=0.3):
    """
    Detect multiple obstacles based on their colors.
    
    Args:
        points: Nx3 array of 3D points
        colors: Nx3 array of RGB colors
        obstacle_colors: Dict mapping color names to RGB values
        color_tolerance: Color matching tolerance
        
    Returns:
        obstacles: List of dicts containing obstacle information
    """
    if obstacle_colors is None:
        obstacle_colors = {
            'Blue': [0.0, 0.0, 1.0],
            'Pink': [1.0, 0.4, 0.7],
            'Orange': [1.0, 0.5, 0.0],
            'Yellow': [1.0, 1.0, 0.0],
            'Black': [0.1, 0.1, 0.1]
        }
    
    obstacles = []
    
    for color_name, color_rgb in obstacle_colors.items():
        obstacle_points, obstacle_center = detect_objects_by_color(
            points, colors, color_rgb, color_tolerance
        )
        
        if obstacle_points is not None:
            # Estimate pose using PCA
            centroid, principal_axes, dimensions = estimate_pose_pca(obstacle_points)
            
            obstacles.append({
                'color': color_name,
                'center': obstacle_center,
                'points': obstacle_points,
                'centroid': centroid,
                'principal_axes': principal_axes,
                'dimensions': dimensions
            })
    
    return obstacles


def process_sensor_data(sensor_data, robot_id=None):
    """
    Main perception pipeline: processes raw sensor data and extracts meaningful information.
    
    Args:
        sensor_data: Dict containing 'rgb', 'depth', 'mask' from sensor_wrapper
        robot_id: PyBullet robot ID for coordinate transformation
        
    Returns:
        perception_output: Dict containing:
            - target: Target object information (or None)
            - table: Table information (or None)
            - obstacles: List of obstacle information
            - point_cloud: Full point cloud for debugging
    """
    rgb = sensor_data['rgb']
    depth = sensor_data['depth']
    mask = sensor_data['mask']
    
    # Convert to point cloud in world coordinates
    points, colors, masks = rgb_to_point_cloud(rgb, depth, mask, robot_id=robot_id)
    
    # Detect table
    table_plane, table_center, table_points = detect_table(points, colors)
    
    # Detect target object
    target_center, target_points, target_pose = detect_target_object(points, colors)
    
    # Detect obstacles
    obstacles = detect_obstacles(points, colors)
    
    perception_output = {
        'target': {
            'center': target_center,
            'points': target_points,
            'pose': target_pose
        } if target_center is not None else None,
        'table': {
            'plane': table_plane,
            'center': table_center,
            'points': table_points
        } if table_center is not None else None,
        'obstacles': obstacles,
        'point_cloud': {
            'points': points,
            'colors': colors,
            'masks': masks
        }
    }
    
    return perception_output


# Testing function
if __name__ == "__main__":
    print("Perception module loaded successfully.")
    print("Functions available:")
    print("  - rgb_to_point_cloud()")
    print("  - detect_objects_by_color()")
    print("  - ransac_plane_detection()")
    print("  - estimate_pose_pca()")
    print("  - detect_table()")
    print("  - detect_target_object()")
    print("  - detect_obstacles()")
    print("  - process_sensor_data()")

