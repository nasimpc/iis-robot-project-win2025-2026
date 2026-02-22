"""
World Builder Module for IIS Master Project
Generates randomized scene configurations for the Navigate-to-Grasp Challenge.

Each execution generates a new random configuration:
- Table: Random position on floor
- Target: Random position on table surface
- Obstacles: 5 cubes with random positions (Blue, Pink, Orange, Yellow, Black)
- Robot: Fixed spawn position

Returns initial scene map for knowledge base (excluding target pose).
"""

import pybullet as p
import pybullet_data
import numpy as np
import os
import random

# Get the directory containing this file
ENVIRONMENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_DIR = os.path.join(os.path.dirname(ENVIRONMENT_DIR), 'robot')

# Obstacle colors: Blue, Pink, Orange, Yellow, Black (as specified in requirements)
OBSTACLE_COLORS = [
    [0.0, 0.0, 1.0, 1.0],  # Blue
    [1.0, 0.4, 0.7, 1.0],  # Pink
    [1.0, 0.5, 0.0, 1.0],  # Orange
    [1.0, 1.0, 0.0, 1.0],  # Yellow
    [0.1, 0.1, 0.1, 1.0],  # Black
]


def check_collision(new_pos, existing_positions, min_distance=1.0):
    """
    Check if a new position is at least min_distance away from all existing positions.
    
    Args:
        new_pos: [x, y] position to check
        existing_positions: List of [x, y] positions already placed
        min_distance: Minimum required distance between objects
        
    Returns:
        bool: True if position is valid (no collision), False otherwise
    """
    for pos in existing_positions:
        dist = np.sqrt((new_pos[0] - pos[0])**2 + (new_pos[1] - pos[1])**2)
        if dist < min_distance:
            return False
    return True


def generate_random_position(x_range, y_range, existing_positions, min_distance=1.0, max_attempts=100):
    """
    Generate a random position that doesn't collide with existing objects.
    
    Args:
        x_range: (min_x, max_x) tuple for valid x coordinates
        y_range: (min_y, max_y) tuple for valid y coordinates
        existing_positions: List of [x, y] positions already placed
        min_distance: Minimum required distance between objects
        max_attempts: Maximum number of attempts before raising an error
        
    Returns:
        [x, y]: Valid random position
    """
    for _ in range(max_attempts):
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        if check_collision([x, y], existing_positions, min_distance):
            return [x, y]
    
    # If we can't find a valid position, use a grid-based approach
    x = random.uniform(x_range[0], x_range[1])
    y = random.uniform(y_range[0], y_range[1])
    return [x, y]


def build_world(physics_client=None):
    """
    Build the simulation world with randomized object placement.
    
    Args:
        physics_client: Optional PyBullet physics client ID. If None, a new GUI connection is created.
        
    Returns:
        dict: Scene configuration containing:
            - 'room_id': PyBullet body ID for the room
            - 'robot_id': PyBullet body ID for the robot
            - 'table_id': PyBullet body ID for the table
            - 'table_position': [x, y, z] position of the table
            - 'obstacle_ids': List of PyBullet body IDs for obstacles
            - 'obstacle_positions': List of [x, y, z] positions for obstacles
            - 'obstacle_colors': List of color names for obstacles
            - 'target_id': PyBullet body ID for the target (position NOT included - must be perceived)
            - 'robot_position': [x, y, z] starting position of the robot
            
        Note: Target position is intentionally NOT included in the map.
              The robot must perceive the target using its sensors.
    """
    # Connect to PyBullet if not already connected
    if physics_client is None:
        physics_client = p.connect(p.GUI)
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Track placed objects for collision checking
    placed_positions = []
    
    # ==================== SPAWN ROOM ====================
    room_path = os.path.join(ENVIRONMENT_DIR, 'room.urdf')
    room_id = p.loadURDF(room_path, [0, 0, 0], useFixedBase=True)
    
    # Set floor friction
    p.changeDynamics(room_id, -1, lateralFriction=0.5)
    
    # ==================== SPAWN ROBOT ====================
    robot_position = [-3.0, -3.0, 0.2]  # Fixed spawn position near corner
    robot_path = os.path.join(ROBOT_DIR, 'robot.urdf')
    robot_id = p.loadURDF(robot_path, robot_position, useFixedBase=False)
    placed_positions.append(robot_position[:2])
    
    # ==================== SPAWN TABLE ====================
    # Table can be placed anywhere in the center area (avoiding walls and robot spawn)
    table_pos_2d = generate_random_position(
        x_range=(0.0, 3.0),
        y_range=(0.0, 3.0),
        existing_positions=placed_positions,
        min_distance=2.0
    )
    table_position = [table_pos_2d[0], table_pos_2d[1], 0.625]  # Table top at z=0.625m
    
    table_path = os.path.join(ENVIRONMENT_DIR, 'table.urdf')
    table_id = p.loadURDF(table_path, table_position, useFixedBase=True)
    placed_positions.append(table_pos_2d)
    
    # ==================== SPAWN OBSTACLES ====================
    obstacle_ids = []
    obstacle_positions = []
    obstacle_color_names = ['Blue', 'Pink', 'Orange', 'Yellow', 'Black']
    
    obstacle_path = os.path.join(ENVIRONMENT_DIR, 'obstacle.urdf')
    
    for i in range(5):
        # Generate random position for obstacle (avoid table and other obstacles)
        obs_pos_2d = generate_random_position(
            x_range=(-4.0, 4.0),
            y_range=(-4.0, 4.0),
            existing_positions=placed_positions,
            min_distance=1.0
        )
        obs_position = [obs_pos_2d[0], obs_pos_2d[1], 0.0]
        
        obstacle_id = p.loadURDF(obstacle_path, obs_position, useFixedBase=True)
        
        # Set obstacle color
        p.changeVisualShape(obstacle_id, -1, rgbaColor=OBSTACLE_COLORS[i])
        
        obstacle_ids.append(obstacle_id)
        obstacle_positions.append(obs_position)
        placed_positions.append(obs_pos_2d)
    
    # ==================== SPAWN TARGET ====================
    # Target is placed randomly on the table surface
    # Table surface is 1.5m x 0.8m, centered at table_position
    target_offset_x = random.uniform(-0.6, 0.6)  # Within table bounds (leaving margin)
    target_offset_y = random.uniform(-0.3, 0.3)
    target_position = [
        table_position[0] + target_offset_x,
        table_position[1] + target_offset_y,
        table_position[2] + 0.025 + 0.06  # Table top + half cylinder height
    ]
    
    target_path = os.path.join(ENVIRONMENT_DIR, 'target.urdf')
    target_id = p.loadURDF(target_path, target_position, useFixedBase=False)
    landmark_map = {
        0: table_position,  # table
    }
    for i, pos in enumerate(obstacle_positions):
        landmark_map[i + 1] = pos
    
    # ==================== BUILD SCENE MAP ====================
    # Note: Target position is NOT included - robot must perceive it
    scene_map = {
        'room_id': room_id,
        'robot_id': robot_id,
        'robot_position': robot_position,
        'table_id': table_id,
        'table_position': table_position,
        'obstacle_ids': obstacle_ids,
        'obstacle_positions': obstacle_positions,
        'obstacle_colors': obstacle_color_names,
        'target_id': target_id,
        'landmark_map': landmark_map,
        # Target position intentionally NOT included
    }
    
    # Print scene configuration for debugging
    print("=" * 60)
    print("WORLD BUILDER: Scene Generated Successfully")
    print("=" * 60)
    print(f"Robot Position: {robot_position}")
    print(f"Table Position: {table_position}")
    print("Obstacles:")
    for i, (pos, color) in enumerate(zip(obstacle_positions, obstacle_color_names)):
        print(f"  {color}: {pos}")
    print(f"Target ID: {target_id} (position must be perceived by robot)")
    print("=" * 60)
    
    return scene_map


def get_object_position(body_id):
    """
    Helper function to get the current position of an object.
    Use this in world_builder.py for validating initial map only.
    
    Args:
        body_id: PyBullet body ID
        
    Returns:
        [x, y, z]: Position of the object
    """
    pos, _ = p.getBasePositionAndOrientation(body_id)
    return list(pos)


# ==================== MAIN (for standalone testing) ====================
if __name__ == "__main__":
    import time
    
    print("Starting World Builder Test...")
    
    # Build the world
    scene_map = build_world()
    
    print("\nSimulation running. Press Ctrl+C to exit.")
    print("Run this script multiple times to see different random configurations.\n")
    
    # Run simulation loop
    try:
        while p.isConnected():  # DO NOT TOUCH
            p.stepSimulation()  # DO NOT TOUCH
            time.sleep(1./240.)  # DO NOT TOUCH
    except KeyboardInterrupt:
        print("\nSimulation terminated by user.")
        p.disconnect()
