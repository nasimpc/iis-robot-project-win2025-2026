import pybullet as p
import pybullet_data
import time

import numpy as np

# PID Constants (These are the "tuning knobs" for your students)
Kp_dist = 52.0  # Force to move forward
Kp_angle = 10.0 # Force to turn
Kd = 50.0        # Damping to prevent oscillation

def pid_to_target(robot_id, target_pos):
    # 1. SENSE: Get robot position and orientation
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    euler = p.getEulerFromQuaternion(orn)
    yaw = euler[2] # Current heading

    # 2. COMPUTE: Calculate errors
    dx = target_pos[0] - pos[0]
    dy = target_pos[1] - pos[1]
    dist_error = np.sqrt(dx**2 + dy**2)
    
    target_yaw = np.arctan2(dy, dx)
    angle_error = target_yaw - yaw
    
    # Normalize angle error to [-pi, pi]
    angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

    # 3. CONTROL LAW: Calculate Torques
    # Forward torque + Turning torque
    linear_effort = Kp_dist * dist_error
    steering_effort = Kp_angle * angle_error
    
    left_torque = linear_effort - steering_effort
    right_torque = linear_effort + steering_effort

    # 4. ACT: Apply raw torque to wheels
    for i in [2, 3]: # Left
        p.setJointMotorControl2(robot_id, i, p.TORQUE_CONTROL, force=2)
    for i in [4, 5]: # Righ
        p.setJointMotorControl2(robot_id, i, p.TORQUE_CONTROL, force=2)

    return dist_error

def setup_simulation():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # 1. Spawn the Room
    p.loadURDF("room.urdf", [0, 0, 0], useFixedBase=True)

    # 2. Spawn the Target Table
    table_id = p.loadURDF("table/table.urdf", basePosition=[2, 2, 0], useFixedBase=True)

    # 3. Spawn Obstacles (Random Blocks)
    # A heavy crate
    p.loadURDF("block.urdf", basePosition=[1, 0, 0.1], globalScaling=5.0) 
    # A simple block obstacle
    p.loadURDF("block.urdf", basePosition=[-1, 2, 0.1], globalScaling=2.0)

    # 4. Spawn the Husky Robot
    robot_id = p.loadURDF("husky/husky.urdf", basePosition=[-3, -3, 0.2])
    
    return robot_id, table_id

def main():
    robot_id, table_id = setup_simulation()
    
    # Set camera to see the whole room
    p.resetDebugVisualizerCamera(cameraDistance=12, cameraYaw=45, cameraPitch=-45, cameraTargetPosition=[0,0,0])

    print("Room initialized. Husky is at (-3, -3). Table is at (2, 2).")

    target = [2, 2, 0] # The table position
    
    # Run this ONCE before your simulation loop
    for i in [2, 3, 4, 5]:
      p.setJointMotorControl2(
        bodyUniqueId=robot_id, 
        jointIndex=i, 
        controlMode=p.VELOCITY_CONTROL, 
        targetVelocity=0, 
        force=0  # This "disables" the internal motor
      )

    while p.isConnected():
          dist = pid_to_target(robot_id, target)
          print ('Distance: ', dist)
          if dist < 0.5:
             print("Target Reached!")
             # Apply braking torque
             for i in range(2, 6):
                 p.setJointMotorControl2(robot_id, i, p.TORQUE_CONTROL, force=0)  
          p.stepSimulation()
          time.sleep(1./240.)

if __name__ == "__main__":
    main()
