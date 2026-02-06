"""
Module 6: Motion Control
Implements PID controllers for both navigation (difference drive) and arm manipulation.

This module provides:
- PID controller for differential drive navigation
- PID controller for arm joint control
- Path following utilities
- Inverse kinematics wrapper for grasping
"""

import numpy as np
import pybullet as p
from typing import List, Tuple, Optional


class PIDController:
    """
    Generic PID controller for controlling a single variable.
    """
    
    def __init__(self, kp, ki, kd, output_limits=None):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_limits: (min, max) tuple for output clamping
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        # Internal state
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None
    
    def compute(self, setpoint, measured_value, dt=1.0/240.0):
        """
        Compute control output.
        
        Args:
            setpoint: Desired value
            measured_value: Current measured value
            dt: Time step
            
        Returns:
            control_output: PID control signal
        """
        # Calculate error
        error = setpoint - measured_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
        if dt > 0:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = 0.0
        d_term = self.kd * derivative
        
        # Compute output
        output = p_term + i_term + d_term
        
        # Apply output limits
        if self.output_limits is not None:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # Update state
        self.previous_error = error
        
        return output
    
    def reset(self):
        """Reset internal state."""
        self.integral = 0.0
        self.previous_error = 0.0


class DifferentialDriveController:
    """
    PID controller for differential drive robot navigation.
    Controls both linear and angular motion to reach a target.
    """
    
    def __init__(self, kp_linear=2.0, kd_linear=0.5, kp_angular=4.0, kd_angular=1.0,
                 max_linear_velocity=1.5, max_angular_velocity=2.0):
        """
        Initialize differential drive controller.
        
        Args:
            kp_linear: Proportional gain for distance control
            kd_linear: Derivative gain for distance control
            kp_angular: Proportional gain for angular control
            kd_angular: Derivative gain for angular control
            max_linear_velocity: Maximum linear velocity (m/s)
            max_angular_velocity: Maximum angular velocity (rad/s)
        """
        self.linear_pid = PIDController(kp_linear, 0.0, kd_linear, 
                                       output_limits=(-max_linear_velocity, max_linear_velocity))
        self.angular_pid = PIDController(kp_angular, 0.0, kd_angular,
                                        output_limits=(-max_angular_velocity, max_angular_velocity))
        
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
    
    def compute_control(self, current_pose, target_pose, dt=1.0/240.0):
        """
        Compute control velocities to reach target.
        
        Args:
            current_pose: (x, y, theta) current robot pose
            target_pose: (x, y, theta) target pose
            dt: Time step
            
        Returns:
            (v, omega): Linear and angular velocities
            distance_error: Current distance to target
        """
        x, y, theta = current_pose
        target_x, target_y, target_theta = target_pose
        
        # Calculate distance and angle to target
        dx = target_x - x
        dy = target_y - y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Angle to target
        target_angle = np.arctan2(dy, dx)
        
        # Angular error (normalize to [-pi, pi])
        angle_error = target_angle - theta
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
        
        # Compute angular velocity
        omega = self.angular_pid.compute(0, -angle_error, dt)
        
        # Compute linear velocity (reduce when turning)
        # Only move forward if roughly facing the target
        if abs(angle_error) < np.pi / 3:  # Within 60 degrees
            v = self.linear_pid.compute(0, -distance, dt)
            # Reduce speed when turning
            v = v * (1.0 - abs(angle_error) / np.pi)
        else:
            # Just turn in place if facing wrong direction
            v = 0.0
        
        return (v, omega), distance
    
    def reset(self):
        """Reset PID controllers."""
        self.linear_pid.reset()
        self.angular_pid.reset()


class WheelController:
    """
    Low-level controller to convert (v, omega) to wheel velocities and apply them to PyBullet.
    """
    
    def __init__(self, wheel_radius=0.1651, wheel_base=0.555):
        """
        Initialize wheel controller.
        
        Args:
            wheel_radius: Radius of wheels (m)
            wheel_base: Distance between left and right wheels (m)
        """
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
    
    def apply_velocity_command(self, robot_id, v, omega, wheel_indices=[2, 3, 4, 5], max_force=1500):
        """
        Apply velocity command to robot wheels.
        
        Args:
            robot_id: PyBullet body ID of robot
            v: Linear velocity (m/s)
            omega: Angular velocity (rad/s)
            wheel_indices: Indices of wheel joints [left_front, right_front, left_rear, right_rear]
            max_force: Maximum force to apply to motors
        """
        # Convert to wheel velocities using differential drive kinematics
        v_left = (v - omega * self.wheel_base / 2.0) / self.wheel_radius
        v_right = (v + omega * self.wheel_base / 2.0) / self.wheel_radius
        
        # Apply to left wheels
        for i in wheel_indices[:2]:  # Left wheels
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=i,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=v_left,
                force=max_force
            )
        
        # Apply to right wheels
        for i in wheel_indices[2:]:  # Right wheels
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=i,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=v_right,
                force=max_force
            )
    
    def stop(self, robot_id, wheel_indices=[2, 3, 4, 5], max_force=1500):
        """
        Stop the robot by setting all wheel velocities to zero.
        
        Args:
            robot_id: PyBullet body ID of robot
            wheel_indices: Indices of wheel joints
            max_force: Maximum braking force
        """
        for i in wheel_indices:
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=i,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0.0,
                force=max_force
            )


class ArmController:
    """
    Controller for robot arm manipulation using PyBullet's internal IK and position control.
    """
    
    def __init__(self, arm_id, end_effector_index=6, num_joints=7):
        """
        Initialize arm controller.
        
        Args:
            arm_id: PyBullet body ID of robot arm
            end_effector_index: Index of end effector link
            num_joints: Number of controllable joints
        """
        self.arm_id = arm_id
        self.end_effector_index = end_effector_index
        self.num_joints = num_joints
    
    def move_to_position(self, target_position, target_orientation=None, max_force=200):
        """
        Move end effector to target position using inverse kinematics.
        
        Args:
            target_position: [x, y, z] target position in world frame
            target_orientation: Optional [x, y, z, w] quaternion orientation
            max_force: Maximum force for joint control
            
        Returns:
            joint_poses: Computed joint angles from IK
        """
        # Compute inverse kinematics
        if target_orientation is not None:
            joint_poses = p.calculateInverseKinematics(
                bodyUniqueId=self.arm_id,
                endEffectorLinkIndex=self.end_effector_index,
                targetPosition=target_position,
                targetOrientation=target_orientation
            )
        else:
            joint_poses = p.calculateInverseKinematics(
                bodyUniqueId=self.arm_id,
                endEffectorLinkIndex=self.end_effector_index,
                targetPosition=target_position
            )
        
        # Apply joint positions
        for i in range(min(self.num_joints, len(joint_poses))):
            p.setJointMotorControl2(
                bodyIndex=self.arm_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=max_force
            )
        
        return joint_poses[:self.num_joints]
    
    def get_end_effector_pose(self):
        """
        Get current end effector position and orientation.
        
        Returns:
            position: [x, y, z]
            orientation: [x, y, z, w] quaternion
        """
        state = p.getLinkState(self.arm_id, self.end_effector_index)
        position = state[0]
        orientation = state[1]
        return position, orientation
    
    def grasp(self, target_position, approach_offset=0.1, grasp_offset=0.02):
        """
        Execute a grasp sequence: approach, descend, grasp.
        
        Args:
            target_position: [x, y, z] position of object to grasp
            approach_offset: Offset above object for approach
            grasp_offset: Final offset for grasping
            
        Returns:
            success: True if grasp sequence completed
        """
        # Phase 1: Move above target
        approach_position = [target_position[0], target_position[1], target_position[2] + approach_offset]
        self.move_to_position(approach_position)
        
        # Phase 2: Move down to grasp position
        grasp_position = [target_position[0], target_position[1], target_position[2] + grasp_offset]
        self.move_to_position(grasp_position)
        
        # Phase 3: Close gripper (if available)
        # This would require gripper joint control - implement based on your URDF
        
        return True


class PathFollower:
    """
    Follow a path using waypoint-based control.
    """
    
    def __init__(self, waypoints, waypoint_tolerance=0.3):
        """
        Initialize path follower.
        
        Args:
            waypoints: List of (x, y, theta) waypoints
            waypoint_tolerance: Distance threshold to consider waypoint reached
        """
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.waypoint_tolerance = waypoint_tolerance
    
    def get_current_target(self):
        """
        Get current target waypoint.
        
        Returns:
            waypoint: (x, y, theta) or None if path complete
        """
        if self.current_waypoint_index < len(self.waypoints):
            return self.waypoints[self.current_waypoint_index]
        return None
    
    def update(self, current_position):
        """
        Update waypoint based on current position.
        
        Args:
            current_position: (x, y) current position
            
        Returns:
            is_complete: True if all waypoints reached
        """
        if self.current_waypoint_index >= len(self.waypoints):
            return True
        
        current_target = self.waypoints[self.current_waypoint_index]
        distance = np.sqrt((current_position[0] - current_target[0])**2 + 
                          (current_position[1] - current_target[1])**2)
        
        if distance < self.waypoint_tolerance:
            self.current_waypoint_index += 1
            return self.current_waypoint_index >= len(self.waypoints)
        
        return False
    
    def reset(self):
        """Reset to first waypoint."""
        self.current_waypoint_index = 0


# Testing functions
if __name__ == "__main__":
    print("Motion Control module loaded successfully.")
    print("Controllers available:")
    print("  - PIDController")
    print("  - DifferentialDriveController")
    print("  - WheelController")
    print("  - ArmController")
    print("  - PathFollower")
    
    # Simple PID test
    pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
    print("\nPID Test:")
    for i in range(5):
        output = pid.compute(setpoint=10.0, measured_value=i*2.0)
        print(f"Step {i}: output = {output:.2f}")

