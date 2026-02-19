"""
Module 6: Motion Control
- PID controller for differential drive base
- Path planning using Prolog (via PySwip)
- Grasp planning using PyBullet Inverse Kinematics
"""

import numpy as np
import pybullet as p
from pyswip import Prolog
import time


class PIDController:
    """Simple P-controller for base navigation (can be extended with I and D)."""
    def __init__(self, kp_lin=0.5, kp_ang=1.0, max_lin=0.5, max_ang=1.0):
        self.kp_lin = kp_lin
        self.kp_ang = kp_ang
        self.max_lin = max_lin
        self.max_ang = max_ang

    def compute(self, current_pose, target_pose):
        """
        Compute linear and angular velocity to reach a target (x, y).
        current_pose: (x, y, theta) from particle filter.
        target_pose: (x, y) waypoint.
        Returns (v, omega).
        """
        dx = target_pose[0] - current_pose[0]
        dy = target_pose[1] - current_pose[1]
        distance = np.hypot(dx, dy)

        # Desired heading to target
        desired_theta = np.arctan2(dy, dx)
        angle_error = desired_theta - current_pose[2]
        # Normalise to [-pi, pi]
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

        # P controller
        v = self.kp_lin * distance
        omega = self.kp_ang * angle_error

        # Apply limits
        v = np.clip(v, -self.max_lin, self.max_lin)
        omega = np.clip(omega, -self.max_ang, self.max_ang)

        return v, omega


class PathPlanner:
    """
    Simple path planner using Prolog for obstacle knowledge.
    For now, just returns a straight line if free; otherwise a simple detour.
    In a full implementation, you would implement A* or RRT in Prolog.
    """
    def __init__(self, prolog_file="map.pl"):
        self.prolog = Prolog()
        self.prolog.consult(prolog_file)

    def plan_path(self, start, goal, obstacles):
        """
        Args:
            start: (x, y) robot start.
            goal: (x, y) target (table position).
            obstacles: list of (x, y, size) for each obstacle.
        Returns:
            list of waypoints [(x1,y1), (x2,y2), ...] or empty list if no path.
        """
        # Very simple: if direct line is collision-free, return [goal]
        if self._is_line_free(start, goal, obstacles):
            return [goal]
        else:
            # Fallback: go to an intermediate point beside the obstacle
            # (You would implement a proper planner here)
            # For demonstration, just return an empty list (meaning no plan)
            print("PathPlanner: Direct path blocked – implement proper planner.")
            return []

    def _is_line_free(self, a, b, obstacles, clearance=0.5):
        """Check if line from a to b is free of obstacles."""
        for obs in obstacles:
            # Simple check: distance from line segment to obstacle center
            # This is approximate; you might want a more accurate method.
            ox, oy, size = obs
            # Closest point on line segment to obstacle center
            # (omitted for brevity; use point-line distance)
            # For now, assume all lines are free – you should implement.
            pass
        return True  # placeholder


class GraspPlanner:
    """Plan arm motion to grasp the target using IK."""
    def __init__(self, robot_id, arm_joint_indices, end_effector_link_index):
        self.robot_id = robot_id
        self.arm_joints = arm_joint_indices
        self.ee_link = end_effector_link_index

    def compute_ik(self, target_position, target_orientation=None):
        """
        Compute joint angles to place end effector at target pose.
        target_position: [x, y, z] in world coordinates.
        target_orientation: quaternion [x,y,z,w] or None (use default).
        Returns list of joint angles or None if no solution.
        """
        if target_orientation is None:
            # Default orientation: gripper pointing down (suitable for vertical cylinder)
            target_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])  # adjust as needed

        joint_angles = p.calculateInverseKinematics(
            self.robot_id, self.ee_link, target_position, target_orientation,
            maxNumIterations=100, residualThreshold=1e-4
        )
        # IK returns values for all joints; we need only our arm joints
        # In PyBullet, the order matches the joint indices, but we must ensure we pick the right ones.
        # For simplicity, assume joint_angles is a tuple in the same order as the arm joints.
        # If the robot has more joints (e.g., wheels), we need to map.
        # Here we assume arm_joints are consecutive and IK returns them in order.
        return joint_angles[:len(self.arm_joints)]

    def execute_grasp(self, target_pos, gripper_joints):
        """
        Move arm to pre-grasp, close gripper, lift.
        target_pos: (x,y,z) of target in world.
        gripper_joints: indices of left and right finger joints.
        """
        # Step 1: pre-grasp pose (above the object)
        pre_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.1]
        angles = self.compute_ik(pre_pos)
        if angles is None:
            return False
        # Set arm joint positions (position control)
        for i, joint_idx in enumerate(self.arm_joints):
            p.setJointMotorControl2(self.robot_id, joint_idx,
                                    p.POSITION_CONTROL,
                                    targetPosition=angles[i],
                                    force=100)  # use suitable force
        # Wait for motion to complete (simulate a few steps)
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./240.)

        # Step 2: grasp pose (at the object)
        angles = self.compute_ik(target_pos)
        if angles is None:
            return False
        for i, joint_idx in enumerate(self.arm_joints):
            p.setJointMotorControl2(self.robot_id, joint_idx,
                                    p.POSITION_CONTROL,
                                    targetPosition=angles[i],
                                    force=100)
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./240.)

        # Step 3: close gripper
        p.setJointMotorControl2(self.robot_id, gripper_joints[0],
                                p.POSITION_CONTROL, targetPosition=0.0, force=20)
        p.setJointMotorControl2(self.robot_id, gripper_joints[1],
                                p.POSITION_CONTROL, targetPosition=0.0, force=20)
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)

        # Step 4: lift slightly
        lift_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.15]
        angles = self.compute_ik(lift_pos)
        if angles is None:
            return False
        for i, joint_idx in enumerate(self.arm_joints):
            p.setJointMotorControl2(self.robot_id, joint_idx,
                                    p.POSITION_CONTROL,
                                    targetPosition=angles[i],
                                    force=100)
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./240.)

        return True
