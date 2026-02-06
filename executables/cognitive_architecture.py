"""
Module 10: Cognitive Architecture
Integrates all 10 modules into a unified Sense-Think-Act loop.

This is the main execution file that brings together:
- M1-M2: Task specification and hardware (URDF)
- M3: Sensor acquisition
- M4: Perception (RANSAC, PCA)
- M5: State estimation (Particle Filter)
- M6: Motion control (PID)
- M7: Action planning (FSM)
- M8: Knowledge reasoning (Prolog)
- M9: Learning (parameter optimization)
- M10: Cognitive architecture integration
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import sensor wrappers
from src.robot.sensor_wrapper import *

# Import all modules
from src.environment.world_builder import build_world
from src.modules.perception import process_sensor_data
from src.modules.state_estimation import ParticleFilter, fuse_odometry_imu, create_measurement_from_sensors
from src.modules.motion_control import DifferentialDriveController, WheelController, ArmController, PathFollower
from src.modules.action_planning import FiniteStateMachine, RobotState
from src.modules.knowledge_reasoning import KnowledgeBase, create_world_knowledge_base
from src.modules.learning import ParameterOptimizer, PerformanceMetrics

####################### GLOBAL CONFIGURATION #################################

# Simulation parameters
DT = 1.0 / 240.0  # Time step
SAVE_IMAGES = False  # Set to True to save camera images

# Learning configuration
ENABLE_LEARNING = True
LOAD_LEARNED_PARAMS = True


######################## UTILITY FUNCTIONS ############################

def save_camera_data(rgb, depth, filename_prefix="frame"):
    """Save RGB-D camera data to disk."""
    try:
        rgb_array = np.reshape(rgb, (240, 320, 4)).astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_normalized.astype(np.uint8)
        cv2.imwrite(f"{filename_prefix}_rgb.png", rgb_bgr)
        cv2.imwrite(f"{filename_prefix}_depth.png", depth_uint8)
    except Exception as e:
        print(f"Error saving camera data: {e}")


def get_joint_map(object_id):
    """Creates a dictionary mapping joint names to their integer indices."""
    joint_map = {}
    for i in range(p.getNumJoints(object_id)):
        info = p.getJointInfo(object_id, i)
        joint_name = info[1].decode('utf-8')
        joint_map[joint_name] = i
    return joint_map


############################### COGNITIVE AGENT CLASS ###############################

class CognitiveAgent:
    """
    Main cognitive agent that integrates all 10 modules.
    Implements the Sense-Think-Act loop.
    """
    
    def __init__(self, scene_map, learning_enabled=True):
        """Initialize the cognitive agent with all modules."""
        print("=" * 60)
        print("INITIALIZING COGNITIVE ARCHITECTURE")
        print("=" * 60)
        
        # Extract IDs from scene map
        self.robot_id = scene_map['robot_id']
        self.table_id = scene_map.get('table_id')
        self.target_id = scene_map.get('target_id')
        self.obstacle_ids = scene_map.get('obstacle_ids', [])
        
        # M9: Learning & Parameter Optimization
        self.optimizer = ParameterOptimizer()
        if learning_enabled and LOAD_LEARNED_PARAMS:
            self.optimizer.load_parameters()
        
        # M8: Knowledge Base
        self.knowledge_base = create_world_knowledge_base(scene_map)
        
        # M5: State Estimation (Particle Filter)
        initial_pose = tuple(scene_map['robot_position'])
        self.particle_filter = ParticleFilter(
            num_particles=100,
            initial_state=initial_pose,
            initial_uncertainty=0.5
        )
        
        # M6: Motion Controllers
        pid_gains = self.optimizer.parameters['pid_gains']
        self.drive_controller = DifferentialDriveController(
            kp_linear=pid_gains['linear'][0],
            kd_linear=pid_gains['linear'][2],
            kp_angular=pid_gains['angular'][0],
            kd_angular=pid_gains['angular'][2]
        )
        self.wheel_controller = WheelController()
        
        # M7: Action Planning (FSM)
        self.fsm = FiniteStateMachine()
        
        # Path follower
        self.path_follower = None
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.start_time = time.time()
        self.step_count = 0
        
        # State tracking
        self.current_target_position = None
        self.planned_path = None
        
        print("Cognitive Agent Initialized Successfully!")
        print("=" * 60)
    
    def sense(self):
        """MODULE 3: Sensor Acquisition & Preprocessing"""
        rgb, depth, mask = get_camera_image(self.robot_id)
        depth = np.clip(depth, 0.1, 5.0)
        
        lidar = get_lidar_data(self.robot_id)
        joints = get_joint_states(self.robot_id)
        imu = get_imu_data(self.robot_id)
        
        return {
            "rgb": rgb,
            "depth": depth,
            "mask": mask,
            "lidar": lidar,
            "joints": joints,
            "imu": imu
        }
    
    def perceive(self, sensor_data):
        """MODULE 4: Perception"""
        perception_data = process_sensor_data(sensor_data)
        
        # Debug: Print perception info every 240 steps (1 second)
        if self.step_count % 240 == 0:
            target_found = perception_data.get('target') is not None and perception_data.get('target', {}).get('center') is not None
            table_found = perception_data.get('table') is not None and perception_data.get('table', {}).get('center') is not None
            num_obstacles = len(perception_data.get('obstacles', []))
            
            if target_found:
                target_pos = perception_data['target']['center']
                print(f"[Perception] Target FOUND at ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})")
            else:
                print(f"[Perception] Target: NOT FOUND | Table: {table_found} | Obstacles: {num_obstacles}")
        
        return perception_data
    
    def estimate_state(self, sensor_data, control_input):
        """MODULE 5: State Estimation using Particle Filter"""
        self.particle_filter.predict(control_input, DT)
        
        measurement = create_measurement_from_sensors(sensor_data['imu'], sensor_data['joints'])
        if measurement:
            self.particle_filter.update(measurement)
        
        self.particle_filter.resample()
        estimated_pose = self.particle_filter.get_estimate()
        return estimated_pose
    
    def think(self, perception_data, robot_pose):
        """MODULE 7 & 8: Action Planning + Knowledge Reasoning"""
        elapsed_time = time.time() - self.start_time
        action = self.fsm.update(perception_data, robot_pose, elapsed_time, DT)
        
        # Handle path planning when requested
        if action['action'] == 'plan_path':
            obstacles = [obs['center'][:2] for obs in perception_data.get('obstacles', []) 
                        if obs.get('center') is not None]
            
            self.planned_path = self.knowledge_base.plan_path_grid(
                start=robot_pose[:2],
                goal=action['goal'],
                obstacles=obstacles,
                safe_distance=1.0
            )
            
            if self.planned_path:
                waypoints = [(p[0], p[1], 0) for p in self.planned_path]
                self.path_follower = PathFollower(waypoints)
        
        return action
    
    def act(self, action, robot_pose):
        """MODULE 6: Motion Control - Execute actions"""
        action_type = action.get('action', 'idle')
        
        if action_type == 'search':
            omega = action.get('angular_velocity', 0.5)
            self.wheel_controller.apply_velocity_command(self.robot_id, 0.0, omega)
            return (0.0, omega)
        
        elif action_type in ['navigate', 'approach']:
            target_pos = action.get('target', [0, 0, 0])
            target_pose = (target_pos[0], target_pos[1], 0)
            
            if self.path_follower and action_type == 'navigate':
                current_waypoint = self.path_follower.get_current_target()
                if current_waypoint:
                    target_pose = current_waypoint
                    self.path_follower.update(robot_pose[:2])
            
            (v, omega), distance = self.drive_controller.compute_control(
                robot_pose, target_pose, DT
            )
            
            self.wheel_controller.apply_velocity_command(self.robot_id, v, omega)
            self.metrics.avg_distance_error = distance
            
            return (v, omega)
        
        elif action_type == 'done':
            self.wheel_controller.stop(self.robot_id)
            self.metrics.mission_success = True
        
        else:
            self.wheel_controller.stop(self.robot_id)
        
        return (0.0, 0.0)
    
    def learn(self):
        """MODULE 9: Learning - Adapt parameters based on performance"""
        if ENABLE_LEARNING:
            self.optimizer.adapt_online(self.metrics)
            
            pid_gains = self.optimizer.parameters['pid_gains']
            self.drive_controller.linear_pid.kp = pid_gains['linear'][0]
            self.drive_controller.linear_pid.kd = pid_gains['linear'][2]
            self.drive_controller.angular_pid.kp = pid_gains['angular'][0]
            self.drive_controller.angular_pid.kd = pid_gains['angular'][2]
    
    def step(self):
        """Execute one iteration of the Sense-Think-Act loop."""
        self.step_count += 1
        
        # SENSE
        sensor_data = self.sense()
        
        # PERCEIVE
        perception_data = self.perceive(sensor_data)
        
        # ESTIMATE STATE
        control_input = fuse_odometry_imu(sensor_data['imu'], sensor_data['joints'])
        robot_pose = self.estimate_state(sensor_data, control_input)
        
        # THINK
        action = self.think(perception_data, robot_pose)
        
        # ACT
        control_output = self.act(action, robot_pose)
        
        # LEARN (periodically)
        if self.step_count % 240 == 0:
            self.learn()
        
        # Print status update every 2 seconds (480 steps at 240Hz)
        if self.step_count % 480 == 0:
            state = self.fsm.get_current_state()
            elapsed = time.time() - self.start_time
            print(f"[Status] t={elapsed:.1f}s | State={state.name} | Pos=({robot_pose[0]:.2f}, {robot_pose[1]:.2f}) | v={control_output[0]:.2f}m/s, ω={control_output[1]:.2f}rad/s")
        
        # Update metrics
        self.metrics.total_time = time.time() - self.start_time
        
        # Save images if enabled
        if SAVE_IMAGES and self.step_count % 240 == 0:
            save_camera_data(sensor_data['rgb'], sensor_data['depth'], 
                           filename_prefix=f"frame_{self.step_count}")
        
        # Check if mission complete
        if self.fsm.is_mission_complete():
            return True
        
        return False
    
    def finalize(self):
        """Finalize mission and save learning data."""
        print("\n" + "=" * 60)
        print("MISSION COMPLETE")
        print("=" * 60)
        print(f"Success: {self.metrics.mission_success}")
        print(f"Total Time: {self.metrics.total_time:.2f}s")
        print(f"Performance Score: {self.metrics.compute_score():.2f}")
        print("=" * 60)
        
        if ENABLE_LEARNING:
            self.optimizer.record_trial(self.metrics)
            self.optimizer.save_parameters()
            self.optimizer.experience_buffer.save_to_file()
            print("Learning data saved.")
        
        self.wheel_controller.stop(self.robot_id)




############################################ The Main Function ###########################################

def main():
    """
    Main execution function - integrates all 10 modules.
    """
    print("\n" + "=" * 60)
    print(" IIS MASTER PROJECT: AUTONOMOUS NAVIGATE-TO-GRASP CHALLENGE")
    print("=" * 60)
    print("Starting integrated cognitive architecture...")
    print("=" * 60 + "\n")
    
    # M1-M2: Build world with randomized configuration
    print("[Main] Building world with randomized configuration...")
    scene_map = build_world()
    
    # M10: Initialize cognitive agent (integrates M3-M9)
    print("\n[Main] Initializing cognitive agent...")
    agent = CognitiveAgent(scene_map, learning_enabled=ENABLE_LEARNING)
    
    print("\n[Main] Starting mission execution...")
    print("Mission: Navigate to table and grasp target object")
    print("=" * 60 + "\n")
    
    # Main simulation loop
    try:
        while p.isConnected(): # DO NOT TOUCH
            # Execute one step of Sense-Think-Act loop
            mission_complete = agent.step()
            
            if mission_complete:
                # Mission finished (success or failure)
                agent.finalize()
                break
            
            p.stepSimulation()  # DO NOT TOUCH
            time.sleep(1./240.) # DO NOT TOUCH
    
    except KeyboardInterrupt:
        print("\n[Main] Simulation interrupted by user")
        agent.finalize()
    
    except Exception as e:
        print(f"\n[Main] Error during execution: {e}")
        import traceback
        traceback.print_exc()
        agent.finalize()
    
    print("\n[Main] Simulation ended.")


################ The Main Thread ###################################################################

if __name__ == "__main__":
    main()

