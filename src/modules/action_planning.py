"""
Module 7: Action Planning
Implements a Finite State Machine (FSM) for high-level mission sequencing.

Mission Sequence: INITIALIZE -> SEARCH -> NAVIGATE -> APPROACH -> GRASP -> LIFT -> DONE
Includes failure recovery and state transitions.
"""

from enum import Enum
import numpy as np
import time
from typing import Optional, Dict, Any


class RobotState(Enum):
    """
    Enumeration of possible robot states in the mission.
    """
    INITIALIZE = "INITIALIZE"           # Initialize sensors and world knowledge
    SEARCH = "SEARCH"                   # Search for target object
    PLAN_PATH = "PLAN_PATH"             # Plan collision-free path to table
    NAVIGATE = "NAVIGATE"               # Navigate to table location
    APPROACH = "APPROACH"               # Approach the table (fine positioning)
    REACH = "REACH"                     # Move arm toward target
    GRASP = "GRASP"                     # Grasp the target object
    LIFT = "LIFT"                       # Lift the target object
    DONE = "DONE"                       # Mission complete
    FAILED = "FAILED"                   # Mission failed
    RECOVERY = "RECOVERY"               # Recovery from failure


class FiniteStateMachine:
    """
    Finite State Machine for high-level action planning and mission sequencing.
    """
    
    def __init__(self, initial_state=RobotState.INITIALIZE):
        """
        Initialize FSM.
        
        Args:
            initial_state: Starting state
        """
        self.current_state = initial_state
        self.previous_state = None
        self.state_start_time = 0
        self.state_data = {}  # Storage for state-specific data
        
        # Transition conditions
        self.max_search_time = 100.0  # Maximum time to search for target (seconds)
        self.max_navigation_time = 200.0  # Maximum time to navigate
        self.max_approach_time = 50.0
        self.max_reach_time = 50.0
        self.max_grasp_time = 30.0
        
        # Failure recovery
        self.retry_count = 0
        self.max_retries = 3
    
    def transition_to(self, new_state):
        """
        Transition to a new state.
        
        Args:
            new_state: Target state (RobotState enum)
        """
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_start_time = time.time()  # Reset timer to current time
        print(f"[FSM] Transitioning: {self.previous_state.value} -> {self.current_state.value}")
    
    def update(self, perception_data, robot_pose, elapsed_time, dt):
        """
        Update FSM based on current perception and robot state.
        
        Args:
            perception_data: Output from perception module
            robot_pose: Current robot pose (x, y, theta)
            elapsed_time: Total elapsed time
            dt: Time step
            
        Returns:
            action: Dict containing action to execute
        """
        self.state_start_time += dt
        
        # State machine logic
        if self.current_state == RobotState.INITIALIZE:
            return self._handle_initialize(perception_data, robot_pose)
        
        elif self.current_state == RobotState.SEARCH:
            return self._handle_search(perception_data, robot_pose)
        
        elif self.current_state == RobotState.PLAN_PATH:
            return self._handle_plan_path(perception_data, robot_pose)
        
        elif self.current_state == RobotState.NAVIGATE:
            return self._handle_navigate(perception_data, robot_pose)
        
        elif self.current_state == RobotState.APPROACH:
            return self._handle_approach(perception_data, robot_pose)
        
        elif self.current_state == RobotState.REACH:
            return self._handle_reach(perception_data, robot_pose)
        
        elif self.current_state == RobotState.GRASP:
            return self._handle_grasp(perception_data, robot_pose)
        
        elif self.current_state == RobotState.LIFT:
            return self._handle_lift(perception_data, robot_pose)
        
        elif self.current_state == RobotState.DONE:
            return self._handle_done()
        
        elif self.current_state == RobotState.FAILED:
            return self._handle_failed()
        
        elif self.current_state == RobotState.RECOVERY:
            return self._handle_recovery(perception_data, robot_pose)
        
        else:
            return {'action': 'idle'}
    
    def _handle_initialize(self, perception_data, robot_pose):
        """
        INITIALIZE state: Set up initial knowledge and prepare for mission.
        """
        # Initialization complete, move to search
        self.transition_to(RobotState.SEARCH)
        return {
            'action': 'initialize',
            'command': 'Setup complete'
        }
    
    def _handle_search(self, perception_data, robot_pose):
        """
        SEARCH state: Look for target object and table.
        """
        # Check if target is detected
        if perception_data.get('target') is not None and perception_data['target']['center'] is not None:
            # Target found!
            target_pos = perception_data['target']['center']
            self.state_data['target_position'] = target_pos
            
            # Also check for table
            if perception_data.get('table') is not None and perception_data['table']['center'] is not None:
                self.state_data['table_position'] = perception_data['table']['center']
                print(f"[FSM] Target found at {target_pos}, transitioning to path planning")
                self.transition_to(RobotState.PLAN_PATH)
                return {
                    'action': 'search_complete',
                    'target_found': True,
                    'target_position': target_pos
                }
        
        # Check timeout
        if time.time() - self.state_start_time > self.max_search_time:
            print(f"[FSM] Search timeout, moving to recovery")
            self.transition_to(RobotState.RECOVERY)
            return {'action': 'search_timeout'}
        
        # Continue searching - rotate in place to scan environment
        return {
            'action': 'search',
            'command': 'rotate_scan',
            'angular_velocity': 0.5  # Slow rotation
        }
    
    def _handle_plan_path(self, perception_data, robot_pose):
        """
        PLAN_PATH state: Plan collision-free path to table using Prolog.
        """
        # Extract obstacle positions
        obstacles = perception_data.get('obstacles', [])
        obstacle_positions = [obs['center'] for obs in obstacles if obs['center'] is not None]
        
        # Store path planning data
        self.state_data['obstacles'] = obstacle_positions
        self.state_data['start'] = robot_pose[:2]  # (x, y)
        
        if 'table_position' in self.state_data:
            self.state_data['goal'] = self.state_data['table_position'][:2]
            
            # Path planning will be done by knowledge_reasoning module
            self.transition_to(RobotState.NAVIGATE)
            return {
                'action': 'plan_path',
                'start': self.state_data['start'],
                'goal': self.state_data['goal'],
                'obstacles': obstacle_positions
            }
        else:
            # No table found, go back to search
            self.transition_to(RobotState.SEARCH)
            return {'action': 'no_table_found'}
    
    def _handle_navigate(self, perception_data, robot_pose):
        """
        NAVIGATE state: Follow planned path to table.
        """
        # Check if we've reached the table vicinity
        if 'table_position' in self.state_data:
            table_pos = self.state_data['table_position']
            distance_to_table = np.sqrt((robot_pose[0] - table_pos[0])**2 + 
                                       (robot_pose[1] - table_pos[1])**2)
            
            # If close enough, transition to approach
            if distance_to_table < 1.5:  # Within 1.5 meters
                print(f"[FSM] Reached table vicinity, transitioning to approach")
                self.transition_to(RobotState.APPROACH)
                return {'action': 'navigation_complete'}
        
        # Check timeout
        if time.time() - self.state_start_time > self.max_navigation_time:
            print(f"[FSM] Navigation timeout")
            self.transition_to(RobotState.RECOVERY)
            return {'action': 'navigation_timeout'}
        
        # Continue navigation
        return {
            'action': 'navigate',
            'target': self.state_data.get('table_position', [0, 0, 0])
        }
    
    def _handle_approach(self, perception_data, robot_pose):
        """
        APPROACH state: Fine positioning near table.
        """
        if 'table_position' in self.state_data:
            table_pos = self.state_data['table_position']
            distance_to_table = np.sqrt((robot_pose[0] - table_pos[0])**2 + 
                                       (robot_pose[1] - table_pos[1])**2)
            
            # If in good position for arm reach
            if distance_to_table < 0.8:  # Within arm reach
                print(f"[FSM] In position for grasping, transitioning to reach")
                self.transition_to(RobotState.REACH)
                return {'action': 'approach_complete'}
        
        # Check timeout
        if time.time() - self.state_start_time > self.max_approach_time:
            self.transition_to(RobotState.RECOVERY)
            return {'action': 'approach_timeout'}
        
        # Continue approaching
        return {
            'action': 'approach',
            'target': self.state_data.get('table_position', [0, 0, 0]),
            'precision': True  # Use slower, more precise control
        }
    
    def _handle_reach(self, perception_data, robot_pose):
        """
        REACH state: Move arm toward target object.
        """
        # Check if target is still visible
        if perception_data.get('target') is not None and perception_data['target']['center'] is not None:
            target_pos = perception_data['target']['center']
            
            # Check if arm is close to target
            # This would need end effector position - simplified here
            if time.time() - self.state_start_time > 5.0:  # Assume reach takes ~5 seconds
                print(f"[FSM] Arm reached target, transitioning to grasp")
                self.transition_to(RobotState.GRASP)
                return {'action': 'reach_complete', 'target_position': target_pos}
            
            return {
                'action': 'reach',
                'target_position': target_pos
            }
        else:
            # Lost sight of target
            print(f"[FSM] Lost target visibility")
            self.transition_to(RobotState.RECOVERY)
            return {'action': 'target_lost'}
    
    def _handle_grasp(self, perception_data, robot_pose):
        """
        GRASP state: Close gripper on target object.
        """
        # Simulate grasp execution
        if time.time() - self.state_start_time > 3.0:  # Grasp takes ~3 seconds
            print(f"[FSM] Grasp complete, transitioning to lift")
            self.transition_to(RobotState.LIFT)
            return {'action': 'grasp_complete'}
        
        return {
            'action': 'grasp',
            'command': 'close_gripper'
        }
    
    def _handle_lift(self, perception_data, robot_pose):
        """
        LIFT state: Lift the grasped object.
        """
        # Check if object is lifted (would need force/contact sensing)
        if time.time() - self.state_start_time > 3.0:  # Lift takes ~3 seconds
            print(f"[FSM] Object lifted successfully! Mission complete!")
            self.transition_to(RobotState.DONE)
            return {'action': 'lift_complete', 'success': True}
        
        return {
            'action': 'lift',
            'lift_height': 0.2  # Lift 20cm
        }
    
    def _handle_done(self):
        """
        DONE state: Mission complete.
        """
        return {
            'action': 'done',
            'success': True,
            'message': 'Mission completed successfully!'
        }
    
    def _handle_failed(self):
        """
        FAILED state: Mission failed.
        """
        return {
            'action': 'failed',
            'success': False,
            'message': 'Mission failed after maximum retries.'
        }
    
    def _handle_recovery(self, perception_data, robot_pose):
        """
        RECOVERY state: Attempt to recover from failure.
        """
        self.retry_count += 1
        
        if self.retry_count >= self.max_retries:
            print(f"[FSM] Max retries exceeded, mission failed")
            self.transition_to(RobotState.FAILED)
            return {'action': 'max_retries_exceeded'}
        
        # Reset to search state
        print(f"[FSM] Recovery attempt {self.retry_count}/{self.max_retries}")
        self.transition_to(RobotState.SEARCH)
        return {
            'action': 'recovery',
            'retry_count': self.retry_count
        }
    
    def get_current_state(self):
        """Get current state."""
        return self.current_state
    
    def is_mission_complete(self):
        """Check if mission is complete (success or failure)."""
        return self.current_state in [RobotState.DONE, RobotState.FAILED]


# Testing function
if __name__ == "__main__":
    print("Action Planning module loaded successfully.")
    print("Available states:", [state.value for state in RobotState])
    
    # Simple FSM test
    fsm = FiniteStateMachine()
    print(f"\nInitial state: {fsm.get_current_state().value}")
    
    # Simulate state transitions
    perception = {'target': None, 'table': None, 'obstacles': []}
    robot_pose = (0, 0, 0)
    
    for i in range(5):
        action = fsm.update(perception, robot_pose, elapsed_time=i, dt=0.1)
        print(f"Step {i}: State={fsm.get_current_state().value}, Action={action}")

