import time

class ActionSequencer:
    def __init__(self):
        self.states = ["SEARCH", "NAVIGATE", "GRASP", "COMPLETED"]
        self.current_state = "SEARCH"
        self.target_perceived = False
        self.at_destination = False
        self.grasp_success = False
        self.retry_count = 0

    def update_state(self, perception_data, robot_pose, target_dist):
        if self.current_state == "SEARCH":
            if perception_data['target_found']:
                self.target_perceived = True
                self.current_state = "NAVIGATE"
            else:
                self.execute_search_behavior()

        elif self.current_state == "NAVIGATE":
            if not perception_data['target_found']:
                # Failure recovery: target lost during navigation
                self.current_state = "SEARCH"
            elif target_dist < 0.5: # Threshold to start grasping
                self.current_state = "GRASP"
            else:
                self.execute_navigation_behavior()

        elif self.current_state == "GRASP":
            if perception_data['in_gripper']:
                self.current_state = "COMPLETED"
            elif self.retry_count > 3:
                # Failure recovery: back to search if grasp fails repeatedly
                self.retry_count = 0
                self.current_state = "SEARCH"
            else:
                self.execute_grasp_behavior()
                self.retry_count += 1

    def execute_search_behavior(self):
        # Logic to rotate or move randomly to find target
        pass

    def execute_navigation_behavior(self):
        # Call motion_control to move towards target_pos
        pass

    def execute_grasp_behavior(self):
        # Sequence to move arm and close gripper
        pass

    def get_current_action(self):
        return self.current_state

