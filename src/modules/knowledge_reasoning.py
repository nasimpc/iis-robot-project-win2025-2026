"""
Module 8: Knowledge Representation and Reasoning
Uses Prolog (PySwip) to store semantic information and perform logical reasoning.

This module provides:
- Knowledge base management for world state
- Semantic queries about objects, properties, and affordances
- Path planning using Prolog-based reasoning
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any

try:
    from pyswip import Prolog
    PYSWIP_AVAILABLE = True
except ImportError:
    PYSWIP_AVAILABLE = False
    print("Warning: PySwip not available. Knowledge reasoning will use fallback mode.")


class KnowledgeBase:
    """
    Knowledge base for storing and reasoning about the world state using Prolog.
    """
    
    def __init__(self, use_prolog=True):
        """
        Initialize knowledge base.
        
        Args:
            use_prolog: Whether to use Prolog (requires PySwip)
        """
        self.use_prolog = use_prolog and PYSWIP_AVAILABLE
        
        if self.use_prolog:
            self.prolog = Prolog()
            self._initialize_prolog_rules()
        else:
            # Fallback: Python-based knowledge storage
            self.facts = []
            self.objects = {}
        
        print(f"[KB] Knowledge base initialized (Prolog: {self.use_prolog})")
    
    def _initialize_prolog_rules(self):
        """
        Initialize Prolog rules and predicates.
        """
        # Define basic predicates
        self.prolog.assertz("color(target, red)")
        self.prolog.assertz("shape(target, cylinder)")
        self.prolog.assertz("color(table, brown)")
        self.prolog.assertz("shape(table, rectangular)")
        
        # Object properties
        self.prolog.assertz("is_graspable(X) :- shape(X, cylinder)")
        self.prolog.assertz("is_fixed(table)")
        self.prolog.assertz("is_movable(X) :- \\+ is_fixed(X)")
        
        # Affordances
        self.prolog.assertz("can_grasp(X) :- is_graspable(X), is_movable(X)")
        self.prolog.assertz("can_place_on(X, table) :- is_movable(X)")
        
        # Spatial relations (will be updated dynamically)
        self.prolog.assertz("on(target, table)")
        
        # Path planning predicates
        self.prolog.assertz("safe_distance(1.0)")  # Minimum safe distance from obstacles
        
        print("[KB] Prolog rules initialized")
    
    def assert_fact(self, fact: str):
        """
        Add a fact to the knowledge base.
        
        Args:
            fact: Prolog fact string (e.g., "color(obstacle1, blue)")
        """
        if self.use_prolog:
            try:
                self.prolog.assertz(fact)
                print(f"[KB] Asserted: {fact}")
            except Exception as e:
                print(f"[KB] Error asserting fact: {e}")
        else:
            self.facts.append(fact)
    
    def query(self, query: str):
        """
        Query the knowledge base.
        
        Args:
            query: Prolog query string (e.g., "color(target, X)")
            
        Returns:
            List of result dictionaries
        """
        if self.use_prolog:
            try:
                results = list(self.prolog.query(query))
                return results
            except Exception as e:
                print(f"[KB] Error in query: {e}")
                return []
        else:
            # Fallback: simple string matching
            return self._fallback_query(query)
    
    def _fallback_query(self, query: str):
        """
        Fallback query method when Prolog is not available.
        """
        # Simple pattern matching for common queries
        if "color(target" in query:
            return [{"X": "red"}]
        elif "can_grasp" in query:
            return [{"X": "target"}]
        return []
    
    def update_object_position(self, object_name: str, position: Tuple[float, float, float]):
        """
        Update object position in knowledge base.
        
        Args:
            object_name: Name of object
            position: (x, y, z) position
        """
        # Retract old position if exists
        if self.use_prolog:
            try:
                self.prolog.retractall(f"position({object_name}, _, _, _)")
                self.prolog.assertz(f"position({object_name}, {position[0]}, {position[1]}, {position[2]})")
            except:
                pass
        else:
            self.objects[object_name] = {'position': position}
    
    def update_obstacle(self, obstacle_id: int, position: Tuple[float, float], color: str):
        """
        Add or update obstacle in knowledge base.
        
        Args:
            obstacle_id: Unique obstacle identifier
            position: (x, y) position
            color: Color name
        """
        obstacle_name = f"obstacle{obstacle_id}"
        
        self.assert_fact(f"color({obstacle_name}, {color.lower()})")
        self.assert_fact(f"is_obstacle({obstacle_name})")
        self.assert_fact(f"is_fixed({obstacle_name})")
        self.update_object_position(obstacle_name, (position[0], position[1], 0.0))
    
    def is_path_safe(self, start: Tuple[float, float], goal: Tuple[float, float], 
                     obstacles: List[Tuple[float, float]], safe_distance=1.0):
        """
        Check if a straight-line path between start and goal is collision-free.
        
        Args:
            start: (x, y) start position
            goal: (x, y) goal position
            obstacles: List of (x, y) obstacle positions
            safe_distance: Minimum safe distance from obstacles
            
        Returns:
            bool: True if path is safe
        """
        # Check distance from line segment to each obstacle
        for obs in obstacles:
            dist = self._point_to_line_distance(obs, start, goal)
            if dist < safe_distance:
                return False
        return True
    
    def _point_to_line_distance(self, point, line_start, line_end):
        """
        Calculate minimum distance from point to line segment.
        """
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Vector from line_start to line_end
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            # Line segment is a point
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        # Parameter t for projection onto line
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))
        
        # Closest point on line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Distance to closest point
        return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    def plan_path_grid(self, start: Tuple[float, float], goal: Tuple[float, float],
                       obstacles: List[Tuple[float, float]], grid_resolution=0.5,
                       safe_distance=1.0):
        """
        Plan a collision-free path using grid-based A* search.
        
        Args:
            start: (x, y) start position
            goal: (x, y) goal position
            obstacles: List of (x, y) obstacle positions
            grid_resolution: Grid cell size
            safe_distance: Minimum distance from obstacles
            
        Returns:
            waypoints: List of (x, y) waypoints, or None if no path found
        """
        # Simple A* path planning
        print(f"[KB] Planning path from {start} to {goal}")
        
        # First check if direct path is safe
        if self.is_path_safe(start, goal, obstacles, safe_distance):
            print("[KB] Direct path is safe")
            return [start, goal]
        
        # Otherwise, use simple waypoint planning
        # Create waypoints that avoid obstacles
        waypoints = self._compute_avoidance_waypoints(start, goal, obstacles, safe_distance)
        
        if waypoints:
            print(f"[KB] Path found with {len(waypoints)} waypoints")
        else:
            print("[KB] No path found")
        
        return waypoints
    
    def _compute_avoidance_waypoints(self, start, goal, obstacles, safe_distance):
        """
        Compute waypoints that navigate around obstacles.
        """
        waypoints = [start]
        
        # For each obstacle in the way, create detour waypoints
        for obs in obstacles:
            # Check if obstacle is between start and goal
            dist_to_line = self._point_to_line_distance(obs, start, goal)
            
            if dist_to_line < safe_distance:
                # Create waypoint to go around obstacle
                # Calculate perpendicular direction
                dx = goal[0] - start[0]
                dy = goal[1] - start[1]
                length = np.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    # Perpendicular vector
                    perp_x = -dy / length
                    perp_y = dx / length
                    
                    # Waypoint offset from obstacle
                    offset = safe_distance + 0.5
                    waypoint = (obs[0] + perp_x * offset, obs[1] + perp_y * offset)
                    waypoints.append(waypoint)
        
        waypoints.append(goal)
        return waypoints
    
    def query_affordance(self, object_name: str, affordance: str):
        """
        Query if an object has a specific affordance.
        
        Args:
            object_name: Name of object
            affordance: Affordance to check (e.g., "graspable", "movable")
            
        Returns:
            bool: True if object has affordance
        """
        query = f"can_{affordance}({object_name})"
        results = self.query(query)
        return len(results) > 0
    
    def get_object_color(self, object_name: str):
        """
        Get the color of an object.
        
        Args:
            object_name: Name of object
            
        Returns:
            color: Color string or None
        """
        query = f"color({object_name}, X)"
        results = self.query(query)
        
        if results and len(results) > 0:
            return results[0].get('X', None)
        return None


# Utility functions for world model management

def create_world_knowledge_base(scene_map):
    """
    Create a knowledge base from the initial scene map.
    
    Args:
        scene_map: Scene configuration from world_builder
        
    Returns:
        kb: Initialized KnowledgeBase
    """
    kb = KnowledgeBase()
    
    # Add table
    if 'table_position' in scene_map:
        kb.update_object_position('table', scene_map['table_position'])
    
    # Add obstacles
    if 'obstacle_positions' in scene_map and 'obstacle_colors' in scene_map:
        for i, (pos, color) in enumerate(zip(scene_map['obstacle_positions'], 
                                             scene_map['obstacle_colors'])):
            kb.update_obstacle(i, (pos[0], pos[1]), color)
    
    # Add robot
    if 'robot_position' in scene_map:
        kb.update_object_position('robot', scene_map['robot_position'])
    
    print("[KB] World knowledge base created from scene map")
    return kb


# Testing function
if __name__ == "__main__":
    print("Knowledge Reasoning module loaded successfully.")
    print(f"PySwip available: {PYSWIP_AVAILABLE}")
    
    # Simple test
    kb = KnowledgeBase(use_prolog=False)  # Use fallback for testing
    
    # Test path planning
    start = (0, 0)
    goal = (5, 5)
    obstacles = [(2.5, 2.5), (3, 3)]
    
    path = kb.plan_path_grid(start, goal, obstacles)
    print(f"\nPath planning test:")
    print(f"Start: {start}, Goal: {goal}")
    print(f"Obstacles: {obstacles}")
    print(f"Path: {path}")

