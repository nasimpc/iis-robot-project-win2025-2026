"""
Module 9: Learning
Implements parameter optimization and learning from experience.

This module provides:
- PID gain tuning based on performance
- Parameter adaptation based on success/failure
- Experience replay and learning from past trials
"""

import numpy as np
import json
import os
from typing import Dict, Any, List, Tuple


class PerformanceMetrics:
    """
    Track performance metrics for learning.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.mission_success = False
        self.total_time = 0.0
        self.navigation_time = 0.0
        self.grasp_attempts = 0
        self.collisions = 0
        self.distance_traveled = 0.0
        self.avg_distance_error = 0.0
        self.avg_angle_error = 0.0
        self.perception_failures = 0
        
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'mission_success': self.mission_success,
            'total_time': self.total_time,
            'navigation_time': self.navigation_time,
            'grasp_attempts': self.grasp_attempts,
            'collisions': self.collisions,
            'distance_traveled': self.distance_traveled,
            'avg_distance_error': self.avg_distance_error,
            'avg_angle_error': self.avg_angle_error,
            'perception_failures': self.perception_failures
        }
    
    def compute_score(self):
        """
        Compute overall performance score (higher is better).
        
        Returns:
            score: Performance score
        """
        if not self.mission_success:
            return 0.0
        
        # Score based on time (faster is better)
        time_score = max(0, 100.0 - self.total_time)
        
        # Penalty for collisions
        collision_penalty = self.collisions * 20.0
        
        # Penalty for grasp failures
        grasp_penalty = max(0, self.grasp_attempts - 1) * 10.0
        
        # Penalty for navigation inefficiency
        navigation_penalty = self.avg_distance_error * 5.0
        
        score = time_score - collision_penalty - grasp_penalty - navigation_penalty
        return max(0, score)


class ExperienceBuffer:
    """
    Store and manage experience from past trials.
    """
    
    def __init__(self, max_size=100):
        """
        Initialize experience buffer.
        
        Args:
            max_size: Maximum number of experiences to store
        """
        self.max_size = max_size
        self.experiences = []
    
    def add_experience(self, parameters: Dict, metrics: PerformanceMetrics):
        """
        Add a new experience.
        
        Args:
            parameters: Dictionary of parameters used
            metrics: Performance metrics achieved
        """
        experience = {
            'parameters': parameters.copy(),
            'metrics': metrics.to_dict(),
            'score': metrics.compute_score()
        }
        
        self.experiences.append(experience)
        
        # Keep only the most recent experiences
        if len(self.experiences) > self.max_size:
            self.experiences = self.experiences[-self.max_size:]
    
    def get_best_experience(self):
        """
        Get the experience with the highest score.
        
        Returns:
            best_experience: Dict or None if empty
        """
        if not self.experiences:
            return None
        
        return max(self.experiences, key=lambda x: x['score'])
    
    def get_recent_experiences(self, n=10):
        """
        Get the n most recent experiences.
        
        Args:
            n: Number of experiences to retrieve
            
        Returns:
            experiences: List of experience dicts
        """
        return self.experiences[-n:]
    
    def save_to_file(self, filename='experience_buffer.json'):
        """
        Save experience buffer to file.
        
        Args:
            filename: File path
        """
        try:
            with open(filename, 'w') as f:
                json.dump(self.experiences, f, indent=2)
            print(f"[Learning] Saved {len(self.experiences)} experiences to {filename}")
        except Exception as e:
            print(f"[Learning] Error saving experiences: {e}")
    
    def load_from_file(self, filename='experience_buffer.json'):
        """
        Load experience buffer from file.
        
        Args:
            filename: File path
        """
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    self.experiences = json.load(f)
                print(f"[Learning] Loaded {len(self.experiences)} experiences from {filename}")
            else:
                print(f"[Learning] No saved experiences found at {filename}")
        except Exception as e:
            print(f"[Learning] Error loading experiences: {e}")


class PIDTuner:
    """
    Adaptive PID gain tuner based on performance.
    """
    
    def __init__(self, initial_gains: Dict[str, Tuple[float, float, float]]):
        """
        Initialize PID tuner.
        
        Args:
            initial_gains: Dict mapping controller names to (Kp, Ki, Kd) tuples
                          e.g., {'linear': (2.0, 0.0, 0.5), 'angular': (4.0, 0.0, 1.0)}
        """
        self.gains = initial_gains.copy()
        self.learning_rate = 0.1
        self.exploration_noise = 0.05
    
    def adapt_gains(self, controller_name: str, error_metric: float, gradient_estimate: float = None):
        """
        Adapt PID gains based on error metric.
        
        Args:
            controller_name: Name of controller to adapt
            error_metric: Performance error (e.g., avg distance error)
            gradient_estimate: Optional gradient estimate for directed adaptation
        """
        if controller_name not in self.gains:
            return
        
        kp, ki, kd = self.gains[controller_name]
        
        if gradient_estimate is not None:
            # Gradient-based adaptation
            delta_kp = -self.learning_rate * gradient_estimate
        else:
            # Simple heuristic: if error is high, increase proportional gain
            if error_metric > 0.5:
                delta_kp = self.learning_rate * kp * 0.1
            else:
                delta_kp = -self.learning_rate * kp * 0.05
        
        # Add exploration noise
        delta_kp += np.random.normal(0, self.exploration_noise)
        
        # Update gain with limits
        new_kp = np.clip(kp + delta_kp, 0.1, 10.0)
        
        # Also adapt derivative gain proportionally
        new_kd = kd * (new_kp / kp) if kp > 0 else kd
        new_kd = np.clip(new_kd, 0.0, 5.0)
        
        self.gains[controller_name] = (new_kp, ki, new_kd)
        
        print(f"[Learning] Adapted {controller_name} gains: Kp={new_kp:.3f}, Kd={new_kd:.3f}")
    
    def get_gains(self, controller_name: str):
        """
        Get current gains for a controller.
        
        Args:
            controller_name: Name of controller
            
        Returns:
            (Kp, Ki, Kd): Tuple of gains
        """
        return self.gains.get(controller_name, (1.0, 0.0, 0.1))
    
    def reset_to_best(self, best_parameters: Dict):
        """
        Reset gains to best known parameters.
        
        Args:
            best_parameters: Dictionary containing 'pid_gains'
        """
        if 'pid_gains' in best_parameters:
            self.gains = best_parameters['pid_gains'].copy()
            print(f"[Learning] Reset to best gains: {self.gains}")


class ParameterOptimizer:
    """
    General parameter optimizer using experience-based learning.
    """
    
    def __init__(self):
        """Initialize parameter optimizer."""
        self.parameters = {
            'pid_gains': {
                'linear': (2.0, 0.0, 0.5),
                'angular': (4.0, 0.0, 1.0)
            },
            'perception': {
                'color_tolerance': 0.3,
                'ransac_iterations': 200
            },
            'navigation': {
                'waypoint_tolerance': 0.3,
                'max_linear_velocity': 1.5,
                'max_angular_velocity': 2.0
            }
        }
        
        self.experience_buffer = ExperienceBuffer()
        self.pid_tuner = PIDTuner(self.parameters['pid_gains'])
    
    def optimize_from_experience(self):
        """
        Optimize parameters based on past experiences.
        """
        best_exp = self.experience_buffer.get_best_experience()
        
        if best_exp is None:
            print("[Learning] No experiences available for optimization")
            return
        
        print(f"[Learning] Best experience score: {best_exp['score']:.2f}")
        
        # Update parameters to best known
        if 'parameters' in best_exp:
            best_params = best_exp['parameters']
            
            # Update PID gains
            if 'pid_gains' in best_params:
                self.pid_tuner.reset_to_best(best_params)
                self.parameters['pid_gains'] = best_params['pid_gains']
            
            # Update other parameters
            for key in ['perception', 'navigation']:
                if key in best_params:
                    self.parameters[key] = best_params[key]
        
        print("[Learning] Parameters optimized from experience")
    
    def record_trial(self, metrics: PerformanceMetrics):
        """
        Record a trial's results.
        
        Args:
            metrics: Performance metrics from the trial
        """
        current_params = {
            'pid_gains': self.parameters['pid_gains'].copy(),
            'perception': self.parameters['perception'].copy(),
            'navigation': self.parameters['navigation'].copy()
        }
        
        self.experience_buffer.add_experience(current_params, metrics)
        
        print(f"[Learning] Trial recorded - Success: {metrics.mission_success}, " + 
              f"Score: {metrics.compute_score():.2f}")
    
    def adapt_online(self, metrics: PerformanceMetrics):
        """
        Perform online adaptation during mission.
        
        Args:
            metrics: Current performance metrics
        """
        # Adapt PID gains based on tracking errors
        if metrics.avg_distance_error > 0:
            self.pid_tuner.adapt_gains('linear', metrics.avg_distance_error)
        
        if metrics.avg_angle_error > 0:
            self.pid_tuner.adapt_gains('angular', metrics.avg_angle_error)
        
        # Update parameters
        self.parameters['pid_gains'] = dict(self.pid_tuner.gains)
    
    def get_parameter(self, category: str, param_name: str):
        """
        Get a specific parameter value.
        
        Args:
            category: Parameter category ('pid_gains', 'perception', etc.)
            param_name: Parameter name
            
        Returns:
            value: Parameter value or None
        """
        if category in self.parameters:
            return self.parameters[category].get(param_name, None)
        return None
    
    def save_parameters(self, filename='learned_parameters.json'):
        """
        Save current parameters to file.
        
        Args:
            filename: File path
        """
        try:
            with open(filename, 'w') as f:
                json.dump(self.parameters, f, indent=2)
            print(f"[Learning] Parameters saved to {filename}")
        except Exception as e:
            print(f"[Learning] Error saving parameters: {e}")
    
    def load_parameters(self, filename='learned_parameters.json'):
        """
        Load parameters from file.
        
        Args:
            filename: File path
        """
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    self.parameters = json.load(f)
                self.pid_tuner = PIDTuner(self.parameters['pid_gains'])
                print(f"[Learning] Parameters loaded from {filename}")
            else:
                print(f"[Learning] No saved parameters found at {filename}")
        except Exception as e:
            print(f"[Learning] Error loading parameters: {e}")


# Testing function
if __name__ == "__main__":
    print("Learning module loaded successfully.")
    
    # Test performance metrics
    metrics = PerformanceMetrics()
    metrics.mission_success = True
    metrics.total_time = 50.0
    metrics.collisions = 1
    metrics.grasp_attempts = 2
    metrics.avg_distance_error = 0.2
    
    score = metrics.compute_score()
    print(f"\nPerformance score: {score:.2f}")
    
    # Test parameter optimizer
    optimizer = ParameterOptimizer()
    print(f"\nInitial PID gains: {optimizer.parameters['pid_gains']}")
    
    # Record trial
    optimizer.record_trial(metrics)
    
    # Test adaptation
    optimizer.adapt_online(metrics)
    print(f"Adapted PID gains: {optimizer.parameters['pid_gains']}")

