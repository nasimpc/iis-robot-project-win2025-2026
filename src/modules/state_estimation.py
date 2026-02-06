"""
Module 5: State Estimation
Implements a Particle Filter to fuse noisy sensor data and control inputs into reliable state estimates.

The Particle Filter estimates robot state (x, y, theta) by:
- Predicting particle states based on motion model
- Updating particle weights based on sensor measurements
- Resampling particles based on weights
"""

import numpy as np
from typing import List, Tuple, Dict


class Particle:
    """
    Represents a single particle in the particle filter.
    Each particle is a hypothesis of the robot's state.
    """
    def __init__(self, x, y, theta, weight=1.0):
        self.x = x
        self.y = y
        self.theta = theta  # Orientation in radians
        self.weight = weight
    
    def __repr__(self):
        return f"Particle(x={self.x:.2f}, y={self.y:.2f}, θ={self.theta:.2f}, w={self.weight:.4f})"


class ParticleFilter:
    """
    Particle Filter for robot state estimation.
    Estimates (x, y, theta) from noisy sensor data.
    """
    
    def __init__(self, num_particles=100, initial_state=(0, 0, 0), initial_uncertainty=0.5):
        """
        Initialize the particle filter.
        
        Args:
            num_particles: Number of particles to use
            initial_state: (x, y, theta) initial estimate
            initial_uncertainty: Standard deviation for initial particle distribution
        """
        self.num_particles = num_particles
        self.particles = []
        
        # Initialize particles around initial state
        x0, y0, theta0 = initial_state
        for _ in range(num_particles):
            x = np.random.normal(x0, initial_uncertainty)
            y = np.random.normal(y0, initial_uncertainty)
            theta = np.random.normal(theta0, initial_uncertainty * 0.5)
            self.particles.append(Particle(x, y, theta, 1.0 / num_particles))
        
        # Motion noise parameters
        self.motion_noise_translation = 0.1  # Noise in linear motion
        self.motion_noise_rotation = 0.05    # Noise in angular motion
        
        # Sensor noise parameters
        self.sensor_noise = 0.2
        
    def predict(self, control_input, dt=1.0/240.0):
        """
        Prediction step: Move particles based on control input and motion model.
        
        Args:
            control_input: (v, omega) linear and angular velocity
            dt: Time step
        """
        v, omega = control_input
        
        for particle in self.particles:
            # Motion model with noise
            v_noisy = v + np.random.normal(0, self.motion_noise_translation)
            omega_noisy = omega + np.random.normal(0, self.motion_noise_rotation)
            
            # Update particle state using differential drive model
            if abs(omega_noisy) < 1e-6:
                # Straight line motion
                particle.x += v_noisy * np.cos(particle.theta) * dt
                particle.y += v_noisy * np.sin(particle.theta) * dt
            else:
                # Circular arc motion
                radius = v_noisy / omega_noisy
                particle.x += radius * (np.sin(particle.theta + omega_noisy * dt) - np.sin(particle.theta))
                particle.y += radius * (-np.cos(particle.theta + omega_noisy * dt) + np.cos(particle.theta))
                particle.theta += omega_noisy * dt
            
            # Normalize theta to [-pi, pi]
            particle.theta = np.arctan2(np.sin(particle.theta), np.cos(particle.theta))
    
    def update(self, sensor_measurement):
        """
        Update step: Update particle weights based on sensor measurements.
        
        Args:
            sensor_measurement: Dict containing sensor data (e.g., IMU, odometry, landmarks)
        """
        # Extract measurements
        if 'position' in sensor_measurement:
            measured_x, measured_y = sensor_measurement['position'][:2]
            
            # Calculate weights based on how well each particle matches the measurement
            for particle in self.particles:
                # Distance between particle and measurement
                distance = np.sqrt((particle.x - measured_x)**2 + (particle.y - measured_y)**2)
                
                # Gaussian likelihood: weight decreases with distance
                particle.weight *= self._gaussian_likelihood(distance, 0, self.sensor_noise)
        
        if 'orientation' in sensor_measurement:
            measured_theta = sensor_measurement['orientation']
            
            for particle in self.particles:
                # Angular difference
                angle_diff = self._angle_difference(particle.theta, measured_theta)
                
                # Gaussian likelihood for orientation
                particle.weight *= self._gaussian_likelihood(angle_diff, 0, self.sensor_noise * 0.5)
        
        # Normalize weights
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight
        else:
            # If all weights are zero, reset to uniform
            for particle in self.particles:
                particle.weight = 1.0 / self.num_particles
    
    def resample(self):
        """
        Resample particles based on their weights (Low Variance Resampling).
        This removes low-weight particles and duplicates high-weight particles.
        """
        # Low variance resampling
        new_particles = []
        weights = [p.weight for p in self.particles]
        
        # Generate random starting point
        r = np.random.uniform(0, 1.0 / self.num_particles)
        c = weights[0]
        i = 0
        
        for m in range(self.num_particles):
            u = r + m * (1.0 / self.num_particles)
            while u > c:
                i += 1
                if i >= len(weights):
                    i = len(weights) - 1
                    break
                c += weights[i]
            
            # Copy selected particle
            p = self.particles[i]
            new_particles.append(Particle(p.x, p.y, p.theta, 1.0 / self.num_particles))
        
        self.particles = new_particles
    
    def get_estimate(self):
        """
        Get the current best estimate of robot state.
        
        Returns:
            (x, y, theta): Weighted average of particle states
        """
        x = sum(p.x * p.weight for p in self.particles)
        y = sum(p.y * p.weight for p in self.particles)
        
        # For theta, use circular mean
        sin_sum = sum(np.sin(p.theta) * p.weight for p in self.particles)
        cos_sum = sum(np.cos(p.theta) * p.weight for p in self.particles)
        theta = np.arctan2(sin_sum, cos_sum)
        
        return (x, y, theta)
    
    def get_variance(self):
        """
        Get the variance of the particle distribution (measure of uncertainty).
        
        Returns:
            (var_x, var_y, var_theta): Variance in each dimension
        """
        estimate = self.get_estimate()
        x_est, y_est, theta_est = estimate
        
        var_x = sum(((p.x - x_est) ** 2) * p.weight for p in self.particles)
        var_y = sum(((p.y - y_est) ** 2) * p.weight for p in self.particles)
        
        # Circular variance for theta
        var_theta = 1.0 - np.sqrt(sum(np.cos(p.theta - theta_est) * p.weight for p in self.particles) ** 2 +
                                  sum(np.sin(p.theta - theta_est) * p.weight for p in self.particles) ** 2)
        
        return (var_x, var_y, var_theta)
    
    def _gaussian_likelihood(self, x, mu, sigma):
        """
        Calculate Gaussian likelihood for a measurement.
        
        Args:
            x: Measured value
            mu: Expected value
            sigma: Standard deviation
            
        Returns:
            Likelihood (unnormalized)
        """
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    def _angle_difference(self, angle1, angle2):
        """
        Calculate the smallest difference between two angles.
        
        Args:
            angle1: First angle in radians
            angle2: Second angle in radians
            
        Returns:
            Difference in range [-pi, pi]
        """
        diff = angle1 - angle2
        return np.arctan2(np.sin(diff), np.cos(diff))


def fuse_odometry_imu(imu_data, joint_states, wheel_indices=[2, 3, 4, 5], wheel_radius=0.1651, wheel_base=0.555):
    """
    Fuse IMU and wheel odometry to estimate control inputs (v, omega).
    
    Args:
        imu_data: Dict with 'gyroscope_data' and 'accelerometer_data'
        joint_states: Dict with joint names as keys (from get_joint_states)
        wheel_indices: Indices of wheel joints
        wheel_radius: Radius of wheels in meters
        wheel_base: Distance between left and right wheels in meters
        
    Returns:
        (v, omega): Linear and angular velocity
    """
    # Convert joint_states dict to list indexed by joint index
    joint_list = {}
    for joint_name, joint_info in joint_states.items():
        idx = joint_info['index']
        joint_list[idx] = joint_info['velocity']
    
    # Extract wheel velocities
    if all(i in joint_list for i in wheel_indices):
        left_wheels = [joint_list[i] for i in wheel_indices[:2]]  # First two are left
        right_wheels = [joint_list[i] for i in wheel_indices[2:]]  # Last two are right
        
        # Average velocities
        v_left = np.mean(left_wheels) * wheel_radius
        v_right = np.mean(right_wheels) * wheel_radius
        
        # Differential drive model
        v = (v_left + v_right) / 2.0
        omega = (v_right - v_left) / wheel_base
    else:
        v = 0.0
        omega = 0.0
    
    # Optionally fuse with IMU gyroscope for omega
    if 'gyroscope_data' in imu_data:
        omega_imu = imu_data['gyroscope_data'][2]  # Z-axis rotation
        omega = 0.7 * omega + 0.3 * omega_imu  # Weighted fusion
    
    return (v, omega)


def create_measurement_from_sensors(imu_data, joint_states):
    """
    Create a sensor measurement dict from raw sensor data for the particle filter.
    
    Args:
        imu_data: IMU sensor data
        joint_states: Joint states
        
    Returns:
        measurement: Dict with 'position' and 'orientation' estimates
    """
    measurement = {}
    
    # For now, we don't have direct position measurement
    # In a real system, you might integrate accelerations or use landmarks
    
    # Orientation from IMU
    if 'orientation' in imu_data:
        # Convert quaternion to euler angle (yaw)
        quat = imu_data['orientation']
        # Simple yaw extraction from quaternion
        yaw = np.arctan2(2.0 * (quat[3] * quat[2] + quat[0] * quat[1]),
                        1.0 - 2.0 * (quat[1]**2 + quat[2]**2))
        measurement['orientation'] = yaw
    
    return measurement


# Testing function
if __name__ == "__main__":
    print("State Estimation module loaded successfully.")
    print("Particle Filter initialized.")
    
    # Simple test
    pf = ParticleFilter(num_particles=50, initial_state=(0, 0, 0))
    print(f"Initial estimate: {pf.get_estimate()}")
    
    # Simulate motion
    pf.predict(control_input=(0.5, 0.1))  # Move forward with slight turn
    print(f"After prediction: {pf.get_estimate()}")
    
    # Simulate measurement
    pf.update(sensor_measurement={'position': (0.1, 0.05), 'orientation': 0.05})
    print(f"After update: {pf.get_estimate()}")
    
    # Resample
    pf.resample()
    print(f"After resample: {pf.get_estimate()}")
    print(f"Variance: {pf.get_variance()}")

