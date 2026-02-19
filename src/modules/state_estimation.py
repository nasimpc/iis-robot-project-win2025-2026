"""
Module 5: State Estimation – Particle Filter
Estimates robot pose (x, y, theta) by fusing noisy odometry and landmark observations.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class Particle:
    """A single particle representing a hypothesis of the robot's pose."""
    def __init__(self, x: float, y: float, theta: float, weight: float = 1.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight


class ParticleFilter:
    """
    Particle filter for 2D robot pose estimation.

    Attributes:
        num_particles (int): Number of particles.
        particles (List[Particle]): List of particles.
        map_landmarks (Dict[int, Tuple[float, float]]): Known landmark positions
            {id: (x, y)}. IDs: 0 = table, 1..5 = obstacles.
        motion_noise (Tuple[float, float, float]): Noise std for (x, y, theta) in prediction.
        measurement_noise (float): Std deviation for landmark distance measurement.
    """
    def __init__(self,
                 num_particles: int,
                 initial_pose: Tuple[float, float, float],
                 map_landmarks: Dict[int, Tuple[float, float]],
                 motion_noise: Tuple[float, float, float] = (0.02, 0.02, 0.05),
                 measurement_noise: float = 0.1):
        """
        Args:
            num_particles: Number of particles.
            initial_pose: (x, y, theta) starting pose.
            map_landmarks: Dictionary of known landmarks {id: (x, y)}.
            motion_noise: (std_x, std_y, std_theta) for motion model.
            measurement_noise: Std dev for landmark distance measurement.
        """
        self.num_particles = num_particles
        self.map_landmarks = map_landmarks
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise

        # Initialize particles around the initial pose with small noise
        self.particles = []
        for _ in range(num_particles):
            x = initial_pose[0] + np.random.normal(0, 0.05)
            y = initial_pose[1] + np.random.normal(0, 0.05)
            theta = initial_pose[2] + np.random.normal(0, 0.05)
            self.particles.append(Particle(x, y, theta, 1.0 / num_particles))

    def predict(self, v: float, omega: float, dt: float):
        """
        Motion update: move each particle according to the control input (odometry).

        Args:
            v: Linear velocity (m/s) in robot's forward direction.
            omega: Angular velocity (rad/s).
            dt: Time step (s).
        """
        for p in self.particles:
            # Add noise to control inputs
            v_noisy = v + np.random.normal(0, self.motion_noise[0])
            omega_noisy = omega + np.random.normal(0, self.motion_noise[2])

            # Update pose using differential drive kinematics
            p.x += v_noisy * dt * np.cos(p.theta)
            p.y += v_noisy * dt * np.sin(p.theta)
            p.theta += omega_noisy * dt

            # Normalise angle
            p.theta = (p.theta + np.pi) % (2 * np.pi) - np.pi

            # Add extra pose noise
            p.x += np.random.normal(0, self.motion_noise[0] * dt)
            p.y += np.random.normal(0, self.motion_noise[1] * dt)
            p.theta += np.random.normal(0, self.motion_noise[2] * dt)
            p.theta = (p.theta + np.pi) % (2 * np.pi) - np.pi

    def update(self, landmark_id: int, observed_rel_pos: Tuple[float, float]):
        """
        Measurement update: adjust particle weights based on observed landmark.

        Args:
            landmark_id: ID of the observed landmark (must exist in map_landmarks).
            observed_rel_pos: (dx, dy) position of the landmark relative to robot
                              (in robot base frame: x forward, y left).
        """
        if landmark_id not in self.map_landmarks:
            return  # Unknown landmark, ignore

        landmark_true = self.map_landmarks[landmark_id]  # (x, y) in world
        dx_obs, dy_obs = observed_rel_pos

        for p in self.particles:
            # Predict what the landmark's relative position should be from this particle
            dx_world = landmark_true[0] - p.x
            dy_world = landmark_true[1] - p.y

            # Rotate into robot's frame (particle's heading)
            cos_t = np.cos(p.theta)
            sin_t = np.sin(p.theta)
            dx_pred = dx_world * cos_t + dy_world * sin_t   # forward
            dy_pred = -dx_world * sin_t + dy_world * cos_t  # left

            # Compute error (Euclidean distance)
            err = np.sqrt((dx_pred - dx_obs)**2 + (dy_pred - dy_obs)**2)

            # Gaussian likelihood
            likelihood = (1.0 / (np.sqrt(2 * np.pi) * self.measurement_noise)) * \
                         np.exp(-0.5 * (err / self.measurement_noise)**2)

            p.weight *= likelihood

        # Normalise weights
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            for p in self.particles:
                p.weight /= total_weight
        else:
            # If all weights zero, reset to uniform
            for p in self.particles:
                p.weight = 1.0 / self.num_particles

    def resample(self):
        """Systematic resampling to avoid particle depletion."""
        new_particles = []
        N = self.num_particles

        cum_weights = np.cumsum([p.weight for p in self.particles])
        step = 1.0 / N
        r = np.random.uniform(0, step)
        j = 0
        for i in range(N):
            u = r + i * step
            while u > cum_weights[j]:
                j += 1
            # Copy particle j with small jitter
            p = self.particles[j]
            x = p.x + np.random.normal(0, 0.01)
            y = p.y + np.random.normal(0, 0.01)
            theta = p.theta + np.random.normal(0, 0.01)
            new_particles.append(Particle(x, y, theta, 1.0 / N))

        self.particles = new_particles

    def get_estimate(self) -> Tuple[float, float, float]:
        """Return the weighted mean pose as the best estimate."""
        x_mean = np.average([p.x for p in self.particles], weights=[p.weight for p in self.particles])
        y_mean = np.average([p.y for p in self.particles], weights=[p.weight for p in self.particles])
        # Circular mean for angle
        sin_sum = np.average([np.sin(p.theta) for p in self.particles], weights=[p.weight for p in self.particles])
        cos_sum = np.average([np.cos(p.theta) for p in self.particles], weights=[p.weight for p in self.particles])
        theta_mean = np.arctan2(sin_sum, cos_sum)
        return x_mean, y_mean, theta_mean

    def get_particles(self) -> List[Particle]:
        """Return all particles (for debugging / visualisation)."""
        return self.particles