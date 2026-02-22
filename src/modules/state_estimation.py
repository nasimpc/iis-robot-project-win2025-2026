"""
Module 5: State Estimation

Particle Filter implementation for robot pose estimation.
Estimates (x, y, theta) using probabilistic particles.
"""

import pybullet as p
import numpy as np

# Helper function to convert quaternion to yaw angle
def quaternion_to_yaw(q):
    x, y, z, w = q
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


class ParticleFilterEstimator:

    # Number of particles in the filter
    def __init__(self, num_particles=200):
        self.num_particles = num_particles
        
        # Particles: [x, y, theta]
        self.particles = np.zeros((num_particles, 3))
        
        # Weights
        self.weights = np.ones(num_particles) / num_particles

        # Initialize randomly in environment
        self.init_particles()

    # Initialize particles randomly in the environment
    def init_particles(self):
        """
        Initialize particles randomly in the room.
        """
        # Room approx [-5,5] range
        self.particles[:, 0] = np.random.uniform(-5, 5, self.num_particles)
        self.particles[:, 1] = np.random.uniform(-5, 5, self.num_particles)
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, self.num_particles)

    # Motion update based on robot's movement (placeholder, extend with actual odometry)
    def motion_update(self, robot_id):

        pos, orn = p.getBasePositionAndOrientation(robot_id)
        theta = quaternion_to_yaw(orn)

        # Move particles toward robot motion
        noise_pos = np.random.normal(0, 0.02, (self.num_particles, 2))
        noise_theta = np.random.normal(0, 0.03, self.num_particles)

        self.particles[:, 0] += noise_pos[:, 0]
        self.particles[:, 1] += noise_pos[:, 1]
        self.particles[:, 2] += noise_theta

        # Keep particles inside environment
        self.particles[:, 0] = np.clip(self.particles[:, 0], -5, 5)
        self.particles[:, 1] = np.clip(self.particles[:, 1], -5, 5)

    # Sensor update based on proximity to real robot pose (placeholder, extend with perception)
    def sensor_update(self, robot_id):
        # TODO: replace with perception-based likelihood
        """
        Weight particles based on proximity to real robot pose.
        (Placeholder sensor model, extend later with perception)
        """
        real_pos, real_orn = p.getBasePositionAndOrientation(robot_id)
        real_theta = quaternion_to_yaw(real_orn)

        dx = self.particles[:, 0] - real_pos[0]
        dy = self.particles[:, 1] - real_pos[1]
        dtheta = self.particles[:, 2] - real_theta

        dist = np.sqrt(dx**2 + dy**2)

        # Gaussian likelihood
        self.weights = np.exp(-dist**2 / 0.1) + 1e-300
        self.weights /= np.sum(self.weights)

    # Resample particles based on weights
    def resample(self):

        indices = np.random.choice(
            self.num_particles,
            self.num_particles,
            p=self.weights
        )
        self.particles = self.particles[indices]

        # Inject random particles (kidnapped robot recovery)
        num_random = int(0.1 * self.num_particles)
        self.particles[:num_random, 0] = np.random.uniform(-5, 5, num_random)
        self.particles[:num_random, 1] = np.random.uniform(-5, 5, num_random)
        self.particles[:num_random, 2] = np.random.uniform(-np.pi, np.pi, num_random)

        self.weights.fill(1.0 / self.num_particles)

    # Estimate pose as mean of particles
    def estimate(self):
        """
        Compute estimated pose from particles.
        """
        x = np.mean(self.particles[:, 0])
        y = np.mean(self.particles[:, 1])
        theta = np.mean(self.particles[:, 2])
        return x, y, theta

    # Full update step
    def update(self, robot_id):
        """
        Full particle filter step.
        """
        self.motion_update(robot_id)
        self.sensor_update(robot_id)
        self.resample()
        return self.estimate()
