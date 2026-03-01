import numpy as np
from typing import List, Tuple, Dict, Optional


class Particle:
    """Single particle representing a pose hypothesis."""

    def __init__(self, x: float, y: float, theta: float, weight: float = 1.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight


class ParticleFilter:
    """Monte-Carlo Localization (particle filter) for a differential-drive robot.

    Tuning notes (informed by EKF / PF theory):
    * motion_noise – σ added to (v, v, ω) during prediction.  Keep small
      when odometry is derived from wheel encoders (already noisy).
      Only ONE round of noise is injected (no double-diffusion).
    * measurement_noise – σ for the distance-based likelihood.  Smaller
      values make weight updates more discriminative → faster convergence
      but risk of particle depletion.  Balance with N_eff resampling.
    * mu – optional friction coefficient (à la EKF reference).  Models
      the ratio of actual velocity to commanded velocity.  Set to 0.0
      when feeding wheel-odometry (friction already baked in).
    """

    def __init__(
        self,
        num_particles: int,
        initial_pose: Tuple[float, float, float],
        map_landmarks: Dict[int, Tuple[float, float]],
        motion_noise: Tuple[float, float, float] = (0.01, 0.01, 0.02),
        measurement_noise: float = 0.03,
        mu: float = 0.0,
    ):
        self.num_particles = num_particles
        self.map_landmarks = map_landmarks
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        self.mu = mu  # friction coefficient (0 = trust odometry as-is)

        self.particles: List[Particle] = []
        for _ in range(num_particles):
            x = initial_pose[0] + np.random.normal(0, 0.05)
            y = initial_pose[1] + np.random.normal(0, 0.05)
            theta = initial_pose[2] + np.random.normal(0, 0.02)
            self.particles.append(Particle(x, y, theta, 1.0 / num_particles))

    # ── Prediction step ─────────────────────────────────────────────────
    def predict(self, v: float, omega: float, dt: float):
        """Propagate particles through the diff-drive motion model.

        A single round of Gaussian noise is injected into (v, ω).
        Optionally applies friction damping (mu) before integration.
        """
        for p in self.particles:
            # Add velocity-level noise (one round only)
            v_noisy = v + np.random.normal(0, self.motion_noise[0])
            omega_noisy = omega + np.random.normal(0, self.motion_noise[2])

            # Apply friction damping (like EKF motion model)
            v_eff = v_noisy * (1.0 - self.mu)
            omega_eff = omega_noisy * (1.0 - self.mu)

            # Integrate kinematics
            p.x += v_eff * dt * np.cos(p.theta)
            p.y += v_eff * dt * np.sin(p.theta)
            p.theta += omega_eff * dt

            # Wrap θ to [-π, π]
            p.theta = (p.theta + np.pi) % (2 * np.pi) - np.pi

    # ── Measurement update ──────────────────────────────────────────────
    def update(self, landmark_id: int, observed_rel_pos: Tuple[float, float]):
        """Weight particles by landmark observation likelihood.

        Uses a Gaussian over the Euclidean distance between the predicted
        and observed relative landmark positions.
        """
        if landmark_id not in self.map_landmarks:
            return

        landmark_true = self.map_landmarks[landmark_id]
        dx_obs, dy_obs = observed_rel_pos

        for p in self.particles:
            # Expected relative position of the landmark in robot frame
            dx_world = landmark_true[0] - p.x
            dy_world = landmark_true[1] - p.y

            cos_t = np.cos(p.theta)
            sin_t = np.sin(p.theta)
            dx_pred =  dx_world * cos_t + dy_world * sin_t
            dy_pred = -dx_world * sin_t + dy_world * cos_t

            # Per-axis Gaussian likelihood (more informative than 1-D distance)
            sigma = self.measurement_noise
            lx = np.exp(-0.5 * ((dx_pred - dx_obs) / sigma) ** 2)
            ly = np.exp(-0.5 * ((dy_pred - dy_obs) / sigma) ** 2)
            likelihood = lx * ly + 1e-300  # avoid zero

            p.weight *= likelihood

        # Normalise weights
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            for p in self.particles:
                p.weight /= total_weight
        else:
            for p in self.particles:
                p.weight = 1.0 / self.num_particles

    # ── Effective sample size ───────────────────────────────────────────
    def neff(self) -> float:
        """Return effective sample size  1 / Σwᵢ²."""
        weights = np.array([p.weight for p in self.particles])
        return 1.0 / np.sum(weights ** 2)

    # ── Resampling ──────────────────────────────────────────────────────
    def resample(self, force: bool = False, neff_threshold: float = 0.5):
        """Low-variance resampling.

        If *force* is True (default usage), always resample.
        Otherwise only when N_eff < threshold * N.
        A small jitter is added to resampled particles to maintain diversity.
        """
        # Optionally skip resampling if particle diversity is still healthy
        if not force and self.neff() > neff_threshold * self.num_particles:
            return

        new_particles: List[Particle] = []
        N = self.num_particles

        cum_weights = np.cumsum([p.weight for p in self.particles])

        step = 1.0 / N
        r = np.random.uniform(0, step)
        j = 0
        for i in range(N):
            u = r + i * step
            while u > cum_weights[j]:
                j += 1

            p = self.particles[j]
            # Jitter must be large enough for landmark updates to
            # discriminate between particles (σ >= measurement_noise/3).
            x = p.x + np.random.normal(0, 0.015)
            y = p.y + np.random.normal(0, 0.015)
            theta = p.theta + np.random.normal(0, 0.008)
            new_particles.append(Particle(x, y, theta, 1.0 / N))

        self.particles = new_particles

    # ── Pose estimate ───────────────────────────────────────────────────
    def get_estimate(self) -> Tuple[float, float, float]:
        """Weighted mean pose across all particles."""
        weights = [p.weight for p in self.particles]
        x_mean = np.average([p.x for p in self.particles], weights=weights)
        y_mean = np.average([p.y for p in self.particles], weights=weights)

        sin_sum = np.average([np.sin(p.theta) for p in self.particles], weights=weights)
        cos_sum = np.average([np.cos(p.theta) for p in self.particles], weights=weights)
        theta_mean = np.arctan2(sin_sum, cos_sum)
        return x_mean, y_mean, theta_mean

    def get_particles(self) -> List[Particle]:
        """Return current particle set."""
        return self.particles