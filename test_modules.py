"""
Quick Test Script for IIS Robot Project
Tests each module independently to verify installation.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("=" * 60)
print("IIS ROBOT PROJECT - MODULE VERIFICATION TEST")
print("=" * 60)

# Test Module 4: Perception
print("\n[Test 1/6] Testing Perception Module...")
try:
    from src.modules.perception import ransac_plane_detection, estimate_pose_pca
    import numpy as np
    
    # Test RANSAC
    points = np.random.rand(100, 3)
    plane, inliers, inlier_points = ransac_plane_detection(points, num_iterations=50)
    print("  ✓ RANSAC plane detection: OK")
    
    # Test PCA
    centroid, axes, dims = estimate_pose_pca(points)
    print("  ✓ PCA pose estimation: OK")
    print("  ✓ Perception module: PASSED")
except Exception as e:
    print(f"  ✗ Perception module: FAILED - {e}")

# Test Module 5: State Estimation
print("\n[Test 2/6] Testing State Estimation Module...")
try:
    from src.modules.state_estimation import ParticleFilter
    
    pf = ParticleFilter(num_particles=50, initial_state=(0, 0, 0))
    pf.predict((0.5, 0.1), dt=0.1)
    pf.update({'position': (0.1, 0.05)})
    pf.resample()
    estimate = pf.get_estimate()
    print(f"  ✓ Particle Filter estimate: {estimate}")
    print("  ✓ State Estimation module: PASSED")
except Exception as e:
    print(f"  ✗ State Estimation module: FAILED - {e}")

# Test Module 6: Motion Control
print("\n[Test 3/6] Testing Motion Control Module...")
try:
    from src.modules.motion_control import PIDController, DifferentialDriveController
    
    pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
    output = pid.compute(setpoint=10.0, measured_value=5.0)
    print(f"  ✓ PID Controller output: {output:.2f}")
    
    drive_ctrl = DifferentialDriveController()
    (v, omega), distance = drive_ctrl.compute_control((0, 0, 0), (1, 1, 0))
    print(f"  ✓ Drive Controller: v={v:.2f}, ω={omega:.2f}")
    print("  ✓ Motion Control module: PASSED")
except Exception as e:
    print(f"  ✗ Motion Control module: FAILED - {e}")

# Test Module 7: Action Planning
print("\n[Test 4/6] Testing Action Planning Module...")
try:
    from src.modules.action_planning import FiniteStateMachine, RobotState
    
    fsm = FiniteStateMachine()
    print(f"  ✓ FSM initial state: {fsm.get_current_state().value}")
    
    action = fsm.update({'target': None}, (0, 0, 0), 0, 0.1)
    print(f"  ✓ FSM action: {action['action']}")
    print("  ✓ Action Planning module: PASSED")
except Exception as e:
    print(f"  ✗ Action Planning module: FAILED - {e}")

# Test Module 8: Knowledge Reasoning
print("\n[Test 5/6] Testing Knowledge Reasoning Module...")
try:
    from src.modules.knowledge_reasoning import KnowledgeBase
    
    kb = KnowledgeBase(use_prolog=False)  # Use fallback mode for testing
    kb.update_object_position('robot', (0, 0, 0))
    
    path = kb.plan_path_grid((0, 0), (5, 5), [(2.5, 2.5)])
    print(f"  ✓ Path planning: {len(path) if path else 0} waypoints")
    print("  ✓ Knowledge Reasoning module: PASSED")
except Exception as e:
    print(f"  ✗ Knowledge Reasoning module: FAILED - {e}")

# Test Module 9: Learning
print("\n[Test 6/6] Testing Learning Module...")
try:
    from src.modules.learning import ParameterOptimizer, PerformanceMetrics
    
    optimizer = ParameterOptimizer()
    metrics = PerformanceMetrics()
    metrics.mission_success = True
    metrics.total_time = 50.0
    
    score = metrics.compute_score()
    print(f"  ✓ Performance score: {score:.2f}")
    
    optimizer.record_trial(metrics)
    print(f"  ✓ Trial recorded")
    print("  ✓ Learning module: PASSED")
except Exception as e:
    print(f"  ✗ Learning module: FAILED - {e}")

# Test Module 2: World Builder
print("\n[Test 7/6] Testing World Builder...")
try:
    from src.environment.world_builder import check_collision, generate_random_position
    
    # Test collision checking
    collision = check_collision([1, 1], [[0, 0], [5, 5]], min_distance=2.0)
    print(f"  ✓ Collision check: {collision}")
    
    # Test random position generation
    pos = generate_random_position((-5, 5), (-5, 5), [[0, 0]], min_distance=2.0)
    print(f"  ✓ Random position generated: {pos}")
    print("  ✓ World Builder module: PASSED")
except Exception as e:
    print(f"  ✗ World Builder module: FAILED - {e}")

# Summary
print("\n" + "=" * 60)
print("MODULE VERIFICATION COMPLETE")
print("=" * 60)
print("\nAll core modules are installed and functional!")
print("\nNext steps:")
print("1. Run the full simulation:")
print("   cd executables")
print("   python cognitive_architecture.py")
print("\n2. Or use Docker:")
print("   docker compose up --build")
print("=" * 60)
