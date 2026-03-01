# Module 1: System Requirements

## 1. Embodiedness

The robot agent is a **mobile manipulator** with clearly defined physical constraints that must be respected during all operations.

### 1.1 Mobile Base

| Property | Value |
|----------|-------|
| Dimensions | 0.6m × 0.4m × 0.15m (L × W × H) |
| Mass | 20 kg |
| Locomotion | 4-wheel differential drive |
| Ground Clearance | 0.0m (wheels at base level) |

### 1.2 Wheels

| Property | Value |
|----------|-------|
| Count | 4 (FL, FR, BL, BR) |
| Radius | 0.1m |
| Width | 0.05m |
| Mass | 1 kg each |
| Joint Type | Continuous (unlimited rotation) |
| Damping | 0.1 |
| Friction | 0.1 |

### 1.3 Manipulator Arm

**7 Degrees of Freedom (7-DoF)** articulated arm mounted on the mobile base:

| Joint | Name | Axis | Range (rad) | Effort (Nm) | Velocity (rad/s) |
|-------|------|------|-------------|-------------|------------------|
| 1 | Shoulder Pan | Z | ±π | 100 | 2.0 |
| 2 | Shoulder Lift | Y | ±2.0 | 100 | 2.0 |
| 3 | Elbow | Y | ±2.5 | 80 | 2.5 |
| 4 | Forearm Rotation | Z | ±π | 60 | 3.0 |
| 5 | Wrist Bend | Y | ±2.0 | 40 | 3.0 |
| 6 | Wrist Rotation | Z | ±π | 20 | 4.0 |
| 7 | End Effector Mount | Y | ±2.0 | 10 | 4.0 |

**Arm Link Specifications:**

| Link | Radius (m) | Length (m) | Mass (kg) |
|------|------------|------------|-----------|
| Arm Base | 0.06 | 0.05 | 2.0 |
| Link 1 | 0.05 | 0.15 | 1.5 |
| Link 2 | 0.045 | 0.20 | 1.2 |
| Link 3 | 0.04 | 0.15 | 1.0 |
| Link 4 | 0.035 | 0.15 | 0.8 |
| Link 5 | 0.03 | 0.10 | 0.5 |
| Link 6 | 0.025 | 0.06 | 0.3 |
| Link 7 | 0.02 | 0.03 | 0.2 |

**Estimated Maximum Reach:** ~0.9m from arm base

### 1.4 End Effector (Gripper)

| Property | Value |
|----------|-------|
| Type | 2-finger parallel gripper |
| Gripper Base | 0.06m × 0.08m × 0.03m, 0.3 kg |
| Finger Size | 0.015m × 0.02m × 0.08m each |
| Finger Mass | 0.1 kg each |
| Max Aperture | 0.08m (±0.04m travel per finger) |
| Grip Force | 20 N |
| Grip Velocity | 0.5 m/s |
| Joint Type | Prismatic |

### 1.5 Sensors

| Sensor | Location | Dimensions | Purpose |
|--------|----------|------------|---------|
| **LIDAR** | Base (front-top) | Cylinder (r=0.03m, h=0.02m) | 2D laser scanning / obstacle detection |
| **IMU** | Base (top-center) | 0.02m × 0.02m × 0.01m | Orientation, acceleration |
| **RGB-D Camera** | End effector | 0.05m × 0.15m × 0.03m | Visual perception, depth sensing |

### 1.6 Total System Mass

| Component | Mass (kg) |
|-----------|-----------|
| Base | 20.0 |
| Wheels (4×) | 4.0 |
| Arm + Gripper | ~8.0 |
| Sensors | ~0.31 |
| **Total** | **~32.31 kg** |

---

## 2. Situatedness

The robot operates in a **controlled indoor environment** with randomized object placement at each execution.

### 2.1 Room Specifications

| Property | Value |
|----------|-------|
| Dimensions | 10m × 10m × 10m (cube) |
| Floor Color | Dark gray [0.2, 0.2, 0.2] |
| Floor Friction | μ = 0.5 |
| Wall Color | Light gray [0.8, 0.8, 0.8] |
| Ceiling Color | White [1.0, 1.0, 1.0] |

### 2.2 Table

| Property | Value |
|----------|-------|
| Surface Dimensions | 1.5m × 0.8m |
| Height (to surface) | 0.625m |
| Mass | 10 kg |
| Color | Brown [0.5, 0.3, 0.1] |
| Position | **Randomized** within (0.0 to 3.0, 0.0 to 3.0) |
| Mounting | Fixed (static) |

### 2.3 Target Object

| Property | Value |
|----------|-------|
| Shape | Cylinder |
| Radius | 0.04m |
| Height | 0.12m |
| Mass | 0.5 kg |
| Color | Red [1.0, 0.0, 0.0] |
| Position | **Randomized** on table surface |
| Mounting | Dynamic (can be grasped and moved) |

> **Important:** The target position is NOT provided in the initial knowledge base. The robot must perceive the target using its sensors.

### 2.4 Obstacles

| Property | Value |
|----------|-------|
| Count | 5 |
| Shape | Cube |
| Side Length | 0.4m |
| Mass | 10 kg each |
| Mounting | Fixed (static) |
| Positions | **Randomized** within (−4.0 to 4.0, −4.0 to 4.0) |
| Min Spacing | 1.0m between obstacles |

**Obstacle Colors:**

| # | Color | RGBA |
|---|-------|------|
| 1 | Blue | [0.0, 0.0, 1.0, 1.0] |
| 2 | Pink | [1.0, 0.4, 0.7, 1.0] |
| 3 | Orange | [1.0, 0.5, 0.0, 1.0] |
| 4 | Yellow | [1.0, 1.0, 0.0, 1.0] |
| 5 | Black | [0.1, 0.1, 0.1, 1.0] |

### 2.5 Robot Spawn Configuration

| Property | Value |
|----------|-------|
| Position | (−3.0, −3.0, 0.2) |
| Orientation | Default (facing +X direction) |
| Status | Fixed spawn point |

### 2.6 Scene Randomization

Each execution of `world_builder.py` generates a **unique scene configuration**:
- Table position randomized in the positive quadrant
- Target position randomized on the table
- All 5 obstacles randomly placed with collision avoidance
- Robot spawn position remains fixed

---

## 3. Task Constraints

### 3.1 Mission Objective

The robot must complete a **Navigate-to-Grasp** mission:

```
SEARCH → NAVIGATE → GRASP → LIFT
```

### 3.2 Operational Constraints

| Constraint | Description |
|------------|-------------|
| **Perception** | Target position must be perceived (not given a priori) |
| **Navigation** | Robot must reach table from spawn without collisions |
| **Manipulation** | Robot must use arm to grasp the target on the table |
| **Success Criteria** | Object lifted without any collisions |

### 3.3 Kinematic Constraints

| Constraint | Limit |
|------------|-------|
| Table Height | Arm must reach 0.625m + object height (~0.75m) |
| Object Size | Gripper aperture (0.08m) > object diameter (0.08m) ✓ |
| Object Mass | Gripper force (20N) > object weight (~5N) ✓ |

### 3.4 Information Constraints

| Information | Available in KB | Must Perceive |
|-------------|-----------------|---------------|
| Room layout | ✓ | — |
| Table position | ✓ | — |
| Obstacle positions | ✓ | — |
| Target position | ✗ | ✓ |
| Target properties | ✓ (color, size) | — |

---

## 4. Safety Boundaries

### 4.1 Workspace Boundaries

| Boundary | Limit | Margin |
|----------|-------|--------|
| X-axis | ±5.0m | 0.5m from walls |
| Y-axis | ±5.0m | 0.5m from walls |
| Z-axis | 0.0 to 10.0m | Floor and ceiling |
| **Safe Zone** | ±4.5m from center | |

### 4.2 Collision Avoidance

| Entity | Minimum Clearance |
|--------|-------------------|
| Walls | 0.5m |
| Obstacles | 0.3m (robot body) |
| Table | 0.1m (during approach) |
| Target | Gripper contact only |

### 4.3 Velocity Limits

| Component | Max Velocity | Safety Factor |
|-----------|--------------|---------------|
| Base (linear) | Based on wheel joints | Apply gradual acceleration |
| Base (angular) | Based on differential | Limit turning rate near obstacles |
| Arm joints | Per joint (2.0–4.0 rad/s) | Reduce when carrying payload |
| Gripper | 0.5 m/s | Slow approach for grasping |

### 4.4 Force/Torque Limits

| Component | Max Effort | Safety Behavior |
|-----------|------------|-----------------|
| Arm J1-J2 | 100 Nm | Emergency stop on overload |
| Arm J3 | 80 Nm | Torque monitoring |
| Arm J4-J5 | 40-60 Nm | Force feedback |
| Arm J6-J7 | 10-20 Nm | Compliant control |
| Gripper | 20 N | Grasp detection threshold |

### 4.5 Emergency Behaviors

| Condition | Response |
|-----------|----------|
| Unexpected collision | Immediate stop, back off |
| Target lost during grasp | Search |
| Path blocked | Re-plan navigation |

---

