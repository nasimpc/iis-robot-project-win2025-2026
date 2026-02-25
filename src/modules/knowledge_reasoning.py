from pyswip import Prolog
from typing import Dict, List, Tuple, Any
import logging

class KnowledgeBase:
    """
    Prolog-based KB optimized for the specific task constraints:
    Navigating a 10m x 10m x 10m room to grasp a 0.5kg target on a 1.5m x 0.8m table.
    """

    def __init__(self):
        self.prolog = Prolog()
        self.logger = logging.getLogger("EnvironmentKB")

        # TASK-SPECIFIC CONSTANTS (SITUATEDNESS) 
        self.room_bounds = (10.0, 10.0, 10.0)
        self.table_dims = (1.5, 0.8)   # Length, Width in meters
        self.table_z = 0.625           # Table height in meters
        self.obstacle_mass = 10.0      # Both table and cubes are 10kg
        
        # TARGET CONSTANTS
        self.target_mass = 0.5         # Cylinder mass
        self.target_color = "red"
        self.target_shape = "cylinder"
        
        # ROBOT LIMITS (EMBODIEDNESS)
        self.max_payload = 2.0         # Can lift 0.5kg target, but not 10kg table
        self.max_reach_z = 0.85        # Can reach slightly above the 0.625m table
        self.max_torque = 50.0
        self.lever_arm = 0.5           # Assumed arm length
        self.gravity = 9.81

        self._initialize_rules()
        self._initialize_static_environment()

   
    # RULE DEFINITIONS

    def _initialize_rules(self):
        """Defines Prolog rules using the explicitly defined task constants."""
        
        #Robot constraints
        self.prolog.assertz(f"max_payload({self.max_payload})")
        self.prolog.assertz(f"max_reach_z({self.max_reach_z})")
        self.prolog.assertz(f"max_torque({self.max_torque})")

        #Fixed objects and Obstacles
        # Anything >= 10kg (Table + 5 Cubes) is automatically an obstacle
        self.prolog.assertz(f"is_fixed(Obj) :- mass(Obj, M), M >= {self.obstacle_mass}")
        self.prolog.assertz("is_obstacle(Obj) :- is_fixed(Obj)")

        #Embodied Feasibility
        self.prolog.assertz("can_lift(Obj) :- mass(Obj, M), max_payload(Max), M =< Max")
        self.prolog.assertz("within_reach(Obj) :- position(Obj, _, _, Z), max_reach_z(MaxZ), Z =< MaxZ")
        self.prolog.assertz(
            f"torque_feasible(Obj) :- mass(Obj, M), max_torque(MaxT), "
            f"RequiredT is M * {self.gravity} * {self.lever_arm}, RequiredT =< MaxT"
        )

        #Full grasp affordance
        self.prolog.assertz("is_graspable(Obj) :- can_lift(Obj), within_reach(Obj), torque_feasible(Obj), \\+ is_fixed(Obj)")

        # Spatial reasoning: Object on table surface 
        # Calculate bounding box dynamically (half-length and half-width)
        dx = self.table_dims[0] / 2.0
        dy = self.table_dims[1] / 2.0
        self.prolog.assertz(
            f"on_surface(Obj, Surface) :- position(Obj, X1, Y1, Z1), position(Surface, X2, Y2, Z2), "
            f"abs(X1 - X2) =< {dx}, abs(Y1 - Y2) =< {dy}, Z1 >= Z2"
        )

        # Room boundary constraint 
        rx, ry, rz = self.room_bounds
        self.prolog.assertz(
            f"inside_room(Obj) :- position(Obj, X, Y, Z), "
            f"X >= 0, X =< {rx}, Y >= 0, Y =< {ry}, Z >= 0, Z =< {rz}"
        )

    # STATIC ENVIRONMENT

    def _initialize_static_environment(self):
        """Registers the permanent boundaries of the room."""
        # Colors and friction coefficients per task specs
        self.prolog.assertz("surface(floor, [0.2, 0.2, 0.2], 0.5)")
        self.prolog.assertz("surface(walls, [0.8, 0.8, 0.8], unknown)")
        self.prolog.assertz("surface(ceiling, [1.0, 1.0, 1.0], unknown)")

    # LOAD INITIAL MAP
    def load_initial_map(self, scene_config: Dict[str, Dict[str, Any]]):
        """
        Loads randomized scene configuration (Table and 5 Cubes).
        Target pose is intentionally excluded until visually perceived.
        """
        self.prolog.retractall("position(_,_,_,_)")
        self.prolog.retractall("mass(_,_)")
        self.prolog.retractall("color(_,_)")
        self.prolog.retractall("shape(_,_)")

        for obj_id, props in scene_config.items():
            if obj_id == "target":
                continue  # Target is excluded

            x, y, z = props.get("pos", (0.0, 0.0, 0.0))
            self.prolog.assertz(f"position({obj_id}, {x}, {y}, {z})")
            
            if "mass" in props:
                self.prolog.assertz(f"mass({obj_id}, {props['mass']})")
            if "color" in props:
                self.prolog.assertz(f"color({obj_id}, {props['color']})")
            if "shape" in props:
                self.prolog.assertz(f"shape({obj_id}, {props['shape']})")

    
    # TARGET PERCEPTION (DYNAMIC UPDATE)
    def perceive_target(self, x: float, y: float, z: float):
        """Updates the KB once the robot visually locates the target."""
        self.prolog.retractall("position(target,_,_,_)")
        self.prolog.retractall("mass(target,_)")
        self.prolog.retractall("color(target,_)")
        self.prolog.retractall("shape(target,_)")

        # Inject position + hardcoded task constants for the target
        self.prolog.assertz(f"position(target, {x}, {y}, {z})")
        self.prolog.assertz(f"mass(target, {self.target_mass})")
        self.prolog.assertz(f"color(target, {self.target_color})")
        self.prolog.assertz(f"shape(target, {self.target_shape})")
  
    # QUERIES
    def get_navigation_map(self) -> List[Dict[str, float]]:
        """Returns a list of all obstacles to be used by a path planner."""
        query = "is_obstacle(Obj), position(Obj, X, Y, Z)"
        try:
            results = list(self.prolog.query(query))
            return [{"id": r["Obj"], "x": float(r["X"]), "y": float(r["Y"])} for r in results]
        except Exception as e:
            self.logger.error(f"Error querying navigation map: {e}")
            return []

    def verify_grasp_conditions(self) -> Tuple[bool, str]:
        """
        Validates all Embodied and Situated constraints before attempting a grasp.
        Returns (is_feasible, reasoning_string).
        """
        # 1. Target perceived?
        if not bool(list(self.prolog.query("position(target,_,_,_)"))):
            return False, "Target pose unknown. Exploration required."

        # 2. Inside room?
        if not bool(list(self.prolog.query("inside_room(target)"))):
            return False, "Target is outside room boundaries."

        # 3. On table?
        if not bool(list(self.prolog.query("on_surface(target, table)"))):
            return False, "Target is not positioned on the table surface."

        # 4. Affordance check (embodied constraints)
        if not bool(list(self.prolog.query("is_graspable(target)"))):
            return False, "Embodied constraints violated (payload, reach, or torque)."

        return True, "All embodied and situated constraints satisfied. Ready to grasp."