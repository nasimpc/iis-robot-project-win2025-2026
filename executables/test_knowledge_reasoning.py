#!/usr/bin/env python3
"""
Standalone test for knowledge_reasoning.py (M8 – Knowledge Representation).

Tests every public method of KnowledgeBase and verifies that Prolog facts,
rules, and affordance queries work correctly.

Run inside Docker:
    sudo docker compose run --rm robot_sim python3 test_knowledge_reasoning.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.modules.knowledge_reasoning import KnowledgeBase

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    extra = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{extra}")


def main():
    global PASS, FAIL

    # ── 1. Instantiation ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1. KnowledgeBase instantiation")
    print("=" * 60)
    kb = KnowledgeBase()
    check("KB created", kb is not None)
    check("Prolog engine exists", kb.prolog is not None)

    # Verify built-in robot capability facts
    res = list(kb.prolog.query("max_payload(X)"))
    check("max_payload asserted", len(res) == 1 and float(res[0]["X"]) == 2.0,
          f"got {res}")
    res = list(kb.prolog.query("max_reach_z(X)"))
    check("max_reach_z asserted", len(res) == 1 and float(res[0]["X"]) == 0.85,
          f"got {res}")
    res = list(kb.prolog.query("max_torque(X)"))
    check("max_torque asserted", len(res) == 1 and float(res[0]["X"]) == 50.0,
          f"got {res}")

    # ── 2. load_initial_map ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. load_initial_map  (obstacles + table, NO target)")
    print("=" * 60)

    scene = {
        "table":    {"pos": (2.5, 2.0, 0.625), "mass": 10.0,
                     "color": "brown", "shape": "box"},
        "obs_blue": {"pos": (-1.0, -3.0, 0.0), "mass": 10.0,
                     "color": "blue", "shape": "cube"},
        "obs_pink": {"pos": (-2.5, 2.9, 0.0), "mass": 10.0,
                     "color": "pink", "shape": "cube"},
        "obs_orange": {"pos": (-0.8, -0.6, 0.0), "mass": 10.0,
                       "color": "orange", "shape": "cube"},
        "obs_yellow": {"pos": (2.5, 3.3, 0.0), "mass": 10.0,
                       "color": "yellow", "shape": "cube"},
        "obs_black": {"pos": (-2.6, 3.9, 0.0), "mass": 10.0,
                      "color": "black", "shape": "cube"},
        # target must be SKIPPED by load_initial_map
        "target":   {"pos": (2.55, 1.90, 0.71), "mass": 0.5,
                     "color": "red", "shape": "cylinder"},
    }

    kb.load_initial_map(scene)

    # Obstacles loaded?
    res_pos = list(kb.prolog.query("position(obs_blue, X, Y, Z)"))
    check("obs_blue position loaded", len(res_pos) == 1,
          f"got {res_pos}")

    # All 5 obstacles + table = 6 position facts (target excluded)
    all_pos = list(kb.prolog.query("position(Obj, X, Y, Z)"))
    obj_names = [r["Obj"] for r in all_pos]
    check("6 objects loaded (no target)", len(all_pos) == 6,
          f"loaded: {obj_names}")
    check("target NOT in KB yet", "target" not in obj_names)

    # Color facts
    res_color = list(kb.prolog.query("color(obs_blue, C)"))
    check("obs_blue color = blue", len(res_color) == 1 and res_color[0]["C"] == "blue",
          f"got {res_color}")

    # ── 3. is_fixed / is_obstacle rules ─────────────────────────────────
    print("\n" + "=" * 60)
    print("3. is_fixed / is_obstacle  (mass >= 10 → fixed)")
    print("=" * 60)

    res_fixed = list(kb.prolog.query("is_fixed(obs_blue)"))
    check("obs_blue is_fixed (mass=10)", len(res_fixed) > 0)

    res_fixed_table = list(kb.prolog.query("is_fixed(table)"))
    check("table is_fixed (mass=10)", len(res_fixed_table) > 0)

    res_obst = list(kb.prolog.query("is_obstacle(obs_orange)"))
    check("obs_orange is_obstacle", len(res_obst) > 0)

    # ── 4. get_navigation_map ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. get_navigation_map  (returns obstacle positions)")
    print("=" * 60)

    nav_map = kb.get_navigation_map()
    check("nav map returns obstacles", len(nav_map) >= 5,
          f"got {len(nav_map)} obstacles")
    ids_in_map = [o["id"] for o in nav_map]
    check("obs_blue in nav map", "obs_blue" in ids_in_map)
    check("table in nav map (is_fixed)", "table" in ids_in_map)

    # ── 5. verify_grasp_conditions BEFORE perceiving target ─────────────
    print("\n" + "=" * 60)
    print("5. verify_grasp_conditions BEFORE target perceived")
    print("=" * 60)

    ok, msg = kb.verify_grasp_conditions()
    check("grasp denied (no target)", ok is False, msg)

    # ── 6. perceive_target ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("6. perceive_target  (add target to KB after perception)")
    print("=" * 60)

    kb.perceive_target(2.55, 1.90, 0.71)

    res_tgt = list(kb.prolog.query("position(target, X, Y, Z)"))
    check("target position asserted", len(res_tgt) == 1,
          f"pos={res_tgt}")

    res_col = list(kb.prolog.query("color(target, C)"))
    check("color(target, red)", len(res_col) == 1 and res_col[0]["C"] == "red",
          f"got {res_col}")

    res_shp = list(kb.prolog.query("shape(target, S)"))
    check("shape(target, cylinder)", len(res_shp) == 1 and res_shp[0]["S"] == "cylinder",
          f"got {res_shp}")

    res_mass = list(kb.prolog.query("mass(target, M)"))
    check("mass(target, 0.5)", len(res_mass) == 1 and float(res_mass[0]["M"]) == 0.5,
          f"got {res_mass}")

    # ── 7. Affordance reasoning ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("7. Affordance reasoning  (can_lift, within_reach, etc.)")
    print("=" * 60)

    # Target is lightweight
    res_lift = list(kb.prolog.query("can_lift(target)"))
    check("can_lift(target)  (0.5 <= 2.0)", len(res_lift) > 0)

    # Target z=0.71 is within reach (max_reach_z=0.85)
    res_reach = list(kb.prolog.query("within_reach(target)"))
    check("within_reach(target) (0.71 <= 0.85)", len(res_reach) > 0)

    # Torque feasible
    res_torq = list(kb.prolog.query("torque_feasible(target)"))
    check("torque_feasible(target)", len(res_torq) > 0)

    # NOT fixed (mass < 10)
    res_not_fixed = list(kb.prolog.query("is_fixed(target)"))
    check("target NOT is_fixed (mass=0.5)", len(res_not_fixed) == 0)

    # Combined: is_graspable
    res_grasp = list(kb.prolog.query("is_graspable(target)"))
    check("is_graspable(target)", len(res_grasp) > 0)

    # Obstacle should NOT be graspable (fixed)
    res_obst_grasp = list(kb.prolog.query("is_graspable(obs_blue)"))
    check("obs_blue NOT graspable (is_fixed)", len(res_obst_grasp) == 0)

    # ── 8. Spatial reasoning ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("8. Spatial reasoning  (on_surface, inside_room)")
    print("=" * 60)

    res_on = list(kb.prolog.query("on_surface(target, table)"))
    check("on_surface(target, table)", len(res_on) > 0)

    res_room = list(kb.prolog.query("inside_room(target)"))
    check("inside_room(target)", len(res_room) > 0)

    # Object outside room should fail
    kb.prolog.assertz("position(far_obj, 20, 20, 0)")
    res_far = list(kb.prolog.query("inside_room(far_obj)"))
    check("far_obj NOT inside_room", len(res_far) == 0)
    kb.prolog.retractall("position(far_obj, _, _, _)")

    # ── 9. verify_grasp_conditions AFTER perceiving target ──────────────
    print("\n" + "=" * 60)
    print("9. verify_grasp_conditions AFTER target perceived")
    print("=" * 60)

    ok, msg = kb.verify_grasp_conditions()
    check("grasp approved", ok is True, msg)

    # ── 10. Edge case: target unreachable (z too high) ──────────────────
    print("\n" + "=" * 60)
    print("10. Edge case: target at z=1.0 (above max_reach_z=0.85)")
    print("=" * 60)

    kb.perceive_target(2.55, 1.90, 1.0)
    res_reach_high = list(kb.prolog.query("within_reach(target)"))
    check("target NOT within_reach at z=1.0", len(res_reach_high) == 0)

    ok2, msg2 = kb.verify_grasp_conditions()
    check("grasp denied (out of reach)", ok2 is False, msg2)

    # Restore normal target
    kb.perceive_target(2.55, 1.90, 0.71)

    # ══════════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"RESULTS:  {PASS}/{total} passed,  {FAIL} failed")
    print("=" * 60)

    if FAIL > 0:
        print("\nSome tests FAILED — review output above.")
        sys.exit(1)
    else:
        print("\nAll tests PASSED — knowledge_reasoning.py is ready.")
        sys.exit(0)


if __name__ == "__main__":
    main()
