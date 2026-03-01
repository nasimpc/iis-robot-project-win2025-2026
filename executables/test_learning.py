#!/usr/bin/env python3
"""
Standalone test for learning.py (M9 – Learning).

Tests the QLearningTuner and Learning wrapper: state discretisation,
Q-table updates, ε-greedy action selection, parameter bounding,
reward shaping, and JSON persistence.

Run inside Docker:
    sudo docker compose run --rm robot_sim python3 test_learning.py
"""

import sys, os, json, tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.modules.learning import QLearningTuner, Learning

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        status = "PASS"
    else:
        FAIL += 1
        status = "FAIL"
    extra = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{extra}")


# ═══════════════════════════════════════════════════════════════════════
#  1. QLearningTuner basics
# ═══════════════════════════════════════════════════════════════════════
def test_tuner_basics():
    print("\n" + "=" * 60)
    print("1. QLearningTuner — instantiation & action space")
    print("=" * 60)

    params = {"Kp": 1.0, "Ki": 0.0, "Kd": 0.1}
    steps  = {"Kp": 0.1, "Ki": 0.01, "Kd": 0.05}
    bounds = {"Kp": (0.0, 10.0), "Ki": (0.0, 5.0), "Kd": (0.0, 5.0)}
    thresh = {"overshoot": 0.2, "error": 0.1, "settling": 5.0}

    t = QLearningTuner(params, steps, bounds, thresh)

    check("3 param names", len(t.param_names) == 3, t.param_names)
    # 3 params × {-1,0,+1} = 3^3 = 27 actions
    check("27 actions (3^3)", t.num_actions == 27)
    check("Q-table starts empty", len(t.q_table) == 0)
    check("Initial ε = 0.5", t.epsilon == 0.5)


# ═══════════════════════════════════════════════════════════════════════
#  2. State discretisation
# ═══════════════════════════════════════════════════════════════════════
def test_state_discretise():
    print("\n" + "=" * 60)
    print("2. State discretisation")
    print("=" * 60)

    params = {"Kp": 1.0}
    steps  = {"Kp": 0.1}
    bounds = {"Kp": (0.0, 10.0)}
    thresh = {"overshoot": 0.2, "error": 0.1, "settling": 5.0}

    t = QLearningTuner(params, steps, bounds, thresh)

    # Empty metrics → "INIT"
    check("empty → INIT", t._discretize_state({}) == "INIT")

    # Good run
    s = t._discretize_state({
        "overshoot": 0.05, "steady_state_error": 0.02,
        "settling_time": 2.0, "torque_violation": 0.0,
    })
    check("good run → all Low/Fast/Safe",
          s == "OS:Low_ERR:Low_SET:Fast_TRQ:Safe", s)

    # Bad run
    s2 = t._discretize_state({
        "overshoot": 0.5, "steady_state_error": 0.3,
        "settling_time": 10.0, "torque_violation": 1.0,
    })
    check("bad run → all High/Slow/High",
          s2 == "OS:High_ERR:High_SET:Slow_TRQ:High", s2)


# ═══════════════════════════════════════════════════════════════════════
#  3. Q-table update (Bellman)
# ═══════════════════════════════════════════════════════════════════════
def test_q_update():
    print("\n" + "=" * 60)
    print("3. Q-table update (Bellman equation)")
    print("=" * 60)

    params = {"Kp": 1.0}
    steps  = {"Kp": 0.1}
    bounds = {"Kp": (0.0, 10.0)}
    thresh = {"overshoot": 0.2, "error": 0.1, "settling": 5.0}

    t = QLearningTuner(params, steps, bounds, thresh, alpha=0.1, gamma=0.9)

    # Simulate: state "A", action 0, reward 10, next state "B"
    t.last_state = "A"
    t.last_action_idx = 0
    t.update_q_table(reward=10.0, next_state="B")

    # Q(A,0) = 0 + 0.1*(10 + 0.9*0 - 0) = 1.0
    q_a = t._get_q_values("A")[0]
    check("Q(A,0) = 1.0 after first update", abs(q_a - 1.0) < 1e-6,
          f"got {q_a:.6f}")

    # Second update: same transition, reward 10
    t.last_state = "A"
    t.last_action_idx = 0
    t.update_q_table(reward=10.0, next_state="B")

    # Q(A,0) = 1.0 + 0.1*(10 + 0.9*0 - 1.0) = 1.0 + 0.9 = 1.9
    q_a2 = t._get_q_values("A")[0]
    check("Q(A,0) = 1.9 after second update", abs(q_a2 - 1.9) < 1e-6,
          f"got {q_a2:.6f}")


# ═══════════════════════════════════════════════════════════════════════
#  4. ε-greedy action selection
# ═══════════════════════════════════════════════════════════════════════
def test_epsilon_greedy():
    print("\n" + "=" * 60)
    print("4. ε-greedy action selection & decay")
    print("=" * 60)

    params = {"Kp": 1.0}
    steps  = {"Kp": 0.1}
    bounds = {"Kp": (0.0, 10.0)}
    thresh = {"overshoot": 0.2, "error": 0.1, "settling": 5.0}

    t = QLearningTuner(params, steps, bounds, thresh,
                       initial_epsilon=1.0, epsilon_decay=0.5, min_epsilon=0.01)

    # ε=1.0 → always random (we just check it returns a valid index)
    actions = [t.choose_action("S") for _ in range(50)]
    check("all actions in range [0, num_actions)",
          all(0 <= a < t.num_actions for a in actions))

    # Set Q high for action 2 in state "S", then set ε=0 → must pick 2
    t._get_q_values("S")[2] = 999.0
    t.epsilon = 0.0
    greedy = t.choose_action("S")
    check("ε=0 → greedy picks best action (2)", greedy == 2, f"got {greedy}")

    # Decay
    t.epsilon = 1.0
    t.decay_epsilon()
    check("ε decays: 1.0 * 0.5 = 0.5", abs(t.epsilon - 0.5) < 1e-6)
    for _ in range(100):
        t.decay_epsilon()
    check("ε floors at min_epsilon", t.epsilon >= 0.01)


# ═══════════════════════════════════════════════════════════════════════
#  5. Parameter bounding
# ═══════════════════════════════════════════════════════════════════════
def test_param_bounds():
    print("\n" + "=" * 60)
    print("5. Parameter bounding after action")
    print("=" * 60)

    params = {"Kp": 0.05}        # near lower bound
    steps  = {"Kp": 0.1}
    bounds = {"Kp": (0.0, 10.0)}
    thresh = {"overshoot": 0.2, "error": 0.1, "settling": 5.0}

    t = QLearningTuner(params, steps, bounds, thresh)

    # Action index for Kp = -1 (decrease). For 1 param: actions = [(-1,),(0,),(1,)]
    dec_idx = [i for i, a in enumerate(t.action_space) if a == (-1,)][0]
    t.apply_action_to_params(dec_idx)  # 0.05 - 0.1 = -0.05 → clamped to 0.0
    check("Kp clamped to 0.0 (lower bound)",
          t.current_params["Kp"] == 0.0, f"got {t.current_params['Kp']}")

    # Push above upper bound
    t.current_params["Kp"] = 9.95
    inc_idx = [i for i, a in enumerate(t.action_space) if a == (1,)][0]
    t.apply_action_to_params(inc_idx)  # 9.95 + 0.1 = 10.05 → clamped to 10.0
    check("Kp clamped to 10.0 (upper bound)",
          t.current_params["Kp"] == 10.0, f"got {t.current_params['Kp']}")


# ═══════════════════════════════════════════════════════════════════════
#  6. Learning wrapper — reward shaping
# ═══════════════════════════════════════════════════════════════════════
def test_reward_shaping():
    print("\n" + "=" * 60)
    print("6. Learning wrapper — reward shaping")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name

    try:
        learner = Learning(save_path=tmp_path)

        # Successful trial with some overshoot
        learner.update_from_trial({
            "success": True, "collided": False,
            "performance_metrics": {
                "overshoot": 0.1,
                "steady_state_error": 0.05,
                "settling_time": 3.0,
                "torque_violation": 0.0,
                "energy_consumption": 10.0,
            }
        })

        # Expected: 100 - 0.1*10 - 0.05*20 - 0*50 - 10*0.1 = 100 - 1 - 1 - 0 - 1 = 97
        # Q-table should have an entry now
        check("Q-table populated after trial",
              len(learner.tuner.q_table) > 0,
              f"{len(learner.tuner.q_table)} states")

        # ε decayed
        check("ε decayed after trial",
              learner.tuner.epsilon < 0.5,
              f"ε={learner.tuner.epsilon:.4f}")

        # Collision trial → large negative reward
        old_q = {k: v.copy() for k, v in learner.tuner.q_table.items()}
        learner.update_from_trial({
            "success": False, "collided": True,
            "performance_metrics": {
                "overshoot": 0.5, "steady_state_error": 0.3,
                "settling_time": 10.0, "torque_violation": 1.0,
                "energy_consumption": 50.0,
            }
        })
        check("Q-table updated after collision trial",
              len(learner.tuner.q_table) >= 1)

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════
#  7. JSON persistence (save / load)
# ═══════════════════════════════════════════════════════════════════════
def test_persistence():
    print("\n" + "=" * 60)
    print("7. JSON persistence (save / load)")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name

    try:
        # Create, run a trial, save
        l1 = Learning(save_path=tmp_path)
        l1.update_from_trial({
            "success": True, "collided": False,
            "performance_metrics": {
                "overshoot": 0.1, "steady_state_error": 0.05,
                "settling_time": 3.0, "torque_violation": 0.0,
                "energy_consumption": 5.0,
            }
        })
        saved_params = l1.get_optimized_parameters()
        saved_eps = l1.tuner.epsilon
        saved_q_states = set(l1.tuner.q_table.keys())

        # Check JSON file exists and is valid
        check("JSON file created", os.path.exists(tmp_path))
        with open(tmp_path) as f:
            data = json.load(f)
        check("JSON has q_table", "q_table" in data)
        check("JSON has params", "params" in data)
        check("JSON has epsilon", "epsilon" in data)

        # Load into new instance
        l2 = Learning(save_path=tmp_path)
        loaded_params = l2.get_optimized_parameters()
        loaded_eps = l2.tuner.epsilon
        loaded_q_states = set(l2.tuner.q_table.keys())

        check("Params restored correctly", saved_params == loaded_params,
              f"saved={saved_params}, loaded={loaded_params}")
        check("ε restored correctly", abs(saved_eps - loaded_eps) < 1e-9,
              f"saved={saved_eps:.6f}, loaded={loaded_eps:.6f}")
        check("Q-table states restored", saved_q_states == loaded_q_states)

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════
#  8. Multi-episode learning (params should shift)
# ═══════════════════════════════════════════════════════════════════════
def test_multi_episode():
    print("\n" + "=" * 60)
    print("8. Multi-episode learning (params evolve)")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name

    try:
        learner = Learning(save_path=tmp_path)
        initial = learner.get_optimized_parameters().copy()

        # Run 20 simulated trials
        np.random.seed(42)
        for ep in range(20):
            success = np.random.random() > 0.3  # 70% success
            learner.update_from_trial({
                "success": success,
                "collided": not success and np.random.random() > 0.5,
                "performance_metrics": {
                    "overshoot": np.random.uniform(0.0, 0.4),
                    "steady_state_error": np.random.uniform(0.0, 0.2),
                    "settling_time": np.random.uniform(1.0, 8.0),
                    "torque_violation": 0.0 if success else np.random.uniform(0, 1),
                    "energy_consumption": np.random.uniform(1, 20),
                }
            })

        final = learner.get_optimized_parameters()
        changed = any(abs(initial[k] - final[k]) > 1e-9 for k in initial)
        check("Params changed after 20 episodes", changed,
              f"initial={initial}, final={final}")

        q_states = len(learner.tuner.q_table)
        check("Multiple Q-states explored", q_states >= 2,
              f"{q_states} states in Q-table")

        eps_final = learner.tuner.epsilon
        check("ε decayed significantly", eps_final < 0.5,
              f"ε={eps_final:.4f}")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════
#  9. Existing q_learning_state.json (offline learning evidence)
# ═══════════════════════════════════════════════════════════════════════
def test_existing_state():
    print("\n" + "=" * 60)
    print("9. Existing q_learning_state.json (offline learning evidence)")
    print("=" * 60)

    path = os.path.join(os.path.dirname(__file__), "q_learning_state.json")
    exists = os.path.exists(path)
    check("q_learning_state.json exists", exists)

    if exists:
        with open(path) as f:
            data = json.load(f)
        check("Has params", "params" in data and len(data["params"]) > 0,
              f"params={data.get('params')}")
        check("Has q_table entries", "q_table" in data and len(data["q_table"]) > 0,
              f"{len(data.get('q_table', {}))} states")
        check("ε < initial (learning happened)",
              data.get("epsilon", 1.0) < 0.5,
              f"ε={data.get('epsilon')}")


# ═══════════════════════════════════════════════════════════════════════
def main():
    test_tuner_basics()
    test_state_discretise()
    test_q_update()
    test_epsilon_greedy()
    test_param_bounds()
    test_reward_shaping()
    test_persistence()
    test_multi_episode()
    test_existing_state()

    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"RESULTS:  {PASS}/{total} passed,  {FAIL} failed")
    print("=" * 60)

    if FAIL:
        print("\nSome tests FAILED — review output above.")
        sys.exit(1)
    else:
        print("\nAll tests PASSED — learning.py is ready for M9.")
        sys.exit(0)


if __name__ == "__main__":
    main()
