import numpy as np
from python_lib import StateMachine, par_accumulate


def test_state_machine():
    rate_constants = np.array([[-1.0, 1.0], [1.0, -1.0]])
    sm = StateMachine(0, rate_constants)

    assert isinstance(sm.current_state, int)


def test_state_machine_step():
    rate_constants = np.array([[-1.0, 1.0], [1.0, -1.0]])
    starting_state = 0
    sm = StateMachine(starting_state, rate_constants)
    ctrl_params = np.array([1.0])

    transition = sm.step(ctrl_params)

    assert transition.from_state != transition.to_state
    assert transition.to_state != starting_state
    assert transition.time >= 0.0


def test_par_accumulate():
    num_machines = 10
    rate_constants = np.array([[-1.0, 1.0], [1.0, -1.0]])
    machines = [StateMachine(0, rate_constants) for _ in range(num_machines)]
    ctrl_params = [np.array([1.0]) for _ in range(num_machines)]

    transitions = par_accumulate(machines, ctrl_params)

    assert len(transitions) == num_machines
