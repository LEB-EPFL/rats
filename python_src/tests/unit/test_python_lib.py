import numpy as np
from python_lib import StateMachine, par_accumulate


def test_state_machine():
    sm = StateMachine()

    assert isinstance(sm.current_state, int)


def test_state_machine_step():
    sm = StateMachine()
    ctrl_params = np.array([1.0])

    transition = sm.step(ctrl_params)

    assert transition.from_state != transition.to_state


def test_par_accumulate():
    num_machines = 10
    machines = [StateMachine() for _ in range(num_machines)]
    ctrl_params = [np.array([1.0]) for _ in range(num_machines)]

    transitions = par_accumulate(machines, ctrl_params)

    assert len(transitions) == num_machines
