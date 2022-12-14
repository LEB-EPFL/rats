import numpy as np
from python_lib import StateMachine


def test_state_machine():
    sm = StateMachine()

    assert isinstance(sm.current_state, int)


def test_state_machine_step():
    sm = StateMachine()
    ctrl_params = np.array([1.0])

    transition = sm.step(ctrl_params)

    assert transition.from_state != transition.to_state
