from python_lib import StateMachine


def test_state_machine():
    sm = StateMachine()

    assert isinstance(sm.current_state, int)


def test_state_machine_step():
    sm = StateMachine()

    transition = sm.step()
    assert transition.from_state != transition.to_state
