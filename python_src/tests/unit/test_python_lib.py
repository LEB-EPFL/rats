from python_lib import StateMachine


def test_state_machine():
    sm = StateMachine()

    assert isinstance(sm.current_state, int)
