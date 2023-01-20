import numpy as np
from python_lib import StateMachine

NUM_TRANSITIONS = 10000  # Number of steps to perform
TOL_NUM_SIGMAS = 4  # A single assertion will fail ~ 1 / 16,000 times


def test_mean_transition_times():
    """Verifies that the mean transition times approximately equal their known expected values.

    Transition times should, on average, equal the reciprocal of the sum of a machine's rate
    constants to all accessible states from a given state.

    """
    rate_constants = np.array([[-1.0, 0.5, 1.0], [1.5, -1.0, 2.0], [2.5, 3.5, -1.0]])
    # (i, j) => average time to transition from state i to state j
    cases = {
        (0, 0): 0.0,
        (0, 1): 1 / (rate_constants[0, 1] + rate_constants[0, 2]),
        (0, 2): 1 / (rate_constants[0, 1] + rate_constants[0, 2]),
        (1, 0): 1 / (rate_constants[1, 0] + rate_constants[1, 2]),
        (1, 1): 0.0,
        (1, 2): 1 / (rate_constants[1, 0] + rate_constants[1, 2]),
        (2, 0): 1 / (rate_constants[2, 0] + rate_constants[2, 1]),
        (2, 1): 1 / (rate_constants[2, 0] + rate_constants[2, 1]),
        (2, 2): 0.0,
    }
    sm = StateMachine(0, rate_constants)

    ctrl_params = np.array([])
    transitions = [sm.step(ctrl_params) for _ in range(NUM_TRANSITIONS)]

    for case in cases:
        from_state, to_state = case
        filtered = [
            transition
            for transition in transitions
            if transition.from_state == from_state and transition.to_state == to_state
        ]

        if from_state == to_state:
            assert (
                len(filtered) == 0
            ), f"There should be no transitions from state {from_state} to state {to_state}"
        else:
            mean_transition_time = np.mean([transition.time for transition in filtered])
            std_transition_time = np.std([transition.time for transition in filtered])
            tolerance = TOL_NUM_SIGMAS * std_transition_time / np.sqrt(len(filtered))

            # Verify that the mean is within TOL_NUM_SIGMAS of the theoretical value
            assert (
                np.abs(mean_transition_time - cases[case]) < tolerance
            ), f"Mean transition time for {from_state, to_state} is out of tolerance: {TOL_NUM_SIGMAS} sigmas"


def test_relative_transition_probabilities():
    """Verifies that the relative transition probabilities are approximately correct."""
    rate_constants = np.array([[-1.0, 0.5, 1.0], [1.5, -1.0, 2.0], [2.5, 3.5, -1.0]])
    # (i, j) => probability to transition from state i to state j
    cases = {
        (0, 0): 0.0,
        (0, 1): rate_constants[0, 1] / (rate_constants[0, 1] + rate_constants[0, 2]),
        (0, 2): rate_constants[0, 2] / (rate_constants[0, 1] + rate_constants[0, 2]),
        (1, 0): rate_constants[1, 0] / (rate_constants[1, 0] + rate_constants[1, 2]),
        (1, 1): 0.0,
        (1, 2): rate_constants[1, 2] / (rate_constants[1, 0] + rate_constants[1, 2]),
        (2, 0): rate_constants[2, 0] / (rate_constants[2, 0] + rate_constants[2, 1]),
        (2, 1): rate_constants[2, 1] / (rate_constants[2, 0] + rate_constants[2, 1]),
        (2, 2): 0.0,
    }
    sm = StateMachine(0, rate_constants)

    ctrl_params = np.array([])
    transitions = [sm.step(ctrl_params) for _ in range(NUM_TRANSITIONS)]

    for case in cases:
        from_state, to_state = case
        filtered = [
            transition
            for transition in transitions
            if transition.from_state == from_state and transition.to_state == to_state
        ]

        if from_state == to_state:
            assert (
                len(filtered) == 0
            ), f"There should be no transitions from state {from_state} to state {to_state}"
        else:
            N = len(
                [
                    transition
                    for transition in transitions
                    if transition.from_state == from_state
                ]
            )

            # Each transition is a Bernoulli trial with p(success) = # of times the to_state was
            # reached divided by the total number of transitions from the from_state
            prob_ij = len(filtered) / N
            # std of a Bernoulli distribution is sqrt(p * (1-p)) where p is probability of success
            std_ij = np.sqrt(prob_ij * (1 - prob_ij))
            tolerance = TOL_NUM_SIGMAS * std_ij / np.sqrt(len(filtered))

            assert (
                np.abs(prob_ij - cases[case]) < tolerance
            ), f"Transition probability for {from_state, to_state} is out of tolerance: {tolerance} sigmas"
