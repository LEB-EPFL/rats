"""Stubs for extension functions and classes."""
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

class StateMachine(Protocol):
    current_state: int

@dataclass(frozen=True)
class Transition(Protocol):
    from_state: int
    time: float
    to_state: int

def par_accumulate(
    machines: list[StateMachine], ctrl_params: npt.NDArray[np.float64]
) -> list[list[Transition]]: ...
