"""Stubs for extension functions and classes."""
from dataclasses import dataclass
from typing import Protocol

class StateMachine(Protocol):
    current_state: int

@dataclass(frozen=True)
class Transition(Protocol):
    from_state: int
    time: float
    to_state: int
