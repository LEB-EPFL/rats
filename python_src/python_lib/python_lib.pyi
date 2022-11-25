"""Stubs for extension functions and classes."""

from typing import Protocol

class StateMachine(Protocol):
    current_state: int
