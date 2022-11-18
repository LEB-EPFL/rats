#[cfg(test)]
use rand::prelude::*;

use crate::StateMachine;

#[test]
fn new_state_machine() {
    let result = StateMachine::new(0, 10);
    assert_eq!(result.current_state, 0);
}

#[test]
fn step_state_machine() {
    let mut rng = rand::thread_rng();
    let mut sm = StateMachine::new(0, 10);
    let old_state = sm.current_state;

    sm.step(&mut rng);

    assert_ne!(old_state, sm.current_state);
}
