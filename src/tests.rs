#[cfg(test)]
use rand::prelude::*;

use crate::{Step, Stepper};

#[test]
fn new_state_machine() {
    let result = Stepper::new(0, 10);
    assert_eq!(result.current_state, 0);
}

#[test]
fn step_state_machine() {
    let mut rng = rand::thread_rng();
    let mut sm = Stepper::new(0, 10);
    let ctrl_param = 1.0;
    let old_state = sm.current_state;

    let step = sm.step(ctrl_param, &mut rng);

    assert_ne!(old_state, sm.current_state);
    assert_ne!(step.from, step.to);
}
