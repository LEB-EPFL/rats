#[cfg(test)]
use rand::prelude::*;

use crate::{par_run, StateMachine, Step, StepUntil, Stepper};

#[test]
fn stepper_new() {
    let result = Stepper::new(0, 10);
    assert_eq!(result.current_state, 0);
}

#[test]
fn stepper_step() {
    let mut rng = rand::thread_rng();
    let mut sm = Stepper::new(0, 10);
    let ctrl_param = 1.0;
    let old_state = sm.current_state;

    let step = sm.step(ctrl_param, &mut rng);

    assert_ne!(old_state, sm.current_state);
    assert_ne!(step.from, step.to);
}

#[test]
fn par_run_state_machines() {
    let n = 10;
    let mut machines: Vec<StateMachine<Stepper, StepUntil>> = Vec::with_capacity(n);
    for _ in 0..n {
        machines.push(StateMachine::new(Stepper::new(0, 10), StepUntil::new()))
    }

    let results = par_run(machines.as_mut_slice());

    assert_eq!(n, results.len())
}
