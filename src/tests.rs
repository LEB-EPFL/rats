#[cfg(test)]
use ndarray::{arr1, ArrayView1};
use rand::prelude::*;

use crate::{par_accumulate, StateMachine, Step, StepUntil, Stepper};

#[test]
fn stepper_new() {
    let result = Stepper::new(0, 10);
    assert_eq!(result.current_state, 0);
}

#[test]
fn stepper_step() {
    let mut rng = rand::thread_rng();
    let mut sm = Stepper::new(0, 10);
    let ctrl_params = arr1(&[1.0]);
    let old_state = sm.current_state;

    let transition = sm.step(ctrl_params.view(), &mut rng).unwrap();

    assert_ne!(old_state, sm.current_state);
    assert_ne!(transition.from(), transition.to());
}

#[test]
fn par_accumulate_state_machines() {
    let n = 10;
    let mut machines: Vec<StateMachine<Stepper, StepUntil>> = Vec::with_capacity(n);
    let ctrl_params = arr1(&[1.0]);
    let mut ctrl_params_per_machine: Vec<ArrayView1<f64>> = Vec::with_capacity(n);
    for _ in 0..n {
        machines.push(StateMachine::new(Stepper::new(0, 10), StepUntil::new()));
        ctrl_params_per_machine.push(ctrl_params.view());
    }

    let results = par_accumulate(machines.as_mut_slice(), ctrl_params_per_machine.as_slice());

    assert_eq!(n, results.unwrap().len())
}
