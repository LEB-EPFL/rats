//! High performance, probabilistic state machine simulator
//!
//! Rage against the State Machine (RATS) provides tools to simulate finite Markov random
//! processes as finite state machines. You can use RATS to simulate phenomena such as:
//! - electronic transitions in atoms and molecules
//! - protein binding and unbinding kinetics
//! - queues
//!
//! Importantly, RATS allows you to specify transition probabilities that depend on external
//! control parameters, such as the degree of laser irradiation incident on a flourophore.
use ::thiserror::Error;
use ndarray::ArrayView1;
use pyo3::pyclass;
use rand::prelude::*;
use rand_distr::{Exp, ExpError};
use rayon::prelude::*;

use python_module::Transition;

type State = u32;
type Time = f64;

type Result<T> = std::result::Result<T, StateMachineError>;

/// Error type returned when a function or method fails.
#[derive(Debug, Error)]
pub enum StateMachineError {
    #[error("array has the wrong number of elements: expected {expected:?} element(s), but received {actual:?}")]
    NumElems { actual: usize, expected: usize },
    #[error(transparent)]
    RngError(#[from] ExpError),
}

#[pyclass]
#[derive(Clone)]
/// Memoryless state machines that may undergo a transition from one state to another.
///
/// These types provide the logic for determining the transition probabilities from a state
/// machine's current state and actually transition the machine to a new state.
pub struct Stepper {
    current_state: State,
    num_states: State,
}

impl Stepper {
    pub fn new(current_state: State, num_states: State) -> Self {
        Stepper {
            current_state,
            num_states,
        }
    }

    /// Returns the stepper's current state.
    fn current_state(&self) -> State {
        self.current_state
    }

    /// Returns the stepper's number of states.
    pub fn num_states(&self) -> State {
        self.num_states
    }

    /// 
    fn step<R: rand::Rng + ?Sized>(
        &mut self,
        ctrl_params: ArrayView1<Time>,
        rng: &mut R,
    ) -> Result<Transition> {
        if ctrl_params.len() != 1 {
            return Err(StateMachineError::NumElems {
                actual: ctrl_params.len(),
                expected: 1,
            });
        }

        let old_state = self.current_state;
        let mut new_state: State;
        loop {
            new_state = rng.gen_range(0..self.num_states - 1);
            if new_state != self.current_state {
                break;
            }
        }
        self.current_state = new_state;

        let exp = Exp::new(ctrl_params[0])?;

        Ok(Transition {
            from_state: old_state,
            time: exp.sample(rng),
            to_state: new_state,
        })
    }
}

mod python_module;

mod tests {
    #[cfg(test)]
    use ndarray::{arr1, ArrayView1};

    use crate::Stepper;

    #[test]
    fn stepper_new() {
        let current_state = 0;
        let num_states = 10;

        let result = Stepper::new(current_state, num_states);

        let states = 0..num_states;
        assert!(states.contains(&result.current_state()));
        assert_eq!(current_state, result.current_state());
        assert_eq!(num_states, result.num_states());
    }

    #[test]
    fn stepper_step() {
        let mut rng = rand::thread_rng();
        let mut sm = Stepper::new(0, 10);
        let ctrl_params = arr1(&[1.0]);
        let old_state = sm.current_state();

        let transition = sm.step(ctrl_params.view(), &mut rng).unwrap();

        assert_ne!(old_state, sm.current_state());
        assert_ne!(transition.from_state(), transition.to_state());
    }
}
