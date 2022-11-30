//! Provides concrete implementations of the Step trait.
use ndarray::ArrayView1;
use rand::prelude::*;
use rand_distr::Exp;

use crate::{Result, State, StateMachineError, Step, Time, Transition};

/// A memoryless state machine that steps to a new random state at random times.
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
    pub fn current_state(&self) -> State {
        self.current_state
    }

    /// Returns the stepper's number of states.
    pub fn num_states(&self) -> State {
        self.num_states
    }
}

impl Step for Stepper {
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
            from: old_state,
            time: exp.sample(rng),
            to: new_state,
        })
    }
}

mod tests {
    #[cfg(test)]
    use ndarray::arr1;
    use rand::thread_rng;

    use super::Stepper;
    use crate::Step;

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
        assert_ne!(transition.from(), transition.to());
    }
}
