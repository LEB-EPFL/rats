//! Provides concrete implementations of StateMachines that implement the Step trait.
use std::f64::INFINITY;

use rand::prelude::*;
use rand_distr::Exp;

use crate::{CtrlParam, Rate, Result, State, StateMachineError, Step, Time, Transition};

/// A memoryless state machine that steps to a new random state at random times.
pub struct Stepper {
    current_state: State,
    rate_constants: Vec<Vec<Rate>>,
    stopped: bool,
}

impl Stepper {
    pub fn new(current_state: State, rate_constants: Vec<Vec<Rate>>) -> Self {
        Stepper {
            current_state,
            rate_constants,
            stopped: false,
        }
    }

    /// Returns the stepper's number of states.
    pub fn num_states(&self) -> State {
        self.rate_constants.len()
    }
}

impl Step for Stepper {
    /// Returns the stepper's current state.
    fn current_state(&self) -> State {
        self.current_state
    }

    fn step<R: rand::Rng + ?Sized>(
        &mut self,
        ctrl_params: &[CtrlParam],
        rng: &mut R,
    ) -> Result<Transition> {
        if self.stopped {
            return Err(StateMachineError::Stopped);
        }

        // Draw exponential random numbers using the rate coefficients as the mean and keep the
        // smallest random number. The index of the corresponding rate coefficient is the next
        // state.
        let ks = &self.rate_constants[self.current_state];
        let mut exp: Exp<Time>;
        let mut rn: Time;
        let mut new_state: State = self.current_state; // Initialization needed because the compiler can't tell when the machine is stopped
        let mut transition_time: Time = INFINITY;
        for (state, rate) in ks.into_iter().enumerate() {
            // Negative rate => No transition possible to the corresponding state
            if *rate < 0.0 {
                continue;
            }

            exp = Exp::new(*rate)?;
            rn = exp.sample(rng);

            // The smallest random number determines the transition time and the next state
            if rn < transition_time {
                new_state = state;
                transition_time = rn;
            }
        }

        let old_state = self.current_state;
        self.current_state = new_state;

        // The stepper is stopped when all its rate coefficients out of its current state are < 0
        if self.rate_constants[self.current_state]
            .iter()
            .all(|&rate| rate < 0.0)
        {
            self.stopped = true;
        }

        Ok(Transition {
            from: old_state,
            time: transition_time,
            to: new_state,
        })
    }
}

mod tests {
    #[cfg(test)]
    use std::iter::zip;

    use rand::thread_rng;

    use super::Stepper;
    use crate::{Rate, Step};

    #[test]
    fn stepper_new() {
        let current_state = 0;
        let rate_constants: Vec<Vec<Rate>> = vec![vec![-1.0, 1.0], vec![1.0, -1.0]];

        let result = Stepper::new(current_state, rate_constants.clone());

        assert_eq!(current_state, result.current_state());
        for elements in zip(
            result.rate_constants.iter().flatten(),
            rate_constants.iter().flatten(),
        ) {
            assert!((elements.0 - elements.1).abs() < 0.000001)
        }
    }

    #[test]
    fn stepper_step() {
        let mut rng = rand::thread_rng();
        let rate_constants: Vec<Vec<Rate>> = vec![vec![-1.0, 1.0], vec![1.0, -1.0]];

        let mut sm = Stepper::new(0, rate_constants);
        let ctrl_params = vec![1.0];
        let old_state = sm.current_state();

        let transition = sm.step(ctrl_params.as_slice(), &mut rng).unwrap();

        assert_ne!(old_state, sm.current_state());
        assert_ne!(transition.from(), transition.to());
    }
}
