//! Provides concrete implementations of StateMachines that implement the Step trait.
use rand::prelude::*;
use rand_distr::Exp;

use crate::arrays::{power, tensordot, Array2D, Array4D};
use crate::{CtrlParam, Result, State, StateMachineError, Step, Time, Transition};

/// A memoryless state machine that steps to a new random state at random times.
pub struct Stepper {
    current_state: State,
    rate_constants: Array2D,
    rate_coefficients: Option<Array4D>,
    stopped: bool,
}

impl Stepper {
    pub fn new(current_state: State, rate_constants: Array2D) -> Self {
        // TODO Accept this as an input instead
        let rate_coefficients = None;

        Stepper {
            current_state,
            rate_constants,
            rate_coefficients,
            stopped: false,
        }
    }

    /// Returns the stepper's number of states.
    pub fn num_states(&self) -> State {
        self.rate_constants.shape.0
    }

    /// Compute the rate coefficients subject to the given control parameters.
    ///
    /// Panics if order is greater than 255.
    fn compute_rates(&self, ctrl_params: &[CtrlParam]) -> Array2D {
        if let Some(rate_coefficients) = &self.rate_coefficients {
            // Order is by definition the size of the second dimension of the rate coefficients array
            let order = rate_coefficients.shape.1;

            let powers = power(ctrl_params, order.try_into().expect("order is too large"));
            tensordot(&powers, &rate_coefficients)
        } else {
            self.rate_constants.clone()
        }
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

        // Get the rate coefficients only for the current state
        let (_rows, cols) = self.rate_constants.shape;
        let ks = &self.rate_constants.data
            [(self.current_state * cols)..((self.current_state * cols) + cols)];

        // Draw exponential random numbers using the rate coefficients as the mean and keep the
        // smallest random number. The index of the corresponding rate coefficient is the next
        // state.
        let mut exp: Exp<Time>;
        let mut rn: Time;
        let mut new_state: State = self.current_state; // Initialization needed because the compiler can't tell when the machine is stopped
        let mut transition_time: Time = f64::INFINITY;
        for (state, rate) in ks.iter().enumerate() {
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
        if self.rate_constants.data
            [(self.current_state * cols)..((self.current_state * cols) + cols)]
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
    use crate::arrays::Array2D;
    use crate::{Rate, Step};

    #[test]
    fn stepper_new() {
        let current_state = 0;
        let rate_constants = Array2D {
            data: vec![-1.0, 1.0, 1.0, -1.0],
            shape: (2, 2),
        };

        let result = Stepper::new(
            current_state,
            Array2D {
                data: vec![-1.0, 1.0, 1.0, -1.0],
                shape: (2, 2),
            },
        );

        assert_eq!(current_state, result.current_state());
        for elements in zip(
            result.rate_constants.data.iter(),
            rate_constants.data.iter(),
        ) {
            assert!((elements.0 - elements.1).abs() < 0.000001)
        }
    }

    #[test]
    fn stepper_step() {
        let mut rng = rand::thread_rng();
        let rate_constants = Array2D {
            data: vec![-1.0, 1.0, 1.0, -1.0],
            shape: (2, 2),
        };

        let mut sm = Stepper::new(0, rate_constants);
        let ctrl_params = vec![1.0];
        let old_state = sm.current_state();

        let transition = sm.step(ctrl_params.as_slice(), &mut rng).unwrap();

        assert_ne!(old_state, sm.current_state());
        assert_ne!(transition.from(), transition.to());
    }
}
