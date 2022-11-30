//! Provides concrete implementations of the Accumulate trait.
use ndarray::ArrayView1;

use crate::{Accumulate, Result, Step, Time, Transition};

pub struct StepUntil {
    t_cutoff: Time,
    transition_buffer: Vec<Transition>,
}

impl StepUntil {
    pub fn new() -> Self {
        let transition_buffer = Vec::new();

        // TODO Parameterize the cutoff
        StepUntil {
            t_cutoff: 1.0,
            transition_buffer,
        }
    }
}

impl Accumulate for StepUntil {
    /// Steps a state machine until the cumulative sum of transition times exceeds a given limit.
    fn accumulate<S: Step, R: rand::Rng + ?Sized>(
        &mut self,
        stepper: &mut S,
        ctrl_params: ArrayView1<f64>,
        rng: &mut R,
    ) -> Result<&[Transition]> {
        self.transition_buffer.clear();

        let mut t_cumulative: Time = 0.0;
        let mut transition: Transition;
        loop {
            transition = stepper.step(ctrl_params, rng)?;

            transition.time += t_cumulative;
            if transition.time > self.t_cutoff {
                // The state machine is assumed memoryless, so we don't need to save the transition for
                // future calls to this function.
                break;
            } else {
                t_cumulative = transition.time;
                self.transition_buffer.push(transition);
            }
        }

        Ok(self.transition_buffer.as_slice())
    }
}
