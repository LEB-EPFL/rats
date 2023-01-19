//! Provides concrete implementations of the Accumulate trait.
use ndarray::ArrayView1;

use crate::{Accumulate, Result, Step, Time, Transition};

pub struct StepUntil<S: Step> {
    stepper: S,
    t_cutoff: Time,
    transition_buffer: Vec<Transition>,
}

impl<S: Step> StepUntil<S> {
    pub fn new(stepper: S, t_cutoff: Time) -> Self {
        let transition_buffer = Vec::new();

        StepUntil {
            stepper: stepper,
            t_cutoff: t_cutoff,
            transition_buffer,
        }
    }

    pub fn stepper(&self) -> &S {
        &self.stepper
    }

    pub fn stepper_mut(&mut self) -> &mut S {
        &mut self.stepper
    }
}

impl<S: Step> Accumulate for StepUntil<S> {
    /// Steps a state machine until the cumulative sum of transition times exceeds a given limit.
    fn accumulate<R: rand::Rng + ?Sized>(
        &mut self,
        ctrl_params: ArrayView1<f64>,
        rng: &mut R,
    ) -> Result<&[Transition]> {
        self.transition_buffer.clear();

        let mut t_cumulative: Time = 0.0;
        let mut transition: Transition;
        loop {
            transition = self.stepper.step(ctrl_params, rng)?;

            transition.time += t_cumulative;
            if transition.time > self.t_cutoff {
                // The state machine is assumed memoryless, so we don't need to save the transition
                // for future calls to this function.
                break;
            } else {
                t_cumulative = transition.time;
                self.transition_buffer.push(transition);
            }
        }

        Ok(self.transition_buffer.as_slice())
    }
}
