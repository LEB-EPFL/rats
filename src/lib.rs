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
use rand::prelude::*;
use rand_distr::ExpError;
use rayon::prelude::*;

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

/// A transition of a state machine from one state to another.
///
/// Transitions can occur at any point in time, i.e. the time dimension is continuous.
#[derive(Clone, Debug)]
pub struct Transition {
    from: State,
    time: Time,
    to: State,
}

impl Transition {
    /// Returns the state from which the state machine transitioned
    pub fn from(&self) -> State {
        self.from
    }

    /// Returns the time at which the state machine transitioned
    pub fn time(&self) -> Time {
        self.time
    }

    /// Returns the state to which the state machine transitioned
    pub fn to(&self) -> State {
        self.to
    }
}

/// State machines that may undergo a transition from one state to another.
///
/// `Step` types provide the logic for determining the transition probabilities from a state
/// machine's current state and actually transition the machine to a new state.
///
/// # Arguments
///
/// - **ctrl_params** A 1D slice of zero or more control parameters that determine the transition
///   probabilities from the machine's current state to all the possible subsequent states
/// - **rng** A random number generator
pub trait Step {
    /// Returns the current state of the state machine.
    fn current_state(&self) -> State;

    /// Steps a state machine to a new state and returns information about the transition.
    fn step<R: rand::Rng + ?Sized>(
        &mut self,
        ctrl_params: &[f64],
        rng: &mut R,
    ) -> Result<Transition>;
}

/// Types that accumulate transitions from a `Step` type until a stop conditioned is reached.
pub trait Accumulate {
    fn accumulate<R: rand::Rng + ?Sized>(
        &mut self,
        ctrl_params: &[f64],
        rng: &mut R,
    ) -> Result<&[Transition]>;
}

/// Accumulates transitions from a collection of state machines in parallel.
pub fn par_accumulate<A: Accumulate + Send>(
    accumulators: &mut [A],
    ctrl_params: &[&[f64]],
) -> Result<Vec<Vec<Transition>>> {
    if accumulators.len() != ctrl_params.len() {
        return Err(StateMachineError::NumElems {
            actual: ctrl_params.len(),
            expected: accumulators.len(),
        });
    };

    // This creates an object of type MultiZip from the Rayon crate
    (accumulators, ctrl_params)
        .into_par_iter()
        .map_init(
            || rand::thread_rng(),
            |rng, item| Ok(item.0.accumulate(*item.1, rng)?.to_vec()),
        )
        .collect::<Result<Vec<Vec<Transition>>>>()
}

pub mod accumulators;
pub mod steppers;

mod python_module;

mod tests {
    #[cfg(test)]
    use super::par_accumulate;
    use crate::accumulators::StepUntil;
    use crate::steppers::Stepper;

    #[test]
    fn par_accumulate_state_machines() {
        let n = 10;
        let mut accumulators: Vec<StepUntil<Stepper>> = Vec::with_capacity(n);
        let ctrl_params = vec![1.0];
        let mut ctrl_params_per_machine: Vec<&[f64]> = Vec::with_capacity(n);
        for _ in 0..n {
            accumulators.push(StepUntil::new(Stepper::new(0, 10), 1.0));
            ctrl_params_per_machine.push(ctrl_params.as_slice());
        }

        let results = par_accumulate(
            accumulators.as_mut_slice(),
            ctrl_params_per_machine.as_slice(),
        );

        assert_eq!(n, results.unwrap().len())
    }
}
