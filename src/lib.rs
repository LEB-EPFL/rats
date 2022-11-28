//! High performance, probabilistic state machine simulator
//!
//! Rage against the State Machine (RATS) provides tools to simulate finite Markov random
//! processes as finite state machines. You can use RATS to simulate phenomena such as:
//! - fluorescence
//! - electronic transitions in atoms and molecules
//! - queues
//!
//! Importantly, RATS allows you to specify transition probabilities that depend on external
//! control parameters, such as the degree of laser irradiation incident on a flourophore.

use rand::prelude::*;
use rand_distr::Exp;
use rayon::prelude::*;

type State = u32;
type Time = f64;

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
    pub fn from(&self) -> State {
        self.from
    }

    pub fn time(&self) -> Time {
        self.time
    }

    pub fn to(&self) -> State {
        self.to
    }
}

/// State machines that may undergo a transition from one state to another.
///
/// `Step` types provide the logic for determining the transition probabilities from a state
/// machine's current state and actually transition the machine to a new state.
pub trait Step {
    /// Steps a state machine to a new state and returns information about the transition.
    fn step<R: rand::Rng + ?Sized>(&mut self, ctrl_param: f64, rng: &mut R) -> Transition;
}

/// A memoryless state machine that steps to a new random state at random times.
struct Stepper {
    current_state: State,
    num_states: State,
}

impl Stepper {
    fn new(current_state: State, num_states: State) -> Self {
        Stepper {
            current_state,
            num_states,
        }
    }
}

impl Step for Stepper {
    fn step<R: rand::Rng + ?Sized>(&mut self, ctrl_param: f64, rng: &mut R) -> Transition {
        let old_state = self.current_state;
        let mut new_state: State;
        loop {
            new_state = rng.gen_range(0..self.num_states - 1);
            if new_state != self.current_state {
                break;
            }
        }
        self.current_state = new_state;

        // TODO Handle Result properly
        let exp = Exp::new(ctrl_param).unwrap();

        Transition {
            from: old_state,
            time: exp.sample(rng),
            to: new_state,
        }
    }
}

/// Types that accumulate transitions from a `Step` type until a stop conditioned is reached.
pub trait Accumulate {
    fn accumulate<S: Step, R: rand::Rng + ?Sized>(
        &mut self,
        stepper: &mut S,
        ctrl_param: f64,
        rng: &mut R,
    ) -> &[Transition];
}

struct StepUntil {
    t_cutoff: Time,
    transition_buffer: Vec<Transition>,
}

impl StepUntil {
    fn new() -> Self {
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
        ctrl_param: f64,
        rng: &mut R,
    ) -> &[Transition] {
        self.transition_buffer.clear();

        let mut t_cumulative: Time = 0.0;
        let mut transition: Transition;
        loop {
            transition = stepper.step(ctrl_param, rng);

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

        self.transition_buffer.as_slice()
    }
}

/// Wraps a pair of `Step` and `Accumulate` types to enable customized state machines.
pub struct StateMachine<S: Step, A: Accumulate> {
    stepper: S,
    accumulator: A,
}

impl<S: Step, A: Accumulate> StateMachine<S, A> {
    pub fn new(stepper: S, accumulator: A) -> Self {
        StateMachine {
            stepper,
            accumulator,
        }
    }

    pub fn accumulate<R: rand::Rng + ?Sized>(
        &mut self,
        ctrl_param: f64,
        rng: &mut R,
    ) -> &[Transition] {
        self.accumulator
            .accumulate(&mut self.stepper, ctrl_param, rng)
    }

    pub fn step<R: rand::Rng + ?Sized>(&mut self, ctrl_param: f64, rng: &mut R) -> Transition {
        self.stepper.step(ctrl_param, rng)
    }
}

/// Accumulates transitions from a collection of state machines in parallel.
pub fn par_accumulate<S: Step + Send, A: Accumulate + Send>(
    state_machines: &mut [StateMachine<S, A>],
) -> Vec<Vec<Transition>> {
    // TODO Inject control parameter
    state_machines
        .par_iter_mut()
        .map_init(
            || rand::thread_rng(),
            |rng, sm| sm.accumulate(1.0, rng).to_vec(),
        )
        .collect::<Vec<Vec<Transition>>>()
}

mod python_module;
mod tests;
