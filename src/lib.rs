/// Rage against the State Machine (RATS)
///
/// A high performance probabilistic state machine simulator
///
use rand::prelude::*;
use rand_distr::Exp;

type State = u32;
type Time = f64;

#[derive(Clone)]
struct Transition {
    from: State,
    time: Time,
    to: State,
}

trait Step {
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

trait Accumulate {
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

struct StateMachine<S: Step, A: Accumulate> {
    stepper: S,
    accumulator: A,
}

impl<S: Step, A: Accumulate> StateMachine<S, A> {
    fn new(stepper: S, accumulator: A) -> Self {
        StateMachine {
            stepper,
            accumulator,
        }
    }

    fn accumulate<R: rand::Rng + ?Sized>(&mut self, ctrl_param: f64, rng: &mut R) -> &[Transition] {
        self.accumulator
            .accumulate(&mut self.stepper, ctrl_param, rng)
    }

    fn step<R: rand::Rng + ?Sized>(&mut self, ctrl_param: f64, rng: &mut R) -> Transition {
        self.stepper.step(ctrl_param, rng)
    }
}

mod python_module;
mod tests;
