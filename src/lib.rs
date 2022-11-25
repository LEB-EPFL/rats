/// Rage against the State Machine (RATS)
///
/// A high performance probabilistic state machine simulator
///
use rand::prelude::*;

type State = u32;
type Time = f64;

struct Transition {
    from: State,
    time: Time,
    to: State,
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

    fn step<R: rand::Rng + ?Sized>(&mut self, rng: &mut R) -> Transition {
        let old_state = self.current_state;
        let mut new_state: State;
        loop {
            new_state = rng.gen_range(0..self.num_states - 1);
            if new_state != self.current_state {
                break;
            }
        }
        self.current_state = new_state;

        // TODO Compute the transition time
        Transition {
            from: old_state,
            time: 0.25,
            to: new_state,
        }
    }
}

struct Accumulator {
    t_cutoff: Time,
    transition_buffer: Vec<Transition>,
}

impl Accumulator {
    fn new() -> Self {
        let transition_buffer = Vec::new();

        Accumulator {
            t_cutoff: 1.0,
            transition_buffer,
        }
    }

    /// Steps a state machine until the cumulative sum of transition times exceeds a given limit.
    fn accumulate<R: rand::Rng + ?Sized>(
        &mut self,
        stepper: &mut Stepper,
        rng: &mut R,
    ) -> &mut [Transition] {
        self.transition_buffer.clear();

        let mut t_cumulative: Time = 0.0;
        let mut transition: Transition;
        loop {
            transition = stepper.step(rng);

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

        self.transition_buffer.as_mut_slice()
    }
}

mod python_module;
mod tests;
