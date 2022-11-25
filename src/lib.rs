/// Rage against the State Machine (RATS)
///
/// A high performance probabilistic state machine simulator
///
use rand::prelude::*;

type State = u32;
type Time = f64;

struct Event {
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

    fn step<R: rand::Rng + ?Sized>(&mut self, rng: &mut R) -> Event {
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
        Event {
            from: old_state,
            time: 0.25,
            to: new_state,
        }
    }
}

struct Accumulator {
    t_cutoff: Time,
    event_buffer: Vec<Event>,
}

impl Accumulator {
    fn new() -> Self {
        let event_buffer = Vec::new();

        Accumulator {
            t_cutoff: 1.0,
            event_buffer,
        }
    }

    /// Steps a state machine until the cumulative sum of event times exceeds a given limit.
    fn accumulate<R: rand::Rng + ?Sized>(
        &mut self,
        stepper: &mut Stepper,
        rng: &mut R,
    ) -> &mut [Event] {
        self.event_buffer.clear();

        let mut t_cumulative: Time = 0.0;
        let mut event: Event;
        loop {
            event = stepper.step(rng);

            event.time += t_cumulative;
            if event.time > self.t_cutoff {
                // The state machine is assumed memoryless, so we don't need to save the event for
                // future calls to this function.
                break;
            } else {
                t_cumulative = event.time;
                self.event_buffer.push(event);
            }
        }

        self.event_buffer.as_mut_slice()
    }
}

mod python_module;
mod tests;
