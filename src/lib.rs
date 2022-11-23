/// Rage against the State Machine (RATS)
///
/// A high performance probabilistic state machine simulator
///
use rand::prelude::*;

struct Event {
    from: u32,
    time: f64,
    to: u32,
}

struct StateMachine {
    current_state: u32,
    num_states: u32,
}

impl StateMachine {
    fn new(current_state: u32, num_states: u32) -> Self {
        StateMachine {
            current_state,
            num_states,
        }
    }

    fn step<R: rand::Rng + ?Sized>(&mut self, rng: &mut R) -> Event {
        let old_state = self.current_state;
        let mut new_state: u32;
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

mod python_module;
mod tests;
