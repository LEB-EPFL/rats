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

struct StateMachine {
    current_state: State,
    num_states: u32,
}

impl StateMachine {
    fn new(current_state: State, num_states: u32) -> Self {
        StateMachine { current_state, num_states }
    }

    fn step<R: rand::Rng + ?Sized>(&mut self, rng: &mut R) {
        let mut new_state: u32;
        loop {
            new_state = rng.gen_range(0..self.num_states - 1);
            if new_state != self.current_state {
                break
            }
        }
        self.current_state = new_state;
    }
}

mod python_module;
mod tests;
