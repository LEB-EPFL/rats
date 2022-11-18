/// Rage against the State Machine (RATS)
///
/// A high performance probabilistic state machine simulator
///
use rand::prelude::*;

type State = u32;

struct Event {
    from: State,
    time: f64,
    to: State,
}

struct StateMachine {
    current_state: State,
}

impl StateMachine {
    fn new(current_state: State) -> Self {
        StateMachine { current_state }
    }

    fn step<R: rand::Rng + ?Sized>(&mut self, rng: &mut R) {
        let new_state: u32 = rng.gen_range(0..10);
        self.current_state = new_state;
    }
}

mod python_module;
mod tests;
