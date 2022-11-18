/// Rage against the State Machine (RATS)
///
/// A high performance probabilistic state machine simulator
///

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
}

mod python_module;
mod tests;
