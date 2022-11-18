#[cfg(test)]
use crate::StateMachine;

#[test]
fn new_state_machine() {
    let result = StateMachine::new(0);
    assert_eq!(result.current_state, 0);
}
