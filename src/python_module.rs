use pyo3::prelude::*;

use crate::{Accumulator, State, Stepper};

#[pyclass]
struct StateMachine {
    stepper: Stepper,
    accumulator: Accumulator,
}

#[pymethods]
impl StateMachine {
    #[new]
    fn new() -> Self {
        let stepper = Stepper::new(0, 10);
        let accumulator = Accumulator::new();

        StateMachine {
            stepper,
            accumulator,
        }
    }

    #[getter]
    fn current_state(&self) -> PyResult<State> {
        Ok(self.stepper.current_state)
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn python_lib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<StateMachine>()?;
    Ok(())
}
