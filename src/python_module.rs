use pyo3::prelude::*;
use rand::prelude::*;

use crate::Transition as RustTransition;
use crate::{State, Step, StepUntil, Stepper, Time};

#[pyclass]
struct StateMachine {
    stepper: Stepper,
    accumulator: StepUntil,
}

#[pymethods]
impl StateMachine {
    #[new]
    fn new() -> Self {
        let stepper = Stepper::new(0, 10);
        let accumulator = StepUntil::new();

        StateMachine {
            stepper,
            accumulator,
        }
    }

    #[getter]
    fn current_state(&self) -> PyResult<State> {
        Ok(self.stepper.current_state)
    }

    fn step(&mut self) -> PyResult<Transition> {
        let mut rng = rand::thread_rng();
        let rust_transition = self.stepper.step(&mut rng);

        Ok(Transition::from(rust_transition))
    }
}

#[pyclass(frozen)]
struct Transition {
    #[pyo3(get)]
    from_state: State,

    #[pyo3(get)]
    time: Time,

    #[pyo3(get)]
    to_state: State,
}

impl From<RustTransition> for Transition {
    fn from(item: RustTransition) -> Self {
        Transition {
            from_state: item.from,
            time: item.time,
            to_state: item.to,
        }
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn python_lib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<StateMachine>()?;
    m.add_class::<Transition>()?;
    Ok(())
}
