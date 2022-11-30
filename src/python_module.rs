use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;

use crate::accumulators::StepUntil;
use crate::steppers::Stepper;
use crate::{Accumulate, StateMachineError, Transition as RustTransition};
use crate::{State, Step, Time};

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
        Ok(self.stepper.current_state())
    }

    fn accumulate(&mut self, ctrl_params: PyReadonlyArray1<f64>) -> Result<Vec<Transition>, PyErr> {
        let mut rng = rand::thread_rng();
        let transitions: Vec<Transition> = self
            .accumulator
            .accumulate(&mut self.stepper, ctrl_params.as_array(), &mut rng)?
            .to_vec()
            .into_iter()
            .map(|item| Transition::from(item))
            .collect();

        Ok(transitions)
    }

    fn step(&mut self, ctrl_params: PyReadonlyArray1<f64>) -> Result<Transition, PyErr> {
        let mut rng = rand::thread_rng();
        let rust_transition = self.stepper.step(ctrl_params.as_array(), &mut rng)?;

        Ok(Transition::from(rust_transition))
    }
}

#[derive(Clone)]
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

impl From<StateMachineError> for PyErr {
    fn from(err: StateMachineError) -> PyErr {
        match err {
            StateMachineError::NumElems {
                actual: _,
                expected: _,
            } => PyValueError::new_err(err.to_string()),
            StateMachineError::RngError(_) => PyValueError::new_err(err.to_string()),
        }
    }
}

/// The name of this function must match the `lib.name` setting in the `Cargo.toml`, else Python
/// will not be able to import the module.
#[pymodule]
fn python_lib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<StateMachine>()?;
    m.add_class::<Transition>()?;
    Ok(())
}
