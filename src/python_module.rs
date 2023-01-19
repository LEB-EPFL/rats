use std::ops::DerefMut;

use ndarray::ArrayView1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;

use crate::accumulators::StepUntil;
use crate::steppers::Stepper;
use crate::{Accumulate, StateMachineError, Transition};
use crate::{State, Step, Time};

#[pyclass(name = "StateMachine")]
pub struct PyStateMachine {
    accumulator: StepUntil<Stepper>,
}

#[pymethods]
impl PyStateMachine {
    #[new]
    fn new() -> Self {
        let stepper = Stepper::new(0, 10);
        let accumulator = StepUntil::new(stepper, 1.0);

        PyStateMachine { accumulator }
    }

    #[getter]
    fn current_state(&self) -> PyResult<State> {
        Ok(self.accumulator.stepper().current_state())
    }

    fn accumulate(
        &mut self,
        ctrl_params: PyReadonlyArray1<f64>,
    ) -> Result<Vec<PyTransition>, PyErr> {
        let ctrl_params = ctrl_params.as_slice()?;
        self.base_accumulate(ctrl_params)
    }

    fn step(&mut self, ctrl_params: PyReadonlyArray1<f64>) -> Result<PyTransition, PyErr> {
        let mut rng = rand::thread_rng();
        let transition = self
            .accumulator
            .stepper_mut()
            .step(ctrl_params.as_slice()?, &mut rng)?;

        Ok(PyTransition::from(transition))
    }
}

impl PyStateMachine {
    /// Runs the accumulate method state machine and collects all the transitions that occur.
    ///
    /// Arguments to this function may be sent to other threads.
    ///
    /// # Arguments
    /// - ctrl_params: The control parameters that determine the state machine's transition rates.
    ///
    fn base_accumulate(&mut self, ctrl_params: &[f64]) -> Result<Vec<PyTransition>, PyErr> {
        let mut rng = rand::thread_rng();
        let transitions: Vec<PyTransition> = self
            .accumulator
            .accumulate(ctrl_params, &mut rng)?
            .to_vec()
            .into_iter()
            .map(|item| PyTransition::from(item))
            .collect();

        Ok(transitions)
    }
}

#[derive(Clone, Debug)]
#[pyclass(frozen, name = "Transition")]
pub struct PyTransition {
    #[pyo3(get)]
    from_state: State,

    #[pyo3(get)]
    time: Time,

    #[pyo3(get)]
    to_state: State,
}

impl From<Transition> for PyTransition {
    fn from(item: Transition) -> Self {
        PyTransition {
            from_state: item.from,
            time: item.time,
            to_state: item.to,
        }
    }
}

#[pyfunction]
pub fn par_accumulate(
    machines: Vec<&PyCell<PyStateMachine>>,
    ctrl_params: Vec<PyReadonlyArray1<f64>>,
) -> PyResult<Vec<Vec<PyTransition>>> {
    let ctrl_params: Vec<&[f64]> = ctrl_params
        .iter()
        .map(|item| item.as_slice())
        .collect::<Result<Vec<&[f64]>, _>>()?;

    // I wanted to avoid copying the StateMachine objects that are owned by the Python
    // interpreter and instead mutate them directly; this is the only way I managed to do it.
    let mut machines = machines
        .into_iter()
        .map(|cell| cell.try_borrow_mut())
        .collect::<Result<Vec<PyRefMut<PyStateMachine>>, _>>()?;

    let mut machines = machines
        .iter_mut()
        .map(|refr| refr.deref_mut())
        .collect::<Vec<&mut PyStateMachine>>();

    Ok((machines.as_mut_slice(), ctrl_params.as_slice())
        .into_par_iter()
        .map(|item| item.0.base_accumulate(*item.1))
        .collect::<Result<Vec<Vec<PyTransition>>, _>>()?)
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
    m.add_class::<PyStateMachine>()?;
    m.add_class::<PyTransition>()?;
    m.add_function(wrap_pyfunction!(par_accumulate, m)?)?;
    Ok(())
}
