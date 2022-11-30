use ndarray::ArrayView1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;

use crate::{State, StateMachineError, Stepper, Time};

#[pyclass(name = "StateMachine")]
#[derive(FromPyObject)]
pub struct PyAccumulator {
    stepper: Stepper,
    t_cutoff: Time,
    transition_buffer: Vec<Transition>,
}

impl PyAccumulator {
    /// Steps a state machine until the cumulative sum of transition times exceeds a given limit.
    fn rs_accumulate<R: rand::Rng + ?Sized>(
        &mut self,
        ctrl_params: ArrayView1<f64>,
        rng: &mut R,
    ) -> Result<&[Transition], StateMachineError> {
        self.transition_buffer.clear();

        let mut t_cumulative: Time = 0.0;
        let mut transition: Transition;
        loop {
            transition = self.stepper.step(ctrl_params, rng)?;

            transition.time += t_cumulative;
            if transition.time > self.t_cutoff {
                // The state machine is assumed memoryless, so we don't need to save the transition
                // for future calls to this function.
                break;
            } else {
                t_cumulative = transition.time;
                self.transition_buffer.push(transition);
            }
        }

        Ok(self.transition_buffer.as_slice())
    }
}

#[pymethods]
impl PyAccumulator {
    #[new]
    pub fn new(num_states: State, t_cutoff: Time) -> Self {
        let stepper = Stepper::new(0, num_states);
        let transition_buffer = Vec::new();

        PyAccumulator {
            stepper: stepper,
            t_cutoff: t_cutoff,
            transition_buffer,
        }
    }

    #[getter]
    fn current_state(&self) -> PyResult<State> {
        Ok(self.stepper.current_state())
    }

    fn accumulate(&mut self, ctrl_params: PyReadonlyArray1<f64>) -> Result<Vec<Transition>, PyErr> {
        let mut rng = rand::thread_rng();
        let transitions: Vec<Transition> = self
            .rs_accumulate(ctrl_params.as_array(), &mut rng)?
            .to_vec();

        Ok(transitions)
    }

    fn step(&mut self, ctrl_params: PyReadonlyArray1<f64>) -> Result<Transition, PyErr> {
        let mut rng = rand::thread_rng();
        let transition = self.stepper.step(ctrl_params.as_array(), &mut rng)?;

        Ok(transition)
    }
}

/// Accumulates transitions from a collection of state machines in parallel.
#[pyfunction]
pub fn par_accumulate(
    accumulators: Vec<&PyCell<PyAccumulator>>,
    ctrl_params: Vec<PyReadonlyArray1<f64>>,
) -> Result<Vec<Vec<Transition>>, PyErr> {
    if accumulators.len() != ctrl_params.len() {
        return Err(PyErr::from(StateMachineError::NumElems {
            actual: ctrl_params.len(),
            expected: accumulators.len(),
        }));
    };

    let ctrl_params: Vec<ArrayView1<f64>> = ctrl_params.iter().map(|p| p.as_array()).collect();
    // This creates an object of type MultiZip from the Rayon crate
    (accumulators, ctrl_params)
        .par_bridge()
        .map_init(
            || rand::thread_rng(),
            |rng, item| Ok(item.0.borrow_mut().rs_accumulate(item.1, rng)?.to_vec()),
        )
        .collect::<Result<Vec<Vec<Transition>>, PyErr>>()
}

/// A transition of a state machine from one state to another.
///
/// Transitions can occur at any point in time, i.e. the time dimension is continuous.
#[pyclass(frozen)]
#[derive(Clone, Debug)]
pub struct Transition {
    #[pyo3(get)]
    from_state: State,

    #[pyo3(get)]
    time: Time,

    #[pyo3(get)]
    to_state: State,
}

impl Transition {
    pub fn from_state(&self) -> State {
        self.from_state
    }

    pub fn time(&self) -> Time {
        self.time
    }

    pub fn to_state(&self) -> State {
        self.to_state
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
    m.add_class::<PyAccumulator>()?;
    m.add_class::<Transition>()?;
    Ok(())
}

mod tests {
    #[cfg(test)]
    use ndarray::{arr1, ArrayView1};

    use super::PyAccumulator;

    #[test]
    fn par_accumulate() {
        let n = 10;
        let mut accumulators: Vec<PyAccumulator> = Vec::with_capacity(n);
        let ctrl_params = arr1(&[1.0]);
        let mut ctrl_params_per_machine: Vec<ArrayView1<f64>> = Vec::with_capacity(n);
        for _ in 0..n {
            accumulators.push(PyAccumulator::new(10, 1.0));
            ctrl_params_per_machine.push(ctrl_params.view());
        }

        let results = par_accumulate(accumulators.as_mut_slice(), ctrl_params_per_machine.as_slice());

        assert_eq!(n, results.unwrap().len())
    }
}
