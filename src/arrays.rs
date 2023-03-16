/// Array operations specific to rate coefficient calculations.
use super::{CtrlParam, Rate};

#[derive(Debug)]
pub struct ArrayError;

#[derive(Clone)]
pub struct Array2D {
    pub data: Vec<Rate>,
    pub shape: (usize, usize),
}

impl Array2D {
    pub fn new(data: Vec<Rate>, shape: (usize, usize)) -> Result<Array2D, ArrayError> {
        if data.len() != shape.0 * shape.1 {
            return Err(ArrayError);
        }

        Ok(Array2D { data, shape })
    }
}

/// Raise a vector of control parameters to integer powers of 1 to order.
pub fn power(data: &[CtrlParam], order: u8) -> Array2D {
    let mut result = Vec::with_capacity(data.len() * usize::from(order));

    for i in data.iter() {
        for j in 1..i32::from(order + 1) {
            result.push((*i).powi(j));
        }
    }

    let shape = (data.len(), usize::from(order));

    Array2D::new(result, shape).expect("failed to create 2D array")
}

pub struct Array4D {
    data: Vec<Rate>,
    pub shape: (usize, usize, usize, usize),
}

impl Array4D {
    pub fn new(
        data: Vec<Rate>,
        shape: (usize, usize, usize, usize),
    ) -> Result<Array4D, ArrayError> {
        if data.len() != shape.0 * shape.1 * shape.2 * shape.3 {
            return Err(ArrayError);
        }

        Ok(Array4D { data, shape })
    }
}

/// Compute the Einstein summation "ijkl->kl" of a I x J 2D array and a I x J x K x L 4D array.
pub fn tensordot(arr1: &Array2D, arr2: &Array4D) -> Array2D {
    let (i1, j2) = arr1.shape;
    let (i2, j2, k2, l2) = arr2.shape;

    let mut result = Vec::with_capacity(k2 * l2);

    for k in 0..k2 {
        for l in 0..l2 {
            let mut total = 0f64;

            for i in 0..i2 {
                for j in 0..j2 {
                    total += arr1.data[i * j2 + j]
                        * arr2.data[(i * j2 * k2 * l2) + (j * k2 * l2) + (k * l2) + l];
                }
            }
            result.push(total);
        }
    }

    Array2D {
        data: result,
        shape: (k2, l2),
    }
}

mod tests {
    #[cfg(test)]
    use super::{power, tensordot, Array2D, Array4D};

    #[test]
    fn test_power() {
        let ctrl_params: [f64; 2] = [2.0, 3.0];
        let expected: Vec<f64> = vec![2.0, 4.0, 8.0, 3.0, 9.0, 27.0];

        let result = power(&ctrl_params, 3);

        for (actual, expected) in result.data.into_iter().zip(expected.into_iter()) {
            assert_eq!(actual, expected)
        }
    }

    #[test]
    fn test_tensordot() {
        let ctrl_params: [f64; 2] = [2.0, 3.0];
        let powers = power(&ctrl_params, 3);
        let rate_coefficients = Array4D {
            data: vec![
                0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 4.0, 0.0,
                0.0, 2.0, 0.5, 0.0, 0.0, 2.0, 2.0, 0.0,
            ],
            shape: (2, 3, 2, 2),
        };
        let expected: Vec<f64> = vec![0.0, 91.0, 84.5, 0.0];

        let result = tensordot(&powers, &rate_coefficients);

        for (actual, expected) in result.data.into_iter().zip(expected.into_iter()) {
            assert_eq!(actual, expected)
        }
    }
}
