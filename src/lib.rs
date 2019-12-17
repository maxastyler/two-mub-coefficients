extern crate num;
extern crate ndarray;
#[macro_use]
extern crate itertools;

use ndarray::{Array, Array4, Zip};
use num::complex::Complex;
use num::Zero;
use numpy::{IntoPyArray, PyArray4};
use pyo3::prelude::{pymodule, Py, PyErr, PyModule, PyResult, Python};

fn c_val_32(k: i32, kd: i32, m: i32, md: i32, n: i32, nd: i32, dim: i32) -> Complex<f32> {
    let omega: Complex<f32> =
        Complex::from_polar(&1.0, &(2.0 * std::f32::consts::PI / (dim as f32)));
    let diff = k - kd;

    iproduct!(0..dim, 0..dim, 0..dim).fold(Complex::zero(), |acc: Complex<f32>, (p, pd, r)| {
        let rd = (pd - p + r).rem_euclid(dim);
        acc + omega.powi(
            p * md - pd * nd - r * m
                + rd * n
                + diff * (p.pow(2) - pd.pow(2) - r.pow(2) + rd.pow(2)),
        )
    }) / ((dim.pow(2)) as f32)
}

fn c_val_64(k: i64, kd: i64, m: i64, md: i64, n: i64, nd: i64, dim: i64) -> Complex<f64> {
    let omega: Complex<f64> =
        Complex::from_polar(&1.0, &(2.0 * std::f64::consts::PI / (dim as f64)));
    let diff = k - kd;

    iproduct!(0..dim, 0..dim, 0..dim).fold(Complex::zero(), |acc: Complex<f64>, (p, pd, r)| {
        let rd = (pd - p + r).rem_euclid(dim);
        acc + omega.powi((
            p * md - pd * nd - r * m
                + rd * n
                + diff * (p.pow(2) - pd.pow(2) - r.pow(2) + rd.pow(2))) as i32,
        )
    }) / ((dim.pow(2)) as f64)
}

fn c_mat_32(k: i32, kd: i32, dim: i32) -> Array4<Complex<f32>> {
    let d = dim as usize;
    let mut a = Array4::zeros((d, d, d, d));
    Zip::indexed(&mut a).par_apply(|(m, md, n, nd), a| {
        *a = c_val_32(k, kd, m as i32, md as i32, n as i32, nd as i32, dim)
    });
    a
}

fn c_mat_64(k: i64, kd: i64, dim: i64) -> Array4<Complex<f64>> {
    let d = dim as usize;
    let mut a = Array4::zeros((d, d, d, d));
    Zip::indexed(&mut a).par_apply(|(m, md, n, nd), a| {
        *a = c_val_64(k, kd, m as i64, md as i64, n as i64, nd as i64, dim)
    });
    a
}

#[pymodule]
fn two_mub_coefficients(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "c_mat_32")]
    fn py_c_mat_32(py: Python<'_>, k: i32, kd: i32, d: i32) -> PyResult<Py<PyArray4<Complex<f32>>>> {
        Ok(c_mat_32(k, kd, d).into_pyarray(py).to_owned())
    };
    
    #[pyfn(m, "c_mat_64")]
    fn py_c_mat_64(py: Python<'_>, k: i64, kd: i64, d: i64) -> PyResult<Py<PyArray4<Complex<f64>>>> {
        Ok(c_mat_64(k, kd, d).into_pyarray(py).to_owned())
    };
    Ok(())    
}

#[cfg(test)]
mod tests
{
    use super::{c_val_32, c_val_64};
    use num::complex::Complex;
    use crate::num::Zero;
    #[test]
    fn test_one_equals_two() {
        assert_eq!(Complex::zero(), c_val_32(2, 3, 1, 2, 5, 3, 10));
    }
}
