extern crate num;
extern crate ndarray;
#[macro_use]
extern crate itertools;

use ndarray::{Array, Array4, Zip};
use num::complex::Complex;
use num::Zero;
use numpy::{IntoPyArray, PyArray4};
use pyo3::prelude::{pymodule, Py, PyErr, PyModule, PyResult, Python};

fn c_val(k: i32, kd: i32, m: i32, md: i32, n: i32, nd: i32, dim: i32) -> Complex<f32> {
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

fn c_mat(k: i32, kd: i32, dim: i32) -> Array4<Complex<f32>> {
    let d = dim as usize;
    let mut a = Array4::zeros((d, d, d, d));
    Zip::indexed(&mut a).par_apply(|(m, md, n, nd), a| {
        *a = c_val(k, kd, m as i32, md as i32, n as i32, nd as i32, dim)
    });
    a
}

#[pymodule]
fn two_mub_coefficients(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "c_mat")]
    fn inv(py: Python<'_>, k: i32, kd: i32, d: i32) -> PyResult<Py<PyArray4<Complex<f32>>>> {
        Ok(c_mat(k, kd, d).into_pyarray(py).to_owned())
    };
    Ok(())
}
