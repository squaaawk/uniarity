#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

pub(crate) fn compute_epsilon(a: f64, b: f64, tol: f64) -> f64 {
  (2.0 * tol) * a.abs().max(b.abs())
}

pub mod bracket;
pub mod cheb;
pub mod initial;
pub mod min;
