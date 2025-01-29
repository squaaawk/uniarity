#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

pub(crate) fn compute_epsilon(a: f64, b: f64, tol: f64) -> f64 {
  (2.0 * tol) * a.abs().max(b.abs())
}

/// Represents an x-coordinate on a function, along with a potentially-known evaluation at that coordinate.
pub enum MaybeEval {
  /// The function value at this coordinate is known.
  Known(f64, f64),
  /// The function value at this coordinate is unknown.
  Unknown(f64),
}

impl MaybeEval {
  /// Just the x-coordinate, regardless of whether the evaluation is known or not
  pub fn x(&self) -> f64 {
    match *self {
      MaybeEval::Known(x, _) => x,
      MaybeEval::Unknown(x) => x,
    }
  }

  /// Just the evaluated coordinate
  pub fn fx(&self) -> Option<f64> {
    match *self {
      MaybeEval::Known(_, fx) => Some(fx),
      MaybeEval::Unknown(_) => None,
    }
  }

  /// Returns either the known function evaluation, or, if it is unknown, computes it with the given function.
  pub fn evaled<F>(&self, f: F) -> (f64, f64)
  where
    F: Fn(f64) -> f64,
  {
    match *self {
      MaybeEval::Known(x, fx) => (x, fx),
      MaybeEval::Unknown(x) => (x, f(x)),
    }
  }
}

impl From<f64> for MaybeEval {
  fn from(value: f64) -> Self {
    Self::Unknown(value)
  }
}

impl From<(f64, f64)> for MaybeEval {
  fn from(value: (f64, f64)) -> Self {
    Self::Known(value.0, value.1)
  }
}

pub mod bracket;
pub mod cheb;
pub mod initial;
pub mod min;
