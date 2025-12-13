#![allow(non_snake_case)]

//! A function may be approximated by a Chebyshev polynomial by sampling a number of values and then interpolating.
//! The Chebyshev polynomial makes computing values such as roots convenient. The method is typically more expensive
//! than classic rootfinding methods, but is general and can find all the roots on an interval.
//!
//! This implementation follows the [CPR Paper] and uses ideas from the implementation within [chebfun].
//!
//! [CPR Paper]: https://epubs.siam.org/doi/pdf/10.1137/110838297
//! [chebfun]: https://github.com/chebfun/chebfun

use ordered_float::OrderedFloat;
use std::f64::consts::PI;

use faer::{Col, Mat, Row};

/// Maps an x-value from the range \[a, b\] to \[-1, 1\].
#[inline]
fn local_space(a: f64, b: f64, x: f64) -> f64 {
  (2.0 * x - a - b) / (b - a)
}

/// Maps an x-value from the range \[-1, 1\] to \[a, b\].
#[inline]
fn function_space(a: f64, b: f64, x: f64) -> f64 {
  0.5 * (x * (b - a) + a + b)
}

fn compute_coefficients<F>(f: &F, a: f64, b: f64, n: usize) -> Vec<f64>
where
  F: Fn(f64) -> f64,
{
  let ff = Col::from_fn(n, |i| {
    let x = (PI * (i as f64 + 0.5) / (n as f64)).cos();
    f(function_space(a, b, x))
  });

  // let z = (0..n)
  //   .map(|i| {
  //     let x = (PI * (i as f64 + 0.5) / (n as f64)).cos();
  //     (x, f(function_space(a, b, x)))
  //   })
  //   .collect::<Vec<_>>();
  // println!("{z:?}");

  let mut c: Vec<f64> = (0..n)
    .map(|j| {
      let b = Row::from_fn(n, |x| {
        (PI * ((j as f64 * (x as f64 + 0.5)) / (n as f64))).cos()
      });

      let z = b * &ff;
      2.0 * z / n as f64
    })
    .collect();

  // println!("c {c:?}");
  // println!("{n} {}", c.len());

  // Find the last coefficient greater than tol, and truncate everything after it
  let max_val = c
    .iter()
    .map(|&x| x.abs())
    .max_by_key(|&v| OrderedFloat(v))
    .unwrap();
  let tol = (1e-14 * max_val).max(f64::EPSILON);

  // Truncate all coefficients after trunc_i
  if let Some(k) = c.iter().rev().position(|&x| x.abs() >= tol) {
    let trunc_i = c.len() - k - 1;
    c.drain(trunc_i + 1..);
    assert_ne!(c[trunc_i], 0.0);

    c[0] *= 0.5;
    c
  } else {
    c.clear();
    c
  }
}

/// A Cheybyshev polynomial approximation of a function on a given interval.
pub struct Cheb {
  a: f64,
  b: f64,
  c: Vec<f64>,
}

impl Cheb {
  /// Constructs a Chebyshev approximation of a given function on the given interval.
  pub fn new<F>(f: &F, a: f64, b: f64, n: usize) -> Self
  where
    F: Fn(f64) -> f64,
  {
    assert!(b >= a);

    if n == 0 {
      return Self {
        a,
        b,
        c: Vec::new(),
      };
    }

    let c = compute_coefficients(f, a, b, n);
    Self { a, b, c }
  }

  /// Maps an x-value from the range \[a, b\] to \[-1, 1\].
  #[inline]
  fn local_space(&self, x: f64) -> f64 {
    local_space(self.a, self.b, x)
  }

  /// Maps an x-value from the range \[-1, 1\] to \[a, b\].
  #[inline]
  fn function_space(&self, x: f64) -> f64 {
    function_space(self.a, self.b, x)
  }

  // TODO: Implement splitting
  /// Returns all real roots of the Chebyshev approximation within the initial interval.
  pub fn roots(&self) -> Vec<f64> {
    let n = self.c.len();

    // Trivial cases
    if n <= 1 {
      return vec![];
    }

    if n == 2 {
      let x = -self.c[0] / self.c[1];
      return vec![self.function_space(x)];
    }

    // Set up the Chebyshev Companion Matrix
    let mut A = Mat::zeros(n - 1, n - 1);

    for i in 0..n - 2 {
      A[(i + 1, i)] = 0.5;
      A[(i, i + 1)] = 0.5;
    }

    if n > 2 {
      A[(0, 1)] += 0.5;
    }

    let last = self.c[n - 1];
    for (i, &x) in self.c.iter().take(n - 1).enumerate() {
      A[(n - 2, i)] += -x / (2.0 * last);
    }

    // Compute eigenvalues, and from them, roots
    let i_tol = 1e-8;
    let x_tol = 1e-8;

    let eigvals = A.eigenvalues().unwrap();

    let real_eigvals = eigvals
      .into_iter()
      .filter(|z| z.im.abs() <= i_tol)
      .map(|z| z.re);

    let mut roots: Vec<f64> = real_eigvals
      .filter(|x| x.abs() <= 1.0 + x_tol)
      .map(|x| self.function_space(x))
      .collect();

    roots.sort_unstable_by_key(|&v| OrderedFloat(v));
    roots
  }

  /// Evaluates the Chebyshev approximation at a given x-value.
  pub fn evaluate(&self, x: f64) -> f64 {
    let x = self.local_space(x);

    let mut d = 0.0;
    let mut dd = 0.0;

    for &c in self.c.iter().skip(1).rev() {
      (d, dd) = (2.0 * x * d - dd + c, d);
    }

    x * d - dd + self.c[0]
  }

  /// Prints out `n` xy-coordinates along the Chebyshev approximation for use in debugging.
  pub fn debug(&self, n: usize) {
    let points: Vec<_> = (0..n)
      .map(|i| {
        let x = self.a + (self.b - self.a) * (i as f64 / (n - 1) as f64);
        (x, self.evaluate(x))
      })
      .collect();

    println!("{points:?}");
  }
}
