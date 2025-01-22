//! Methods to determine the a root of a univariate function using an initial approximation.

/// Uses Newton's method to locate the root of a function, given an initial value.
/// Terminates after |f(x)| <= tol, |g(x)| <= tol, or after 100 iterations.
pub fn newtons_method<F, Fp>(f: &F, g: &Fp, mut x: f64, tol: f64) -> f64
where
  F: Fn(f64) -> f64,
  Fp: Fn(f64) -> f64,
{
  let mut fx = f(x);
  let mut gx = g(x);

  let max_iterations = 100;
  let mut iterations = 0;

  while fx.abs() > tol && gx.abs() > tol && iterations < max_iterations {
    x -= fx / gx;
    fx = f(x);
    gx = g(x);
    iterations += 1;
  }

  x
}

/// Uses the secant method to locate the root of a function, given an initial pair of values.
/// Terminates after |x0 - x1| <= tol, |f(x0) - f(x1)| <= tol, or after 100 iterations.
pub fn secant<F>(f: &F, mut x0: f64, mut x1: f64, tol: f64) -> f64
where
  F: Fn(f64) -> f64,
{
  let mut f0 = f(x0);
  let mut f1 = f(x1);

  let max_iterations = 100;
  let mut iterations = 0;

  while (x1 - x0).abs() > tol && (f1 - f0).abs() > tol && iterations < max_iterations {
    let x = x1 - f1 * (x1 - x0) / (f1 - f0);
    (x0, f0) = (x1, f1);
    (x1, f1) = (x, f(x));
    iterations += 1;
  }

  x1
}
