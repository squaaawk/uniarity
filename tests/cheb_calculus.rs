#![allow(non_snake_case)]

use approx::assert_abs_diff_eq;

use uniarity::cheb::Cheb;

/// Max |got - want| over 500 sample points inside [a, b].
fn assert_max_err(
  a: f64,
  b: f64,
  got: impl Fn(f64) -> f64,
  want: impl Fn(f64) -> f64,
  epsilon: f64,
) {
  for i in 0..=500 {
    let x = a + (b - a) * i as f64 / 500.0;
    assert_abs_diff_eq!(got(x), want(x), epsilon = epsilon);
  }
}

#[test]
fn derivative_and_integral_are_inverses() {
  let (a, b) = (0.2, 3.0);
  let c = Cheb::new(&|x: f64| (1.5 * x).exp() * x.cos(), a, b, 30);

  // d/dx of the integral recovers the original series.
  let back = c.integrate().derivative();
  assert_max_err(a, b, |x| back.evaluate(x), |x| c.evaluate(x), 1e-12);

  // The integral of the derivative recovers f(x) - f(a).
  let fwd = c.derivative().integrate();
  let fa = c.evaluate(a);
  assert_max_err(a, b, |x| fwd.evaluate(x), |x| c.evaluate(x) - fa, 1e-9);
}

#[test]
fn constant_function() {
  // derivative is zero, integral is linear
  let c = Cheb::new(&|_| 3.0, 1.0, 4.0, 5);
  assert_abs_diff_eq!(c.derivative().evaluate(2.0), 0.0, epsilon = 1e-14);
  assert_abs_diff_eq!(c.integrate().evaluate(4.0), 9.0, epsilon = 1e-14);
}

#[test]
fn definite_integral_matches_closed_form() {
  // int_0^2 x e^x dx = e^2 + 1
  let c = Cheb::new(&|x: f64| x * x.exp(), 0.0, 2.0, 20);
  assert_abs_diff_eq!(
    c.integrate().evaluate(2.0),
    2.0f64.exp() + 1.0,
    epsilon = 1e-14
  );
}

#[test]
fn derivative_of_sin_is_cos() {
  let (a, b) = (0.3, 2.1);
  let c = Cheb::new(&|x: f64| x.sin(), a, b, 20);
  let d = c.derivative();
  assert_eq!(d.degree(), c.degree() - 1);
  assert_max_err(a, b, |x| d.evaluate(x), |x| x.cos(), 1e-12);
}

#[test]
fn integral_of_cos() {
  let (a, b) = (-0.8, 1.7);
  let c = Cheb::new(&|x: f64| x.cos(), a, b, 20);
  let i = c.integrate();
  assert_eq!(i.degree(), c.degree() + 1);
  assert!(i.evaluate(a).abs() < 1e-14);
  assert_max_err(a, b, |x| i.evaluate(x), |x| x.sin() - a.sin(), 1e-12);
}

#[test]
fn exact_on_a_degree_8_polynomial() {
  // 9 points is exact for degree 8, so derivative and integral should be exact.
  let (a, b) = (-0.7, 1.9);
  let f = |x: f64| x.powi(8) - 3.0 * x.powi(3) + 2.0;
  let df = |x: f64| 8.0 * x.powi(7) - 9.0 * x.powi(2);
  let F = |x: f64| x.powi(9) / 9.0 - 0.75 * x.powi(4) + 2.0 * x;
  let c = Cheb::new(&f, a, b, 9);
  assert_max_err(a, b, |x| c.evaluate(x), f, 1e-12);
  assert_max_err(a, b, |x| c.derivative().evaluate(x), df, 1e-11);
  assert_max_err(a, b, |x| c.integrate().evaluate(x), |x| F(x) - F(a), 1e-12);
}
