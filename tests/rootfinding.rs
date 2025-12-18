use approx::assert_abs_diff_eq;
use std::f64::consts::TAU;

use uniarity::bracket::{bisection, itp};
use uniarity::cheb::Cheb;
use uniarity::initial::{laguerres_method, newtons_method, secant};

struct TestCase {
  function: fn(f64) -> f64,
  a: f64,
  b: f64,
  n: usize,
  low_precision: bool,
}

const TESTS: [TestCase; 8] = [
  TestCase {
    function: |x: f64| 0.72 * x - 1.0,
    a: -5.0,
    b: 5.0,
    n: 4,
    low_precision: false,
  },
  TestCase {
    function: |x: f64| 0.72 * x * x - 1.0,
    a: -1.0,
    b: 5.0,
    n: 4,
    low_precision: false,
  },
  TestCase {
    function: |x: f64| x.powi(3) - x + 0.5,
    a: -2.0,
    b: 2.0,
    n: 4,
    low_precision: false,
  },
  TestCase {
    function: |x: f64| x * x.exp() - 1.0,
    a: 0.0,
    b: 2.0,
    n: 6,
    low_precision: false,
  },
  TestCase {
    function: |x: f64| x - x.sin() - 1.2,
    a: 0.0,
    b: TAU,
    n: 6,
    low_precision: false,
  },
  TestCase {
    function: |x: f64| (x / 1e6) - (x / 1e6).sin() - 1.2,
    a: 0.0,
    b: 1e6 * TAU,
    n: 6,
    low_precision: false,
  },
  TestCase {
    function: |x: f64| -x.powi(11) + 1e-10,
    a: -1.0,
    b: 1.0,
    n: 12,
    low_precision: true,
  },
  TestCase {
    function: |x: f64| (20.0 * x).sin() + 10.0 * x.tanh() + 1.0,
    a: -1.0,
    b: 1.0,
    n: 40,
    low_precision: false,
  },
];

#[test]
fn test_secant() {
  for case in TESTS {
    let f = &case.function;

    // A very crude initial guess
    let x = (case.a + case.b) / 2.0;
    let x = secant(f, x, x + 1e-6, f64::EPSILON);

    let epsilon = if case.low_precision { 1e-10 } else { 1e-14 };
    assert_abs_diff_eq!(f(x), 0.0, epsilon = epsilon);
  }
}

#[test]
fn test_newton() {
  for case in TESTS {
    let f = &case.function;

    // TODO: Compute the derivative directly
    let h = 1e-6;
    let fp = &|x: f64| (f(x + h) - f(x)) / h;

    // A very crude initial guess
    let x = (case.a + case.b) / 2.0;
    let x = newtons_method(f, fp, x, f64::EPSILON);

    let epsilon = if case.low_precision { 1e-10 } else { 1e-14 };
    assert_abs_diff_eq!(f(x), 0.0, epsilon = epsilon);
  }
}

#[test]
fn test_laguerre() {
  for case in TESTS {
    let f = &case.function;

    // TODO: Compute the derivative directly
    let h = 1e-6;
    let fp = &|x: f64| (f(x + h) - f(x)) / h;
    let fpp = &|x: f64| (fp(x + h) - fp(x)) / h;

    // A very crude initial guess
    let x = (case.a + case.b) / 2.0;
    let x = laguerres_method(f, fp, fpp, 1.0, x, f64::EPSILON);

    let epsilon = if case.low_precision { 1e-10 } else { 1e-13 };
    assert_abs_diff_eq!(f(x), 0.0, epsilon = epsilon);
  }
}

#[test]
fn test_bisection() {
  for case in TESTS {
    let f = &case.function;
    let x = bisection(f, case.a, case.b, f64::EPSILON);
    assert_abs_diff_eq!(f(x), 0.0, epsilon = 1e-14);
  }
}

#[test]
fn test_itp() {
  for case in TESTS {
    let f = &case.function;
    let x = itp(f, case.a, case.b, f64::EPSILON);
    assert_abs_diff_eq!(f(x), 0.0, epsilon = 1e-14);
  }
}

#[test]
fn test_cheb() {
  for case in TESTS {
    let f = &case.function;

    for n in case.n..case.n + 20 {
      let cheb = Cheb::new(f, case.a, case.b, n);

      let roots = cheb.roots();
      assert_eq!(roots.len(), 1);

      // The root should be approximately right
      let x = roots[0];
      assert_abs_diff_eq!(f(x), 0.0, epsilon = 1e-1);

      // And close enough to the true root
      let x = secant(f, x, x + 1e-6, f64::EPSILON);
      assert_abs_diff_eq!(f(x), 0.0, epsilon = 1e-15);
    }
  }
}
