use approx::assert_abs_diff_eq;
use autodiff::{Float, F1};
use std::f64::consts::TAU;

use uniarity::bracket::{bisection, itp};
use uniarity::cheb::Cheb;
use uniarity::initial::{laguerres_method, newtons_method, secant};

struct TestCase {
  function: fn(F1) -> F1,
  a: f64,
  b: f64,
  n: usize,
  low_precision: bool,
}

impl TestCase {
  fn f(&self) -> impl Fn(f64) -> f64 + '_ {
    |x: f64| (self.function)(F1::cst(x)).value()
  }

  fn fp(&self) -> impl Fn(f64) -> f64 + '_ {
    |x: f64| (self.function)(F1::var(x)).deriv()
  }

  fn fpp(&self) -> impl Fn(f64) -> f64 + '_ {
    // TODO: Compute the derivative directly
    // type F2 = F<F<f64, f64>, f64>;
    // |x: f64| (self.function)(F2::var(x)).deriv()

    let fp = self.fp();
    let h = 1e-9;
    move |x: f64| (fp(x + h) - fp(x)) / h
  }
}

const TESTS: [TestCase; 8] = [
  TestCase {
    function: |x| 0.72 * x - 1.0,
    a: -5.0,
    b: 5.0,
    n: 4,
    low_precision: false,
  },
  TestCase {
    function: |x| 0.72 * x * x - 1.0,
    a: -1.0,
    b: 5.0,
    n: 4,
    low_precision: false,
  },
  TestCase {
    function: |x| x.powi(3) - x + 0.6,
    a: -2.0,
    b: 2.0,
    n: 4,
    low_precision: false,
  },
  TestCase {
    function: |x| x * x.exp() - 1.0,
    a: 0.0,
    b: 2.0,
    n: 6,
    low_precision: false,
  },
  TestCase {
    function: |x| x - x.sin() - 1.2,
    a: 0.0,
    b: TAU,
    n: 6,
    low_precision: false,
  },
  TestCase {
    function: |x| (x / 1e6) - (x / 1e6).sin() - 1.2,
    a: 0.0,
    b: 1e6 * TAU,
    n: 6,
    low_precision: false,
  },
  TestCase {
    function: |x| -x.powi(11) + 1e-10,
    a: -1.0,
    b: 1.0,
    n: 12,
    low_precision: true,
  },
  TestCase {
    function: |x| (F1::cst(20.0) * x).sin() + 10.0 * x.tanh() + 1.0,
    a: -1.0,
    b: 1.0,
    n: 40,
    low_precision: false,
  },
];

#[test]
fn test_secant() {
  for case in TESTS {
    let f = &case.f();

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
    let f = &case.f();
    let fp = &case.fp();

    // A very crude initial guess
    let x = (case.a + case.b) / 2.0;
    let x = newtons_method(f, fp, x, f64::EPSILON);

    let epsilon = if case.low_precision { 1e-10 } else { 1e-15 };
    assert_abs_diff_eq!(f(x), 0.0, epsilon = epsilon);
  }
}

#[test]
fn test_laguerre() {
  for case in TESTS {
    let f = &case.f();
    let fp = &case.fp();
    let fpp = &case.fpp();

    // A very crude initial guess
    let x = (case.a + case.b) / 2.0;
    let x = laguerres_method(f, fp, fpp, 1.0, x, f64::EPSILON);

    let epsilon = if case.low_precision { 1e-10 } else { 1e-15 };
    assert_abs_diff_eq!(f(x), 0.0, epsilon = epsilon);
  }
}

#[test]
fn test_bisection() {
  for case in TESTS {
    let f = &case.f();
    let x = bisection(f, case.a, case.b, f64::EPSILON);
    assert_abs_diff_eq!(f(x), 0.0, epsilon = 1e-14);
  }
}

#[test]
fn test_itp() {
  for case in TESTS {
    let f = &case.f();
    let x = itp(f, case.a, case.b, f64::EPSILON);
    assert_abs_diff_eq!(f(x), 0.0, epsilon = 1e-14);
  }
}

#[test]
fn test_cheb() {
  for case in TESTS {
    let f = &case.f();

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
