use approx::assert_abs_diff_eq;
use std::f64::consts::PI;

use uniarity::min::min;

#[test]
fn test_minimization_degenerate() {
  // Test to ensure we don't loop forever in a degenerate case
  min(&|_| 0.0, 0.0, 1.0, 1e-15);
}

// TODO: Tolerance should be improved
#[test]
fn test_minimization_linear() {
  let (x, y) = min(&|x| 1.0 - PI * x, 0.0, 1.0, 1e-15);
  assert_abs_diff_eq!(x, 1.0, epsilon = 1e-9);
  assert_abs_diff_eq!(y, 1.0 - PI, epsilon = 1e-9);
}

#[test]
fn test_minimization() {
  let (x, y) = min(&|x| x.exp() + x * x, -2.0, 2.0, 1e-15);
  assert_abs_diff_eq!(x, -0.35173371124919584, epsilon = 1e-9);
  assert_abs_diff_eq!(y, 0.8271840261275243, epsilon = 1e-9);
}
