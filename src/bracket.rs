//! Methods to either identify a bracket of a given function, or, given a bracket, identify the root within it.
//!
//! A root bracket refers to a pair of abscissa `a` and `b` such that `f(a)` and `f(b)` have different signs.
//! By the intermediate value theorem, assuming `f` is continuous, such a bracket must contain a root.
//!
//! A minima bracket refers to a pair of abscissa `a` and `b` such that both `f(a)` and `f(b)` are larger than
//! some minima contained between them.

use crate::{compute_epsilon, MaybeEval};

/// Locates the root within a bracket using the bisection method.
/// Requires that `f` is continuous and that `f(a)` and `f(b)` have opposite signs.
pub fn bisection<F>(f: &F, a: impl Into<MaybeEval>, b: impl Into<MaybeEval>, tol: f64) -> f64
where
  F: Fn(f64) -> f64,
{
  let a = a.into();
  let b = b.into();
  assert!(a.x() < b.x());

  let (mut a, fa) = a.evaled(f);
  let mut b = b.x();

  let epsilon = compute_epsilon(a, b, tol);
  let fa_sign = fa.signum();

  while b - a > epsilon {
    let x = 0.5 * (a + b);
    if f(x).signum() == fa_sign {
      a = x;
    } else {
      b = x;
    }
  }

  0.5 * (a + b)
}

/// Locates the root within a bracket using the [ITP method].
/// Requires that `f` is continuous and that `f(a)` and `f(b)` have opposite signs.
///
/// The parameters for the ITP method are set at:
/// - `k1 = 0.2 / (b - a)`
/// - `k2 = 2`
/// - `n0 = 5`
///
/// [ITP Method]: https://dl.acm.org/doi/10.1145/3423597
pub fn itp<F>(f: &F, a: impl Into<MaybeEval>, b: impl Into<MaybeEval>, tol: f64) -> f64
where
  F: Fn(f64) -> f64,
{
  let a = a.into();
  let b = b.into();
  assert!(a.x() < b.x());

  let (mut a, mut fa) = a.evaled(f);
  let (mut b, mut fb) = b.evaled(f);

  let n0 = 5;
  let k1 = 0.2 / (b - a);
  let k2 = 2;
  let epsilon = compute_epsilon(a, b, tol);

  let n1_2 = (((b - a) / epsilon).log2().ceil() - 1.0).max(0.0) as usize;
  let n_max = n0 + n1_2;
  let mut scaled_epsilon = epsilon * 2f64.powi(n_max as i32);

  // The algorithm assumes f(a) <= f(b). If not, we must correct for it
  let negate = fb < fa;

  while b - a > 2.0 * epsilon {
    let x1_2 = 0.5 * (a + b);
    let r = scaled_epsilon - 0.5 * (b - a);
    let delta = k1 * (b - a).powi(k2);

    // Interpolation
    let xf = (fb * a - fa * b) / (fb - fa);

    // Truncation
    let sigma = x1_2 - xf;
    let xt = if delta <= (x1_2 - xf).abs() {
      xf + delta.copysign(sigma)
    } else {
      x1_2
    };

    // Projection
    let x_itp = if (xt - x1_2).abs() <= r {
      xt
    } else {
      x1_2 - r.copysign(sigma)
    };

    // Update interval
    let f_itp = f(x_itp);

    if f_itp == 0.0 {
      return x_itp;
    } else if negate ^ (f_itp > 0.0) {
      (b, fb) = (x_itp, f_itp);
    } else {
      (a, fa) = (x_itp, f_itp);
    }

    scaled_epsilon *= 0.5;
  }

  0.5 * (a + b)
}

/// Determines a bracket around a minimum of the given function by first evaluating at `x` and then searching in the direction of `step` with successively doubling step sizes.
/// Assumes `f(x)` is positive, `f` decreases in the direction of `step`, and that we're looking for a minimum.
pub fn find_bracket<F>(
  f: &F,
  x: impl Into<MaybeEval>,
  min_x: f64,
  max_x: f64,
  mut step: f64,
) -> Option<(MaybeEval, MaybeEval)>
where
  F: Fn(f64) -> f64,
{
  // Exponentially step along the path until we find a bracket
  // step is in the downhill direction
  let (mut a, mut fa) = x.into().evaled(f);
  assert!(fa >= 0.0);

  let mut b = a;
  let mut fb;

  // TODO: Not forever
  loop {
    b += step;
    fb = f(b);

    // We've explored up to the boundary without finding a bracket
    if b < min_x || b > max_x {
      return None;
    }

    if fb > fa {
      a -= 0.5 * step;
      return Some((a.into(), (b, fb).into()));
    }

    a = b;
    fa = fb;

    step *= 2.0;
  }
}

/// Determines a bracket around a root of the given function by first evaluating at `x`
/// and then searching in the direction of `step` with successively doubling step sizes.
pub fn find_root_bracket<F>(
  f: &F,
  x: impl Into<MaybeEval>,
  mut step: f64,
) -> Option<(MaybeEval, MaybeEval)>
where
  F: Fn(f64) -> f64,
{
  let (mut x, mut fx) = x.into().evaled(f);
  let sign = fx.signum();

  while x.is_finite() {
    let new_x = x + step;
    let new_fx = f(new_x);

    if new_fx.signum() != sign {
      return Some(((x, fx).into(), (new_x, new_fx).into()));
    }

    x = new_x;
    fx = new_fx;

    step *= 2.0;
  }

  None
}

/// Locate a negative value on the given function by first evaluating at `x`
/// and then searching in the direction of `step` with successively doubling step sizes.
/// Assumes f(x) is positive and it decreases in the direction of step.
pub fn find_negative_from<F>(
  f: &F,
  x: impl Into<MaybeEval>,
  mut step: f64,
  min_x: f64,
  max_x: f64,
) -> Option<MaybeEval>
where
  F: Fn(f64) -> f64,
{
  // Exponentially step along the path until we find a bracket
  // step is in the downhill direction
  let (mut a, mut fa) = x.into().evaled(f);

  if fa.is_sign_negative() {
    return Some((a, fa).into());
  }

  let mut b = a;
  let mut fb;

  // TODO: Not forever
  loop {
    b += step;
    fb = f(b);

    // We've explored up to the boundary without finding a bracket
    if b < min_x || b > max_x {
      return None;
    }

    if fb < 0.0 {
      return Some((b, fb).into());
    }

    // We found a bracket; find a negative within it
    if fb > fa {
      let a = a - 0.5 * step;
      return locate_negative(f, a.min(b), a.max(b), 1e-15);
    }

    a = b;
    fa = fb;
    step *= 2.0;
  }
}

/// Locates a negative value within the range bracket defined by `a` and `b`.
// TODO: At the moment, this function uses golden selection search. It would be nice to optionally use brent's algorithm from min
pub fn locate_negative<F>(
  f: F,
  a: impl Into<MaybeEval>,
  b: impl Into<MaybeEval>,
  tol: f64,
) -> Option<MaybeEval>
where
  F: Fn(f64) -> f64,
{
  let a = a.into();
  let b = b.into();
  assert!(a.x() < b.x());

  let (mut a, fa) = a.evaled(&f);
  if fa < 0.0 {
    return Some((a, fa).into());
  }

  let (mut b, fb) = b.evaled(&f);
  if fb < 0.0 {
    return Some((b, fb).into());
  }

  let epsilon = compute_epsilon(a, b, tol);

  let phi: f64 = 0.5 * (1.0 + 5f64.sqrt());
  let phi_inv: f64 = phi.recip();

  let mut c = b - (b - a) * phi_inv;
  let mut d = a + (b - a) * phi_inv;

  while b - a > epsilon {
    let fc = f(c);
    if fc < 0.0 {
      return Some((c, fc).into());
    }

    let fd = f(d);
    if fd < 0.0 {
      return Some((d, fd).into());
    }

    if fc < fd {
      b = d;
    } else {
      a = c;
    }

    c = b - (b - a) * phi_inv;
    d = a + (b - a) * phi_inv;
  }

  None
}
