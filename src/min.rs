//! Methods to determine the minimum of a given function within a given range or bracket.

use ordered_float::OrderedFloat;

use crate::compute_epsilon;

/// Samples `n` points along the function, and returns the point with the minimum value.
pub fn min_by_inspection<F>(f: &F, a: f64, b: f64, n: usize) -> (f64, f64)
where
  F: Fn(f64) -> f64,
{
  let step = (b - a) / (n - 1) as f64;
  (0..n)
    .map(|i| {
      let x = a + i as f64 * step;
      (x, f(x))
    })
    .min_by_key(|&(_, fx)| OrderedFloat(fx))
    .unwrap()
}

// TODO: Provide golden section search as an additional method, as in bracket::locate_negative
// TODO: It may be more useful for Brent's method to take a triplet as a bracket

/// Returns the minimum of a function within the given bracket. This implementation uses Brent's algorithm, as described in this [paper].
///
/// [paper]: https://phys.uri.edu/nigh/NumRec/bookfpdf/f10-2.pdf
#[allow(clippy::collapsible_else_if)]
pub fn min<F>(f: &F, a: f64, b: f64, tol: f64) -> (f64, f64)
where
  F: Fn(f64) -> f64,
{
  let ax = a;
  let cx = b;
  let bx = 0.5 * (a + b);

  let tol = compute_epsilon(a, b, tol);
  let c_gold = 0.3819660112501052; // (phi - 1)^2
  let z_eps = 1e-10;

  let mut d = 0.0;

  let mut a = ax.min(cx);
  let mut b = ax.max(cx);
  let mut v = bx;
  let mut w = v;
  let mut x = v;
  let mut e = 0.0_f64;
  let mut fx = f(x);
  let mut fv = fx;
  let mut fw = fx;

  // TODO: Not forever
  loop {
    let xm = 0.5 * (a + b);
    let tol1 = tol * x.abs() + z_eps;
    let tol2 = 2.0 * tol1;

    if (x - xm).abs() <= tol2 - 0.5 * (b - a) {
      return (x, f(x));
    }

    if e.abs() > tol1 {
      let r = (x - w) * (fx - fv);
      let mut q = (x - v) * (fx - fw);
      let mut p = (x - v) * q - (x - w) * r;

      q = 2.0 * (q - r);
      if q > 0.0 {
        p = -p;
      } else {
        q = -q;
      }

      let e_prev = e;
      e = d;
      if p.abs() >= (0.5 * q * e_prev).abs() || p <= q * (a - x) || p >= q * (b - x) {
        e = if x >= xm { a - x } else { b - x };
        d = c_gold * e;
      } else {
        d = p / q;
        let u = x + d;
        if u - a < tol2 || b - u < tol2 {
          d = tol1.copysign(xm - x);
        }
      }
    } else {
      e = if x >= xm { a - x } else { b - x };
      d = c_gold * e;
    }

    let u = if d.abs() >= tol1 {
      x + d
    } else {
      x + tol1.copysign(d)
    };

    let fu = f(u);

    if fu <= fx {
      if u >= x {
        a = x;
      } else {
        b = x;
      }

      (v, w, x) = (w, x, u);
      (fv, fw, fx) = (fw, fx, fu);
    } else {
      if u < x {
        a = u;
      } else {
        b = u;
      }
    }

    if fu <= fw || w == x {
      v = w;
      fv = fw;
      w = u;
      fw = fu;
    } else if fu <= fv || v == x || v == w {
      v = u;
      fv = fu;
    }
  }
}
