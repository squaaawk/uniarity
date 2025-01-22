# Uniarity

Uniarity implements various utilities for univariate functions, particularly for root and minimum finding.

## Features

* Iterative root finding via Newton's method and the secant method
* Root and minima bracket determination
  * Root finding within a bracket via bisection and ITP
  * Minima finding within a bracket via inspection and Brent's Method
* Function approximation and root finding via Chebyshev polyonimal approximation

## Example

```rust
use std::f64::consts::PI;
let f = |x: f64| x.sin();
let x = uniarity::bracket::bisection(&f, 1.0, 5.0, 1e-15);
assert!((x - PI).abs() < 1e-15);
```
