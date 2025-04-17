pub use crate::const_fma::*;

use super::Quad;

pub const fn quick_two_sum(large: f64, small: f64) -> Quad {
    let s = large + small;
    let e = small - (s - large);
    Quad(s, e)
}

pub const fn fabs(a: f64) -> f64 {
    f64::from_bits(a.to_bits() & (u64::MAX >> 1))
}

pub const fn two_sum(a: f64, b: f64) -> Quad {
    let (a, b) = if fabs(a) > fabs(b) { (a, b) } else { (b, a) };
    quick_two_sum(a, b)
}

pub const fn two_prod(a: f64, b: f64) -> Quad {
    let p = a * b;
    let e = fma(a, b, -p);
    Quad(p, e)
}

pub const fn add_accurate(a: Quad<f64>, b: Quad<f64>) -> Quad<f64> {
    let Quad(s0, e1) = two_sum(a.0, b.0);
    let Quad(s1, e2) = two_sum(a.1, b.1);

    let e1 = e1 + s1;

    let Quad(s0, e1) = quick_two_sum(s0, e1);
    let e1 = e1 + e2;

    quick_two_sum(s0, e1)
}

pub const fn add_estimate(a: Quad<f64>, b: Quad<f64>) -> Quad<f64> {
    let Quad(s0, e1) = two_sum(a.0, b.0);
    let s1 = a.1 + b.1;
    let e1 = e1 + s1;

    quick_two_sum(s0, e1)
}

pub const fn mul(a: Quad, b: Quad) -> Quad {
    let Quad(p0, e1) = two_prod(a.0, b.0);
    let e1 = fma(a.0, b.1, e1);
    let e1 = fma(a.1, b.0, e1);
    quick_two_sum(p0, e1)
}

pub const fn abs(a: Quad) -> Quad {
    if a.0.is_sign_negative() {
        Quad(-a.0, -a.1)
    } else {
        Quad(a.0, a.1)
    }
}
