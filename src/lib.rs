// https://web.mit.edu/tabbott/Public/quaddouble-debian/qd-2.3.4-old/docs/qd.pdf
// https://gitlab.com/hodge_star/mantis
#![cfg_attr(not(feature = "std"), no_std)]

use bytemuck::{Pod, Zeroable};
use pulp::{Scalar, Simd};

#[cfg(not(feature = "std"))]
use libm::Libm;

/// Value representing the implicit sum of two floating point terms, such that the absolute
/// value of the second term is less half a ULP of the first term.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(C)]
pub struct Double<T>(pub T, pub T);

pub struct Pow2(pub f64);

unsafe impl<T: Zeroable> Zeroable for Double<T> {}
unsafe impl<T: Pod> Pod for Double<T> {}

impl<I: Iterator> Iterator for Double<I> {
    type Item = Double<I::Item>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let x0 = self.0.next()?;
        let x1 = self.1.next()?;
        Some(Double(x0, x1))
    }
}

#[inline(always)]
fn quick_two_sum<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
    let s = simd.f64s_add(a, b);
    let err = simd.f64s_sub(b, simd.f64s_sub(s, a));
    (s, err)
}

#[inline(always)]
#[allow(dead_code)]
fn two_sum<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
    let s = simd.f64s_add(a, b);
    let bb = simd.f64s_sub(s, a);

    // (a - (s - bb)) + (b - bb)
    let err = simd.f64s_add(simd.f64s_sub(a, simd.f64s_sub(s, bb)), simd.f64s_sub(b, bb));
    (s, err)
}

#[inline(always)]
#[allow(dead_code)]
fn two_sum_e<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
    let sign_bit = simd.f64s_splat(-0.0);
    let cmp = simd.u64s_greater_than(
        pulp::cast(simd.f64s_or(a, sign_bit)),
        pulp::cast(simd.f64s_or(b, sign_bit)),
    );
    let (a, b) = (
        simd.m64s_select_f64s(cmp, a, b),
        simd.m64s_select_f64s(cmp, b, a),
    );

    quick_two_sum(simd, a, b)
}

#[inline(always)]
#[allow(dead_code)]
fn quick_two_diff<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
    let s = simd.f64s_sub(a, b);
    let err = simd.f64s_sub(simd.f64s_sub(a, s), b);
    (s, err)
}

#[inline(always)]
#[allow(dead_code)]
fn two_diff<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
    let s = simd.f64s_sub(a, b);
    let bb = simd.f64s_sub(s, a);

    // (a - (s - bb)) - (b + bb)
    let err = simd.f64s_sub(simd.f64s_sub(a, simd.f64s_sub(s, bb)), simd.f64s_add(b, bb));
    (s, err)
}

#[inline(always)]
#[allow(dead_code)]
fn two_diff_e<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
    two_sum_e(simd, a, simd.f64s_neg(b))
}

#[inline(always)]
fn two_prod<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
    let p = simd.f64s_mul(a, b);
    let err = simd.f64s_mul_add(a, b, simd.f64s_neg(p));

    (p, err)
}

pub mod double {
    use super::*;

    #[inline(always)]
    pub fn simd_add<S: Simd>(simd: S, a: Double<S::f64s>, b: Double<S::f64s>) -> Double<S::f64s> {
        let (s, e) = two_sum(simd, a.0, b.0);
        let e = simd.f64s_add(e, simd.f64s_add(a.1, b.1));
        let (s, e) = quick_two_sum(simd, s, e);
        Double(s, e)
    }

    #[inline(always)]
    pub fn simd_sub<S: Simd>(simd: S, a: Double<S::f64s>, b: Double<S::f64s>) -> Double<S::f64s> {
        let (s, e) = two_diff_e(simd, a.0, b.0);
        let e = simd.f64s_add(e, a.1);
        let e = simd.f64s_sub(e, b.1);
        let (s, e) = quick_two_sum(simd, s, e);
        Double(s, e)
    }

    #[inline(always)]
    pub fn simd_neg<S: Simd>(simd: S, a: Double<S::f64s>) -> Double<S::f64s> {
        Double(simd.f64s_neg(a.0), simd.f64s_neg(a.1))
    }

    #[inline(always)]
    pub fn simd_mul<S: Simd>(simd: S, a: Double<S::f64s>, b: Double<S::f64s>) -> Double<S::f64s> {
        let (p1, p2) = two_prod(simd, a.0, b.0);
        let p2 = simd.f64s_add(
            p2,
            simd.f64s_add(simd.f64s_mul(a.0, b.1), simd.f64s_mul(a.1, b.0)),
        );
        let (p1, p2) = quick_two_sum(simd, p1, p2);
        Double(p1, p2)
    }

    #[inline(always)]
    fn simd_mul_f64<S: Simd>(simd: S, a: Double<S::f64s>, b: S::f64s) -> Double<S::f64s> {
        let (p1, p2) = two_prod(simd, a.0, b);
        let p2 = simd.f64s_add(p2, simd.f64s_mul(a.1, b));
        let (p1, p2) = quick_two_sum(simd, p1, p2);
        Double(p1, p2)
    }

    #[inline(always)]
    pub fn simd_select<S: Simd>(
        simd: S,
        mask: S::m64s,
        if_true: Double<S::f64s>,
        if_false: Double<S::f64s>,
    ) -> Double<S::f64s> {
        Double(
            simd.m64s_select_f64s(mask, if_true.0, if_false.0),
            simd.m64s_select_f64s(mask, if_true.1, if_false.1),
        )
    }

    #[inline]
    pub fn simd_div<S: Simd>(simd: S, a: Double<S::f64s>, b: Double<S::f64s>) -> Double<S::f64s> {
        simd.vectorize(
            #[inline(always)]
            || {
                let pos_zero = simd.f64s_splat(0.0);
                let pos_infty = simd.f64s_splat(f64::INFINITY);
                let sign_bit = simd.f64s_splat(-0.0);

                let a_sign = simd.f64s_and(a.0, sign_bit);
                let b_sign = simd.f64s_and(b.0, sign_bit);

                let combined_sign = simd.f64s_xor(a_sign, b_sign);

                let a_is_zero = simd_eq(simd, a, Double(pos_zero, pos_zero));
                let b_is_zero = simd_eq(simd, b, Double(pos_zero, pos_zero));
                let a_is_infty = simd_eq(
                    simd,
                    Double(simd.f64s_abs(a.0), simd.f64s_abs(a.1)),
                    Double(pos_infty, pos_infty),
                );
                let b_is_infty = simd_eq(
                    simd,
                    Double(simd.f64s_abs(b.0), simd.f64s_abs(b.1)),
                    Double(pos_infty, pos_infty),
                );

                let q1 = simd.f64s_div(a.0, b.0);
                let r = simd_mul_f64(simd, b, q1);

                let (s1, s2) = two_diff(simd, a.0, r.0);
                let s2 = simd.f64s_sub(s2, r.1);
                let s2 = simd.f64s_add(s2, a.1);

                let q2 = simd.f64s_div(simd.f64s_add(s1, s2), b.0);
                let (q0, q1) = quick_two_sum(simd, q1, q2);

                simd_select(
                    simd,
                    simd.m64s_and(b_is_zero, simd.m64s_not(a_is_zero)),
                    Double(
                        simd.f64s_or(combined_sign, pos_infty),
                        simd.f64s_or(combined_sign, pos_infty),
                    ),
                    simd_select(
                        simd,
                        simd.m64s_and(b_is_infty, simd.m64s_not(a_is_infty)),
                        Double(
                            simd.f64s_or(combined_sign, pos_zero),
                            simd.f64s_or(combined_sign, pos_zero),
                        ),
                        Double(q0, q1),
                    ),
                )
            },
        )
    }

    #[inline(always)]
    pub fn simd_abs<S: Simd>(simd: S, a: Double<S::f64s>) -> Double<S::f64s> {
        let is_negative = simd.f64s_less_than(a.0, simd.f64s_splat(0.0));
        Double(
            simd.m64s_select_f64s(is_negative, simd.f64s_neg(a.0), a.0),
            simd.m64s_select_f64s(is_negative, simd.f64s_neg(a.1), a.1),
        )
    }

    #[inline(always)]
    pub fn simd_less_than<S: Simd>(simd: S, a: Double<S::f64s>, b: Double<S::f64s>) -> S::m64s {
        let lt0 = simd.f64s_less_than(a.0, b.0);
        let eq0 = simd.f64s_equal(a.0, b.0);
        let lt1 = simd.f64s_less_than(a.1, b.1);
        simd.m64s_or(lt0, simd.m64s_and(eq0, lt1))
    }

    #[inline(always)]
    pub fn simd_less_than_or_equal<S: Simd>(
        simd: S,
        a: Double<S::f64s>,
        b: Double<S::f64s>,
    ) -> S::m64s {
        let lt0 = simd.f64s_less_than(a.0, b.0);
        let eq0 = simd.f64s_equal(a.0, b.0);
        let lt1 = simd.f64s_less_than_or_equal(a.1, b.1);
        simd.m64s_or(lt0, simd.m64s_and(eq0, lt1))
    }

    #[inline(always)]
    pub fn simd_greater_than<S: Simd>(simd: S, a: Double<S::f64s>, b: Double<S::f64s>) -> S::m64s {
        let lt0 = simd.f64s_greater_than(a.0, b.0);
        let eq0 = simd.f64s_equal(a.0, b.0);
        let lt1 = simd.f64s_greater_than(a.1, b.1);
        simd.m64s_or(lt0, simd.m64s_and(eq0, lt1))
    }

    #[inline(always)]
    pub fn simd_greater_than_or_equal<S: Simd>(
        simd: S,
        a: Double<S::f64s>,
        b: Double<S::f64s>,
    ) -> S::m64s {
        let lt0 = simd.f64s_greater_than(a.0, b.0);
        let eq0 = simd.f64s_equal(a.0, b.0);
        let lt1 = simd.f64s_greater_than_or_equal(a.1, b.1);
        simd.m64s_or(lt0, simd.m64s_and(eq0, lt1))
    }

    #[inline(always)]
    pub fn simd_eq<S: Simd>(simd: S, a: Double<S::f64s>, b: Double<S::f64s>) -> S::m64s {
        let eq0 = simd.f64s_equal(a.0, b.0);
        let eq1 = simd.f64s_equal(a.1, b.1);
        simd.m64s_and(eq0, eq1)
    }

    #[inline(always)]
    pub fn unpack<S: Simd>(simd: S, packed: Double<S::f64s>) -> Double<S::f64s> {
        let size = core::mem::size_of::<S::f64s>() / core::mem::size_of::<f64>();
        let _ = simd;
        match size {
            1 => packed,
            2 => {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if core::any::TypeId::of::<S>() == core::any::TypeId::of::<pulp::x86::V2>() {
                    let simd = unsafe { pulp::x86::V2::new_unchecked() };
                    let a: pulp::f64x2 = bytemuck::cast(packed.0);
                    let b: pulp::f64x2 = bytemuck::cast(packed.1);
                    let ab_lo = simd.sse2._mm_unpacklo_pd(pulp::cast(a), pulp::cast(b));
                    let ab_hi = simd.sse2._mm_unpackhi_pd(pulp::cast(a), pulp::cast(b));
                    return bytemuck::cast([ab_lo, ab_hi]);
                }
                {
                    let a: [f64; 2] = bytemuck::cast(packed.0);
                    let b: [f64; 2] = bytemuck::cast(packed.1);
                    bytemuck::cast([[a[0], b[0]], [a[1], b[1]]])
                }
            }
            4 => {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if core::any::TypeId::of::<S>() == core::any::TypeId::of::<pulp::x86::V3>() {
                    let simd = unsafe { pulp::x86::V3::new_unchecked() };
                    let a: pulp::f64x4 = bytemuck::cast(packed.0);
                    let b: pulp::f64x4 = bytemuck::cast(packed.1);
                    let ab_lo = simd.avx._mm256_unpacklo_pd(pulp::cast(a), pulp::cast(b));
                    let ab_hi = simd.avx._mm256_unpackhi_pd(pulp::cast(a), pulp::cast(b));
                    return bytemuck::cast([ab_lo, ab_hi]);
                }
                let a: [f64; 4] = bytemuck::cast(packed.0);
                let b: [f64; 4] = bytemuck::cast(packed.1);
                bytemuck::cast([[a[0], b[0], a[2], b[2]], [a[1], b[1], a[3], b[3]]])
            }
            8 => {
                let a: [f64; 8] = bytemuck::cast(packed.0);
                let b: [f64; 8] = bytemuck::cast(packed.1);
                bytemuck::cast([
                    [a[0], b[0], a[2], b[2], a[4], b[4], a[6], b[6]],
                    [a[1], b[1], a[3], b[3], a[5], b[5], a[7], b[7]],
                ])
            }
            _ => unimplemented!(),
        }
    }

    #[inline(always)]
    pub fn pack<S: Simd>(simd: S, unpacked: Double<S::f64s>) -> Double<S::f64s> {
        unpack(simd, unpacked)
    }
}

impl core::ops::Add for Double<f64> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        double::simd_add(Scalar::new(), self, rhs)
    }
}

impl core::ops::Sub for Double<f64> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        double::simd_sub(Scalar::new(), self, rhs)
    }
}

impl core::ops::Mul for Double<f64> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        double::simd_mul(Scalar::new(), self, rhs)
    }
}

impl core::ops::Mul<Pow2> for Double<f64> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Pow2) -> Self::Output {
        Self(self.0 * rhs.0, self.1 * rhs.0)
    }
}
impl core::ops::Mul<Double<f64>> for Pow2 {
    type Output = Double<f64>;

    #[inline(always)]
    fn mul(self, rhs: Double<f64>) -> Self::Output {
        Double(self.0 * rhs.0, self.0 * rhs.1)
    }
}

impl core::ops::Div for Double<f64> {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        double::simd_div(Scalar::new(), self, rhs)
    }
}

impl core::ops::AddAssign for Double<f64> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl core::ops::SubAssign for Double<f64> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl core::ops::MulAssign for Double<f64> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl core::ops::DivAssign for Double<f64> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl core::ops::Neg for Double<f64> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self(-self.0, -self.1)
    }
}

impl Double<f64> {
    /// 2.0^{-100}
    pub const EPSILON: Self = Self(7.888609052210118e-31, 0.0);
    /// 2.0^{-970}: precision below this value begins to degrade.
    pub const MIN_POSITIVE: Self = Self(1.0020841800044864e-292, 0.0);

    pub const MANTISSA_DIGITS: u32 = 100;

    pub const ZERO: Self = Self(0.0, 0.0);
    pub const NAN: Self = Self(f64::NAN, f64::NAN);
    pub const INFINITY: Self = Self(f64::INFINITY, 0.0);
    pub const NEG_INFINITY: Self = Self(-f64::INFINITY, 0.0);

    pub const LN_2: Self = Self(0.6931471805599453, 2.3190468138462996e-17);
    pub const FRAC_1_LN_2: Self = Self(1.4426950408889634, 2.0355273740931033e-17);
    pub const LN_10: Self = Self(2.302585092994046, -2.1707562233822494e-16);
    pub const FRAC_1_LN_10: Self = Self(0.4342944819032518, 1.098319650216765e-17);

    #[inline(always)]
    pub fn abs(self) -> Self {
        double::simd_abs(Scalar::new(), self)
    }

    pub fn recip(self) -> Self {
        double::simd_div(Scalar::new(), Self(1.0, 0.0), self)
    }

    pub fn sqrt(self) -> Self {
        if self == Self::ZERO {
            Self::ZERO
        } else if self < Self::ZERO {
            Self::NAN
        } else if self == Self::INFINITY {
            Self::INFINITY
        } else {
            let a = self;

            let x = a.0.sqrt().recip();
            let ax = Self(a.0 * x, 0.0);

            ax + (a - ax * ax) * Self(x * 0.5, 0.0)
        }
    }

    pub fn exp(self) -> Self {
        let value = self;
        let exp_max = 709.0;
        if value.0 <= -exp_max {
            return Self(0.0, 0.0);
        }
        if value.0 >= exp_max {
            return Self::INFINITY;
        }
        if value.0 == 0.0 {
            return Self(1.0, 0.0);
        }

        let shift = (value.0 / Self::LN_2.0 + 0.5).floor();

        let digits = Self::MANTISSA_DIGITS;
        let num_squares = 9;
        let num_terms = digits / num_squares;

        let scale = (1u32 << num_squares) as f64;
        let inv_scale = scale.recip();

        let r = (value - Self::LN_2 * Self(shift, 0.0)) * Pow2(inv_scale);

        let mut r_power = r * r;
        let mut iterate = r + r_power * Pow2(0.5);

        r_power *= r;

        let mut coefficient = Self(6.0, 0.0).recip();
        let mut term = coefficient * r_power;

        iterate += term;
        let tolerance = Self::EPSILON.0 * inv_scale;

        for j in 4..num_terms {
            r_power *= r;
            coefficient /= Self(j as f64, 0.0);
            term = coefficient * r_power;
            iterate += term;

            if term.0.abs() <= tolerance {
                break;
            }
        }

        for _ in 0..num_squares {
            iterate = iterate * iterate + iterate * Pow2(2.0);
        }

        iterate += Self(1.0, 0.0);
        let shift = 2.0f64.powi(shift as i32);

        iterate * Pow2(shift)
    }

    pub fn powi(self, pow: i32) -> Self {
        let inv = pow < 0;
        let mut pow = pow.unsigned_abs();
        let mut x = self;
        if inv {
            x = x.recip();
        }

        if pow == 0 {
            return Self(1.0, 0.0);
        }

        let mut y = Self(1.0, 0.0);
        while pow > 1 {
            if pow % 2 != 0 {
                y = x * y;
            }
            x = x * x;
            pow /= 2;
        }

        x * y
    }

    fn powf_impl(self, pow: Self) -> Self {
        if pow.0 == 0.0 {
            return Self(1.0, 0.0);
        }

        let mut value = self;
        if pow.0.floor() == pow.0 && pow.1.floor() == pow.1 {
            if ((pow.0 * 0.5).floor() == (pow.0 * 0.5)) == ((pow.1 * 0.5).floor() == (pow.1 * 0.5))
            {
                value = value.abs();
            }
        }
        (value.ln() * pow).exp()
    }

    pub fn powf(self, pow: impl Into<Self>) -> Self {
        self.powf_impl(pow.into())
    }

    pub fn ln(self) -> Self {
        let value = self;
        if value.0 < 0.0 {
            return Self::NAN;
        }
        if value.0 == 0.0 {
            return Self::NEG_INFINITY;
        }

        let mut x = Self(self.0.ln(), 0.0);

        x += value * (-x).exp();
        x -= Self(1.0, 0.0);
        x += value * (-x).exp();
        x -= Self(1.0, 0.0);

        x
    }

    pub fn log2(self) -> Self {
        self.ln() * Self::FRAC_1_LN_2
    }

    pub fn log10(self) -> Self {
        self.ln() * Self::FRAC_1_LN_10
    }
}

impl From<f64> for Double<f64> {
    #[inline]
    fn from(value: f64) -> Self {
        Self(value, 0.0)
    }
}

#[cfg(feature = "faer")]
pub mod entity;

#[cfg(test)]
mod tests {
    use super::*;
    use equator::assert;
    use rug::{ops::Pow, Float};

    const PREC: u32 = 1024;

    fn from_rug(value: &Float) -> Double<f64> {
        let x0 = value.to_f64();
        let x1 = Float::with_val(PREC, value - x0).to_f64();
        Double(x0, x1)
    }

    fn to_rug(value: Double<f64>) -> Float {
        Float::with_val(PREC, value.0) + Float::with_val(PREC, value.1)
    }

    #[test]
    fn test_div_edge_case() {
        assert_eq!(Double(1.0, 0.0) / Double(0.0, 0.0), Double::INFINITY);
        assert_eq!(Double(-1.0, 0.0) / Double(0.0, 0.0), Double::NEG_INFINITY);
        assert_eq!(Double(1.0, 0.0) / Double(-0.0, 0.0), Double::NEG_INFINITY);
        assert_eq!(Double(-1.0, 0.0) / Double(-0.0, 0.0), Double::INFINITY);

        assert_eq!(Double(1.0, 0.0) / Double::INFINITY, Double::ZERO);
        assert_eq!(Double(-1.0, 0.0) / Double::INFINITY, Double::ZERO);
        assert_eq!(Double(1.0, 0.0) / Double::NEG_INFINITY, Double::ZERO);
        assert_eq!(Double(-1.0, 0.0) / Double::NEG_INFINITY, Double::ZERO);
    }

    #[test]
    fn test_sqrt_edge_case() {
        assert_eq!(Double::INFINITY.sqrt(), Double::INFINITY);
    }

    #[test]
    fn test_cmp_edge_case() {
        assert_eq!(Double::INFINITY, Double::INFINITY);
        assert!(Double(1.0, 0.0) < Double::INFINITY);
        assert!(Double(1.0, 0.0) > Double::NEG_INFINITY);
        assert!(Double(1.0, 0.0) == Double(1.0, 0.0));
        assert!(Double(0.0, 0.0) == Double(0.0, 0.0));
        assert!(Double(-0.0, 0.0) == Double(0.0, 0.0));
        assert!(Double::NAN != Double::NAN);
    }

    #[test]
    fn test_two_sum() {
        for _ in 0..1000 {
            for scale in [1, 10, 100, 1000, 10000] {
                let a = rand::random::<f64>();
                let b = rand::random::<f64>() * scale as f64;
                let simd = pulp::Scalar::new();

                assert!(two_sum_e(simd, a, b) == two_sum(simd, a, b));
            }
        }
    }

    #[test]
    fn test_math() {
        dbg!(from_rug(&Float::with_val(PREC, 2.0f64).ln()));
        dbg!(from_rug(&Float::with_val(PREC, 10.0f64).ln()));
        dbg!(from_rug(&Float::with_val(PREC, 2.0f64).ln().recip()));
        dbg!(from_rug(&Float::with_val(PREC, 10.0f64).ln().recip()));

        let mut rng = rug::rand::RandState::new();
        for _ in 0..100 {
            let x = Float::with_val(PREC, Float::random_normal(&mut rng));
            let y = Float::with_val(PREC, Float::random_normal(&mut rng));

            let add_rug = Float::with_val(PREC, &x + &y);
            let sub_rug = Float::with_val(PREC, &x - &y);
            let mul_rug = Float::with_val(PREC, &x * &y);
            let div_rug = Float::with_val(PREC, &x / &y);
            let sqrt_rug = Float::with_val(PREC, x.clone().abs().sqrt());
            let exp_rug = Float::with_val(PREC, x.clone().exp());
            let ln_rug = Float::with_val(PREC, x.clone().abs().ln());
            let powi_rug = Float::with_val(PREC, x.clone().pow(5));
            let powf_rug = Float::with_val(PREC, x.clone().abs().pow(5.5));

            let add = from_rug(&x) + from_rug(&y);
            let sub = from_rug(&x) - from_rug(&y);
            let mul = from_rug(&x) * from_rug(&y);
            let div = from_rug(&x) / from_rug(&y);
            let sqrt = from_rug(&x).abs().sqrt();
            let exp = from_rug(&x).exp();
            let ln = from_rug(&x).abs().ln();
            let powi = from_rug(&x).powi(5);
            let powf = from_rug(&x).abs().powf(5.5);

            let err_add = from_rug(&Float::with_val(PREC, &add_rug - to_rug(add))).abs();
            let err_sub = from_rug(&Float::with_val(PREC, &sub_rug - to_rug(sub))).abs();
            let err_mul = from_rug(&Float::with_val(PREC, &mul_rug - to_rug(mul))).abs();
            let err_div = from_rug(&Float::with_val(PREC, &div_rug - to_rug(div))).abs();
            let err_sqrt = from_rug(&Float::with_val(PREC, &sqrt_rug - to_rug(sqrt))).abs();
            let err_exp = from_rug(&Float::with_val(PREC, &exp_rug - to_rug(exp))).abs();
            let err_ln = from_rug(&Float::with_val(PREC, &ln_rug - to_rug(ln))).abs();
            let err_powi = from_rug(&Float::with_val(PREC, &powi_rug - to_rug(powi))).abs();
            let err_powf = from_rug(&Float::with_val(PREC, &powf_rug - to_rug(powf))).abs();

            assert!(err_add / add.abs() < Double::EPSILON);
            assert!(err_sub / sub.abs() < Double::EPSILON);
            assert!(err_mul / mul.abs() < Double::EPSILON);
            assert!(err_div / div.abs() < Double::EPSILON);
            assert!(err_sqrt / sqrt.abs() < Double::EPSILON);
            assert!(err_exp / exp.abs() < Double::EPSILON);
            assert!(err_ln / ln.abs() < Double::EPSILON);
            assert!(err_powi / powi.abs() < Double::EPSILON);
            assert!(err_powf / powf.abs() < Double::EPSILON);
        }
    }
}
