#![cfg_attr(not(test), no_std)]

#[cfg(feature = "std")]
extern crate std;

use core::cmp::Ordering;
use core::ops::*;

use bytemuck::{Pod, Zeroable};

mod const_fma;
mod const_imp;

#[cfg(not(feature = "std"))]
macro_rules! pick {
	($libm: expr, $std: expr) => {
		$libm
	};
}

#[cfg(feature = "std")]
macro_rules! pick {
	($libm: expr, $std: expr) => {
		$std
	};
}

mod imp {
	use super::*;

	pub use super::const_imp::*;

	pub fn fma(a: f64, b: f64, c: f64) -> f64 {
		pick!(libm::fma, f64::mul_add)(a, b, c)
	}

	pub fn sqrt(a: f64) -> f64 {
		pick!(libm::sqrt, f64::sqrt)(a)
	}

	pub fn two_prod(a: f64, b: f64) -> Quad {
		let p = a * b;
		let e = fma(a, b, -p);
		Quad(p, e)
	}

	pub fn mul(a: Quad, b: Quad) -> Quad {
		let Quad(p0, e1) = two_prod(a.0, b.0);
		let e1 = fma(a.0, b.1, e1);
		let e1 = fma(a.1, b.0, e1);

		const_imp::quick_two_sum(p0, e1)
	}
}

pub mod simd;

/// extended precision floating point type.
///
/// math operations assume that `self.1.abs() < self.0.abs() * ulp / 2.0`
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
#[repr(C)]
pub struct Quad<T = f64>(pub T, pub T);

impl Quad {
	pub const DIGITS: u32 = 31;
	pub const EPSILON: Self = Self(f64::EPSILON * f64::EPSILON, 0.0);
	pub const FRAC_1_LN_10: Self = Self(0.4342944819032518, 1.098319650216765e-17);
	pub const FRAC_1_LN_2: Self = Self(1.4426950408889634, 2.0355273740931033e-17);
	pub const INFINITY: Self = Self(f64::INFINITY, 0.0);
	pub const LN_10: Self = Self(2.302585092994046, -2.1707562233822494e-16);
	pub const LN_2: Self = Self(0.6931471805599453, 2.3190468138462996e-17);
	pub const MANTISSA_DIGITS: u32 = 105;
	pub const MAX: Self = Self(f64::MAX, f64::MAX * f64::EPSILON / 2.0);
	pub const MAX_EXP: i32 = f64::MAX_EXP;
	pub const MIN: Self = Self(-Self::MAX.0, -Self::MAX.1);
	pub const MIN_EXP: i32 = f64::MIN_EXP;
	pub const MIN_POSITIVE: Self = Self(f64::MIN_POSITIVE, 0.0);
	pub const NAN: Self = Self(f64::NAN, f64::NAN);
	pub const NEG_INFINITY: Self = Self(f64::NEG_INFINITY, 0.0);
	pub const ONE: Self = Self(1.0, 0.0);
	pub const PI: Self = Self(3.141592653589793, 1.2246467991473532e-16);
	pub const RADIX: u32 = 2;
	pub const ZERO: Self = Self(0.0, 0.0);

	#[inline(always)]
	pub const fn from_f64(value: f64) -> Self {
		Quad(value, 0.0)
	}

	pub const fn add_estimate(self, rhs: Self) -> Self {
		const_imp::add_estimate(self, rhs)
	}

	pub const fn add_accurate(self, rhs: Self) -> Self {
		const_imp::add_accurate(self, rhs)
	}

	pub const fn sub_estimate(self, rhs: Self) -> Self {
		const_imp::add_estimate(self, rhs.neg())
	}

	pub const fn sub_accurate(self, rhs: Self) -> Self {
		const_imp::add_accurate(self, rhs.neg())
	}

	pub const fn neg(self) -> Self {
		Quad(-self.0, -self.1)
	}

	pub const fn abs(self) -> Self {
		const_imp::abs(self)
	}

	pub fn mul(self, rhs: Self) -> Self {
		imp::mul(self, rhs)
	}

	pub const fn const_mul(self, rhs: Self) -> Self {
		const_imp::mul(self, rhs)
	}

	pub const fn const_div(self, rhs: Self) -> Self {
		let mut quotient = Quad(self.0 / rhs.0, 0.0);
		let mut r = self.sub_accurate(rhs.const_mul(quotient));
		quotient.1 = r.0 / rhs.0;
		r = r.sub_accurate(rhs.const_mul(Quad(quotient.1, 0.0)));

		let update = r.0 / rhs.0;

		quotient = const_imp::quick_two_sum(quotient.0, quotient.1);
		quotient = quotient.add_accurate(Quad(update, 0.0));
		quotient
	}

	pub const fn const_recip(self) -> Self {
		Self::ONE.const_div(self)
	}

	pub fn recip(self) -> Self {
		Self::ONE.div(self)
	}

	pub fn div(self, rhs: Self) -> Self {
		self.__div__(rhs)
	}

	fn __mul__(self, rhs: Self) -> Self {
		imp::mul(self, rhs)
	}

	fn __div__(self, rhs: Self) -> Self {
		let mut quotient = Quad(self.0 / rhs.0, 0.0);
		let mut r = self.sub_accurate(rhs.mul(quotient));
		quotient.1 = r.0 / rhs.0;
		r = r.sub_accurate(rhs.mul(Quad(quotient.1, 0.0)));

		let update = r.0 / rhs.0;

		quotient = imp::quick_two_sum(quotient.0, quotient.1);
		quotient = quotient.add_accurate(Quad(update, 0.0));
		quotient
	}

	pub const fn eq(self, rhs: Self) -> bool {
		self.0 == rhs.0 && self.1 == rhs.1
	}

	pub const fn ne(self, rhs: Self) -> bool {
		self.0 != rhs.0 || self.1 != rhs.1
	}

	pub const fn is_nan(self) -> bool {
		self.0.is_nan() || self.1.is_nan()
	}

	pub const fn is_finite(self) -> bool {
		self.0.is_finite() && self.1.is_finite()
	}

	pub const fn partial_cmp(self, rhs: Self) -> Option<Ordering> {
		if self.is_nan() || rhs.is_nan() {
			return None;
		}

		if self.0 < rhs.0 {
			Some(Ordering::Less)
		} else if self.0 > rhs.0 {
			Some(Ordering::Greater)
		} else {
			if self.1 < rhs.1 {
				Some(Ordering::Less)
			} else if self.1 > rhs.1 {
				Some(Ordering::Greater)
			} else {
				Some(Ordering::Equal)
			}
		}
	}

	pub const fn lt(self, rhs: Self) -> bool {
		self.0 < rhs.0 || (self.0 == rhs.0 && self.1 < rhs.1)
	}

	pub const fn le(self, rhs: Self) -> bool {
		self.0 <= rhs.0 || (self.0 == rhs.0 && self.1 <= rhs.1)
	}

	pub const fn gt(self, rhs: Self) -> bool {
		rhs.lt(self)
	}

	pub const fn ge(self, rhs: Self) -> bool {
		rhs.le(self)
	}

	pub const fn const_sqrt(self) -> Self {
		if self.0 == 0.0 {
			return Self::ZERO;
		}

		let mut iterate;
		{
			let inv_sqrt = 1.0 / const_imp::sqrt(self.0);
			let left = self.0 * inv_sqrt;
			let right = self.sub_accurate(const_imp::two_prod(left, left)).0 * (inv_sqrt / 2.0);
			iterate = const_imp::two_sum(left, right);
		}
		{
			iterate = Self::ONE.const_div(iterate);
			let left = self.0 * iterate.0;
			let right = self.sub_accurate(const_imp::two_prod(left, left)).0 * (iterate.0 / 2.0);
			iterate = const_imp::two_sum(left, right);
		}
		iterate
	}

	pub fn sqrt(self) -> Self {
		if self.0 == 0.0 {
			return Self::ZERO;
		}

		let mut iterate;
		{
			let inv_sqrt = 1.0 / imp::sqrt(self.0);
			let left = self.0 * inv_sqrt;
			let right = self.sub_accurate(imp::two_prod(left, left)).0 * (inv_sqrt / 2.0);
			iterate = imp::two_sum(left, right);
		}
		{
			iterate = Self::ONE.div(iterate);
			let left = self.0 * iterate.0;
			let right = self.sub_accurate(imp::two_prod(left, left)).0 * (iterate.0 / 2.0);
			iterate = imp::two_sum(left, right);
		}
		iterate
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

		let shift = pick!(libm::floor, f64::floor)(value.0 / Self::LN_2.0 + 0.5);

		let num_squares = 9;
		let num_terms = 9;

		let scale = (1u32 << num_squares) as f64;
		let inv_scale = scale.recip();

		let r = (value - Self::LN_2 * Self(shift, 0.0)) * Self::from_f64(inv_scale);

		let mut r_power = r * r;
		let mut iterate = r + r_power * Self::from_f64(0.5);

		r_power = r_power * r;

		let mut coefficient = Self(6.0, 0.0).recip();
		let mut term = coefficient * r_power;

		iterate = iterate + term;
		let tolerance = Self::EPSILON.0 * inv_scale;

		for j in 4..num_terms {
			r_power = r_power * r;
			coefficient = coefficient / Self(j as f64, 0.0);
			term = coefficient * r_power;
			iterate = iterate + term;

			if const_imp::fabs(term.0) <= tolerance {
				break;
			}
		}

		for _ in 0..num_squares {
			iterate = iterate * iterate + iterate * Self::from_f64(2.0);
		}

		iterate = iterate + Self(1.0, 0.0);
		let shift = pick!(libm::pow, f64::powi)(2.0f64, shift as _);

		iterate * Self::from_f64(shift)
	}

	pub fn ln(self) -> Self {
		let value = self;
		if value.0 < 0.0 {
			return Self::NAN;
		}
		if value.0 == 0.0 {
			return Self::NEG_INFINITY;
		}

		let mut x = Self(pick!(libm::log, f64::ln)(self.0), 0.0);

		x = x + value * (-x).exp();
		x = x - Self(1.0, 0.0);

		x
	}

	pub fn log2(self) -> Self {
		self.ln() * Self::FRAC_1_LN_2
	}

	pub fn log10(self) -> Self {
		self.ln() * Self::FRAC_1_LN_10
	}

	pub fn trunc(self) -> Self {
		let trunc = pick!(libm::trunc, f64::trunc);
		let trunc = Self(trunc(self.0), trunc(self.1));

		if trunc.0 == self.0 { trunc } else { Quad(trunc.0, 0.0) }
	}
}

macro_rules! impl_op {
	({$(
		impl $op:ident for $ty:ty {
			fn $op_fn:ident($self:ident, $rhs:ident: Self) -> Self::Output $body:block
		}
	)*}) => {$(
		impl $op for $ty {
			type Output = $ty;
			#[inline(always)]
			fn $op_fn($self, $rhs: Self) -> Self::Output $body
		}
		impl $op<&$ty> for $ty {
			type Output = $ty;
			#[inline(always)]
			fn $op_fn($self, $rhs: &$ty) -> Self::Output { let $rhs = *$rhs; $body}
		}
		impl $op<$ty> for &$ty {
			type Output = $ty;
			#[inline(always)]
			fn $op_fn($self, $rhs: $ty) -> Self::Output $body
		}
		impl $op<&$ty> for &$ty {
			type Output = $ty;
			#[inline(always)]
			fn $op_fn($self, $rhs: &$ty) -> Self::Output { let $rhs = *$rhs; $body}
		}
	)*};
}

macro_rules! impl_assign_op {
	({$(
		impl $op:ident for $ty:ty {
			fn $op_fn:ident(&mut $self:ident, $rhs:ident: Self) $body:block
		}
	)*}) => {$(
		impl $op for $ty {
			#[inline(always)]
			fn $op_fn(&mut $self, $rhs: Self) $body
		}
		impl $op<&$ty> for $ty {
			#[inline(always)]
			fn $op_fn(&mut $self, $rhs: &$ty) { let $rhs = *$rhs; $body}
		}
	)*};
}

impl Neg for Quad {
	type Output = Quad;

	fn neg(self) -> Self::Output {
		self.neg()
	}
}
impl Neg for &Quad {
	type Output = Quad;

	fn neg(self) -> Self::Output {
		(*self).neg()
	}
}

impl_op!({
	impl Add for Quad {
		fn add(self, rhs: Self) -> Self::Output {
			self.add_estimate(rhs)
		}
	}
	impl Sub for Quad {
		fn sub(self, rhs: Self) -> Self::Output {
			self.sub_estimate(rhs)
		}
	}
	impl Mul for Quad {
		fn mul(self, rhs: Self) -> Self::Output {
			self.__mul__(rhs)
		}
	}
	impl Div for Quad {
		fn div(self, rhs: Self) -> Self::Output {
			self.__div__(rhs)
		}
	}
	impl Rem for Quad {
		fn rem(self, rhs: Self) -> Self::Output {
			self - (self / rhs).trunc() * rhs
		}
	}
});

impl_assign_op!({
	impl AddAssign for Quad {
		fn add_assign(&mut self, rhs: Self) {
			*self = *self + rhs
		}
	}
	impl SubAssign for Quad {
		fn sub_assign(&mut self, rhs: Self) {
			*self = *self - rhs
		}
	}
	impl MulAssign for Quad {
		fn mul_assign(&mut self, rhs: Self) {
			*self = *self * rhs
		}
	}
	impl DivAssign for Quad {
		fn div_assign(&mut self, rhs: Self) {
			*self = *self / rhs
		}
	}
	impl RemAssign for Quad {
		fn rem_assign(&mut self, rhs: Self) {
			*self = *self % rhs
		}
	}
});

impl From<f64> for Quad {
	#[inline(always)]
	fn from(value: f64) -> Self {
		Quad(value, 0.0)
	}
}

impl num_traits::Zero for Quad {
	fn zero() -> Self {
		Self::ZERO
	}

	fn is_zero(&self) -> bool {
		*self == Self::ZERO
	}
}

impl num_traits::One for Quad {
	fn one() -> Self {
		Self::ONE
	}

	fn is_one(&self) -> bool {
		*self == Self::ONE
	}
}
impl num_traits::Num for Quad {
	type FromStrRadixErr = num_traits::ParseFloatError;

	fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
		let _ = str;
		let _ = radix;

		Err(num_traits::ParseFloatError {
			kind: num_traits::FloatErrorKind::Invalid,
		})
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_sqrt() {
		let x = Quad(3.5, 0.0);

		let y = x.sqrt();
		assert!(y * y - x < Quad::from(1e-30));

		const {
			let x = Quad(2.0, 0.0);
			let y = x.const_sqrt();
			assert!((y.const_mul(y).sub_accurate(x).abs()).lt(Quad::from_f64(1e-30)));
		};
	}

	#[test]
	fn test_rem() {
		let x = Quad::from(36.0) / Quad::from(10.0);
		assert!((x % Quad::from(0.5) - Quad::from(10.0).recip()).abs() < Quad::from(1e-30));
	}
}

unsafe impl<T: Zeroable> Zeroable for Quad<T> {}
unsafe impl<T: Pod> Pod for Quad<T> {}
