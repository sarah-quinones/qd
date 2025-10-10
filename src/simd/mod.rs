use super::*;
use pulp::Simd;

pub extern crate pulp;

#[inline(always)]
fn quick_two_sum<S: Simd>(simd: S, large: S::f64s, small: S::f64s) -> Quad<S::f64s> {
	let s = simd.add_f64s(large, small);
	let e = simd.sub_f64s(small, simd.sub_f64s(s, large));
	Quad(s, e)
}

#[inline(always)]
fn two_sum<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> Quad<S::f64s> {
	let gt = simd.greater_than_f64s(simd.abs_f64s(a), simd.abs_f64s(b));
	let large = simd.select_f64s_m64s(gt, a, b);
	let small = simd.select_f64s_m64s(gt, b, a);
	quick_two_sum(simd, large, small)
}

#[inline(always)]
fn two_prod<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> Quad<S::f64s> {
	let p = simd.mul_f64s(a, b);
	let e = simd.mul_add_f64s(a, b, simd.neg_f64s(p));

	Quad(p, e)
}

#[inline(always)]
pub fn eq<S: Simd>(simd: S, a: Quad<S::f64s>, b: Quad<S::f64s>) -> S::m64s {
	simd.and_m64s(simd.equal_f64s(a.0, b.0), simd.equal_f64s(a.1, b.1))
}

#[inline(always)]
pub fn ne<S: Simd>(simd: S, a: Quad<S::f64s>, b: Quad<S::f64s>) -> S::m64s {
	simd.not_m64s(eq(simd, a, b))
}

#[inline(always)]
pub fn less_than<S: Simd>(simd: S, a: Quad<S::f64s>, b: Quad<S::f64s>) -> S::m64s {
	let lt0 = simd.less_than_f64s(a.0, b.0);
	let eq0 = simd.equal_f64s(a.0, b.0);
	let lt1 = simd.less_than_f64s(a.1, b.1);

	simd.or_m64s(lt0, simd.and_m64s(eq0, lt1))
}

#[inline(always)]
pub fn greater_than<S: Simd>(simd: S, a: Quad<S::f64s>, b: Quad<S::f64s>) -> S::m64s {
	let lt0 = simd.greater_than_f64s(a.0, b.0);
	let eq0 = simd.equal_f64s(a.0, b.0);
	let lt1 = simd.greater_than_f64s(a.1, b.1);

	simd.or_m64s(lt0, simd.and_m64s(eq0, lt1))
}

#[inline(always)]
pub fn less_than_or_equal<S: Simd>(simd: S, a: Quad<S::f64s>, b: Quad<S::f64s>) -> S::m64s {
	let lt0 = simd.less_than_or_equal_f64s(a.0, b.0);
	let eq0 = simd.equal_f64s(a.0, b.0);
	let lt1 = simd.less_than_or_equal_f64s(a.1, b.1);

	simd.or_m64s(lt0, simd.and_m64s(eq0, lt1))
}

#[inline(always)]
pub fn greater_than_or_equal<S: Simd>(simd: S, a: Quad<S::f64s>, b: Quad<S::f64s>) -> S::m64s {
	let lt0 = simd.greater_than_or_equal_f64s(a.0, b.0);
	let eq0 = simd.equal_f64s(a.0, b.0);
	let lt1 = simd.greater_than_or_equal_f64s(a.1, b.1);

	simd.or_m64s(lt0, simd.and_m64s(eq0, lt1))
}

#[inline(always)]
pub fn neg<S: Simd>(simd: S, a: Quad<S::f64s>) -> Quad<S::f64s> {
	Quad(simd.neg_f64s(a.0), simd.neg_f64s(a.1))
}

#[inline(always)]
pub fn add_accurate<S: Simd>(simd: S, a: Quad<S::f64s>, b: Quad<S::f64s>) -> Quad<S::f64s> {
	let Quad(s0, e1) = two_sum(simd, a.0, b.0);
	let Quad(s1, e2) = two_sum(simd, a.1, b.1);

	let e1 = simd.add_f64s(e1, s1);

	let Quad(s0, e1) = quick_two_sum(simd, s0, e1);
	let e1 = simd.add_f64s(e1, e2);

	quick_two_sum(simd, s0, e1)
}

#[inline(always)]
pub fn add_estimate<S: Simd>(simd: S, a: Quad<S::f64s>, b: Quad<S::f64s>) -> Quad<S::f64s> {
	let Quad(s0, e1) = two_sum(simd, a.0, b.0);
	let s1 = simd.add_f64s(a.1, b.1);
	let e1 = simd.add_f64s(e1, s1);

	quick_two_sum(simd, s0, e1)
}

#[inline(always)]
pub fn sub_accurate<S: Simd>(simd: S, a: Quad<S::f64s>, b: Quad<S::f64s>) -> Quad<S::f64s> {
	add_accurate(simd, a, neg(simd, b))
}

#[inline(always)]
pub fn sub_estimate<S: Simd>(simd: S, a: Quad<S::f64s>, b: Quad<S::f64s>) -> Quad<S::f64s> {
	add_estimate(simd, a, neg(simd, b))
}

#[inline(always)]
pub fn mul<S: Simd>(simd: S, a: Quad<S::f64s>, b: Quad<S::f64s>) -> Quad<S::f64s> {
	let Quad(p0, e1) = two_prod(simd, a.0, b.0);
	let e1 = simd.mul_add_f64s(a.0, b.1, e1);
	let e1 = simd.mul_add_f64s(a.1, b.0, e1);
	quick_two_sum(simd, p0, e1)
}

#[inline(always)]
pub fn select<S: Simd>(simd: S, mask: S::m64s, if_true: Quad<S::f64s>, if_false: Quad<S::f64s>) -> Quad<S::f64s> {
	Quad(
		simd.select_f64s_m64s(mask, if_true.0, if_false.0),
		simd.select_f64s_m64s(mask, if_true.1, if_false.1),
	)
}

#[inline(always)]
pub fn splat<S: Simd>(simd: S, a: Quad) -> Quad<S::f64s> {
	Quad(simd.splat_f64s(a.0), simd.splat_f64s(a.1))
}

#[inline(always)]
pub fn abs<S: Simd>(simd: S, a: Quad<S::f64s>) -> Quad<S::f64s> {
	let sign_bit = simd.and_f64s(a.0, simd.splat_f64s(-0.0));

	Quad(simd.xor_f64s(a.0, sign_bit), simd.xor_f64s(a.1, sign_bit))
}
