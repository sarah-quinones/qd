// mostly taken from libm
const fn scalbn(x: f64, mut n: i32) -> f64 {
	let x1p1023 = f64::from_bits(0x7FE0000000000000); // 0x1p1023 === 2 ^ 1023

	let x1p53 = f64::from_bits(0x4340000000000000); // 0x1p53 === 2 ^ 53

	let x1p_1022 = f64::from_bits(0x0010000000000000); // 0x1p-1022 === 2 ^ (-1022)

	let mut y = x;

	if n > 1023 {
		y *= x1p1023;

		n -= 1023;

		if n > 1023 {
			y *= x1p1023;

			n -= 1023;

			if n > 1023 {
				n = 1023;
			}
		}
	} else if n < -1022 {
		/* make sure final n < -53 to avoid double

		rounding in the subnormal range */

		y *= x1p_1022 * x1p53;

		n += 1022 - 53;

		if n < -1022 {
			y *= x1p_1022 * x1p53;

			n += 1022 - 53;

			if n < -1022 {
				n = -1022;
			}
		}
	}

	y * f64::from_bits(((0x3FF + n) as u64) << 52)
}

const ZEROINFNAN: i32 = 0x7FF - 0x3FF - 52 - 1;

struct Num {
	m: u64,

	e: i32,

	sign: i32,
}

const fn normalize(x: f64) -> Num {
	let x1p63: f64 = f64::from_bits(0x43E0000000000000); // 0x1p63 === 2 ^ 63

	let mut ix: u64 = x.to_bits();

	let mut e: i32 = (ix >> 52) as i32;

	let sign: i32 = e & 0x800;

	e &= 0x7FF;

	if e == 0 {
		ix = (x * x1p63).to_bits();

		e = (ix >> 52) as i32 & 0x7FF;

		e = if e != 0 { e - 63 } else { 0x800 };
	}

	ix &= (1 << 52) - 1;

	ix |= 1 << 52;

	ix <<= 1;

	e -= 0x3FF + 52 + 1;

	Num { m: ix, e, sign }
}

#[inline]
const fn mul(x: u64, y: u64) -> (u64, u64) {
	let t = (x as u128).wrapping_mul(y as u128);
	((t >> 64) as u64, t as u64)
}

pub const fn fma(x: f64, y: f64, z: f64) -> f64 {
	let x1p63: f64 = f64::from_bits(0x43E0000000000000); // 0x1p63 === 2 ^ 63

	let x0_ffffff8p_63 = f64::from_bits(0x3BFFFFFFF0000000); // 0x0.ffffff8p-63

	/* normalize so top 10bits and last bit are 0 */

	let nx = normalize(x);

	let ny = normalize(y);

	let nz = normalize(z);

	if nx.e >= ZEROINFNAN || ny.e >= ZEROINFNAN {
		return x * y + z;
	}

	if nz.e >= ZEROINFNAN {
		if nz.e > ZEROINFNAN {
			/* z==0 */

			return x * y + z;
		}

		return z;
	}

	/* mul: r = x*y */

	let zhi: u64;

	let zlo: u64;

	let (mut rhi, mut rlo) = mul(nx.m, ny.m);

	/* either top 20 or 21 bits of rhi and last 2 bits of rlo are 0 */

	/* align exponents */

	let mut e: i32 = nx.e + ny.e;

	let mut d: i32 = nz.e - e;

	/* shift bits z<<=kz, r>>=kr, so kz+kr == d, set e = e+kr (== ez-kz) */

	if d > 0 {
		if d < 64 {
			zlo = nz.m << d;

			zhi = nz.m >> (64 - d);
		} else {
			zlo = 0;

			zhi = nz.m;

			e = nz.e - 64;

			d -= 64;

			if d == 0 {
			} else if d < 64 {
				rlo = rhi << (64 - d) | rlo >> d | ((rlo << (64 - d)) != 0) as u64;

				rhi = rhi >> d;
			} else {
				rlo = 1;

				rhi = 0;
			}
		}
	} else {
		zhi = 0;

		d = -d;

		if d == 0 {
			zlo = nz.m;
		} else if d < 64 {
			zlo = nz.m >> d | ((nz.m << (64 - d)) != 0) as u64;
		} else {
			zlo = 1;
		}
	}

	/* add */

	let mut sign: i32 = nx.sign ^ ny.sign;

	let samesign: bool = (sign ^ nz.sign) == 0;

	let mut nonzero: i32 = 1;

	if samesign {
		/* r += z */

		rlo = rlo.wrapping_add(zlo);

		rhi += zhi + (rlo < zlo) as u64;
	} else {
		/* r -= z */

		let (res, borrow) = rlo.overflowing_sub(zlo);

		rlo = res;

		rhi = rhi.wrapping_sub(zhi.wrapping_add(borrow as u64));

		if (rhi >> 63) != 0 {
			rlo = (rlo as i64).wrapping_neg() as u64;

			rhi = (rhi as i64).wrapping_neg() as u64 - (rlo != 0) as u64;

			sign = (sign == 0) as i32;
		}

		nonzero = (rhi != 0) as i32;
	}

	/* set rhi to top 63bit of the result (last bit is sticky) */

	if nonzero != 0 {
		e += 64;

		d = rhi.leading_zeros() as i32 - 1;

		/* note: d > 0 */

		rhi = rhi << d | rlo >> (64 - d) | ((rlo << d) != 0) as u64;
	} else if rlo != 0 {
		d = rlo.leading_zeros() as i32 - 1;

		if d < 0 {
			rhi = rlo >> 1 | (rlo & 1);
		} else {
			rhi = rlo << d;
		}
	} else {
		/* exact +-0 */

		return x * y + z;
	}

	e -= d;

	/* convert to double */

	let mut i: i64 = rhi as i64; /* i is in [1<<62,(1<<63)-1] */

	if sign != 0 {
		i = -i;
	}

	let mut r: f64 = i as f64; /* |r| is in [0x1p62,0x1p63] */

	if e < -1022 - 62 {
		/* result is subnormal before rounding */

		if e == -1022 - 63 {
			let mut c: f64 = x1p63;

			if sign != 0 {
				c = -c;
			}

			if r == c {
				/* min normal after rounding, underflow depends

				on arch behaviour which can be imitated by

				a double to float conversion */

				let fltmin: f32 = (x0_ffffff8p_63 * f32::MIN_POSITIVE as f64 * r) as f32;

				return f64::MIN_POSITIVE / f32::MIN_POSITIVE as f64 * fltmin as f64;
			}

			/* one bit is lost when scaled, add another top bit to

			only round once at conversion if it is inexact */

			if (rhi << 53) != 0 {
				i = (rhi >> 1 | (rhi & 1) | 1 << 62) as i64;

				if sign != 0 {
					i = -i;
				}

				r = i as f64;

				r = 2. * r - c; /* remove top bit */

				/* raise underflow portably, such that it

				cannot be optimized away */

				{
					let tiny: f64 = f64::MIN_POSITIVE / f32::MIN_POSITIVE as f64 * r;

					r += (tiny * tiny) * (r - r);
				}
			}
		} else {
			/* only round once when scaled */

			d = 10;

			i = ((rhi >> d | ((rhi << (64 - d)) != 0) as u64) << d) as i64;

			if sign != 0 {
				i = -i;
			}

			r = i as f64;
		}
	}

	scalbn(r, e)
}

pub const fn sqrt(x: f64) -> f64 {
	const SIGN_MASK: u64 = 0x80000000_00000000;
	const MANTISSA_MASK: u64 = 0x7FF00000_00000000;

	let x0 = x.to_bits();

	/* take care of Inf and NaN */

	if (x0 & MANTISSA_MASK) == MANTISSA_MASK {
		return x * x + x; /* sqrt(NaN)=NaN, sqrt(+inf)=+inf, sqrt(-inf)=sNaN */
	}

	/* take care of zero */
	if x0 as i64 <= 0 {
		if x0 & !SIGN_MASK == 0 {
			return x; /* sqrt(+-0) = +-0 */
		}

		if (x0 as i64) < 0 {
			return (x - x) / (x - x); /* sqrt(-ve) = sNaN */
		}
	}

	let biased_exponent = x0 >> 52;

	let mut exp;
	let mantissa;

	if biased_exponent == 0 {
		exp = -1022i64;

		// we need the top bit to be at position 52
		let top_zeros = x0.leading_zeros();
		let top_zeros_52 = const { (1u64 << 52).leading_zeros() };
		let shift = top_zeros - top_zeros_52;

		mantissa = x0 << shift;
		exp -= shift as i64;
	} else {
		exp = biased_exponent as i64 - 1023;

		mantissa = (x0 & ((1 << 52) - 1)) | (1 << 52);
	}

	let sqrt_exp = (exp + 2048) / 2 - 1024;
	let sqrt_biased_exponent = (sqrt_exp + 1023) as u64;

	// at least (1 << 106)
	let shifted_mantissa = (mantissa as u128) << (54 + (exp & 1));

	// at least (1 << 53)
	let sqrt_mantissa = shifted_mantissa.isqrt() as u64 as u128;
	let sqr = sqrt_mantissa * sqrt_mantissa;

	let sqrt_mantissa = sqrt_mantissa as u64;

	let round_up = sqrt_mantissa & 1;
	let mut sqrt_mantissa = sqrt_mantissa >> 1;

	if shifted_mantissa != sqr && round_up == 1 {
		sqrt_mantissa += 1;
	}

	f64::from_bits((sqrt_mantissa & ((1 << 52) - 1)) | (sqrt_biased_exponent << 52))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_sqrt() {
		for x in (0..400).map(|i| 1.0 + i as f64 / 100.0).chain([
			f64::MIN_POSITIVE,
			f64::MIN_POSITIVE / 2.0,
			1.0f64.next_down(),
			1.0f64.next_up(),
			1.0f64.next_down().next_down(),
			1.0f64.next_up().next_up(),
		]) {
			assert_eq!(sqrt(x), f64::sqrt(x));
		}
	}
}
