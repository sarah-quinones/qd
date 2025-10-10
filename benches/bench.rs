mod f128 {
	use aligned_vec::avec;
	use core::iter::zip;
	use qd::*;

	#[divan::bench(args = 8..25)]
	pub fn bench_add_scalar(bencher: divan::Bencher, n: u32) {
		let simd = pulp::Scalar::new();

		let size = 1 << n;
		let u0 = &*avec![0.0; size];
		let u1 = &*avec![0.0; size];
		let v0 = &*avec![0.0; size];
		let v1 = &*avec![0.0; size];
		let o0 = &mut *avec![0.0; size];
		let o1 = &mut *avec![0.0; size];

		bencher.bench_local(|| {
			let o0 = &mut *o0;
			let o1 = &mut *o1;

			for ((o0, o1), ((u0, u1), (v0, v1))) in zip(zip(o0, o1), zip(zip(u0, u1), zip(v0, v1))) {
				let u = Double(*u0, *u1);
				let v = Double(*v0, *v1);
				let o = double::simd_add(simd, u, v);
				*o0 = o.0;
				*o1 = o.1;
			}
		});
	}

	#[divan::bench(args = 8..25)]
	pub fn bench_mul_scalar(bencher: divan::Bencher, n: u32) {
		let simd = pulp::Scalar::new();

		let size = 1 << n;
		let u0 = &*avec![0.0; size];
		let u1 = &*avec![0.0; size];
		let v0 = &*avec![0.0; size];
		let v1 = &*avec![0.0; size];
		let o0 = &mut *avec![0.0; size];
		let o1 = &mut *avec![0.0; size];

		bencher.bench_local(|| {
			let o0 = &mut *o0;
			let o1 = &mut *o1;

			for ((o0, o1), ((u0, u1), (v0, v1))) in zip(zip(o0, o1), zip(zip(u0, u1), zip(v0, v1))) {
				let u = Double(*u0, *u1);
				let v = Double(*v0, *v1);
				let o = double::simd_mul(simd, u, v);
				*o0 = o.0;
				*o1 = o.1;
			}
		});
	}

	#[divan::bench(args = 8..25)]
	pub fn bench_sqrt(bencher: divan::Bencher, n: u32) {
		let size = 1 << n;
		let u0 = &*avec![1.0; size];
		let u1 = &*avec![0.0; size];
		let o0 = &mut *avec![0.0; size];
		let o1 = &mut *avec![0.0; size];

		bencher.bench_local(|| {
			let o0 = &mut *o0;
			let o1 = &mut *o1;

			for ((o0, o1), (u0, u1)) in zip(zip(o0, o1), zip(u0, u1)) {
				let u = Double(*u0, *u1);
				let o = u.sqrt();
				*o0 = o.0;
				*o1 = o.1;
			}
		});
	}

	#[divan::bench(args = 8..25)]
	pub fn bench_exp(bencher: divan::Bencher, n: u32) {
		let size = 1 << n;
		let u0 = &*avec![1.0; size];
		let u1 = &*avec![0.0; size];
		let o0 = &mut *avec![0.0; size];
		let o1 = &mut *avec![0.0; size];

		bencher.bench_local(|| {
			let o0 = &mut *o0;
			let o1 = &mut *o1;

			for ((o0, o1), (u0, u1)) in zip(zip(o0, o1), zip(u0, u1)) {
				let u = Double(*u0, *u1);
				let o = u.exp();
				*o0 = o.0;
				*o1 = o.1;
			}
		});
	}
}

#[cfg(target_arch = "x86_64")]
mod f128_simd {
	use aligned_vec::avec;
	use core::iter::zip;
	use qd::*;

	#[divan::bench(args = 8..25)]
	pub fn bench_add(bencher: divan::Bencher, n: u32) {
		if let Some(simd) = pulp::x86::V4::try_new() {
			use pulp::f64x8;
			let size = 1 << n;
			let u0 = &*avec![0.0; size];
			let u1 = &*avec![0.0; size];
			let v0 = &*avec![0.0; size];
			let v1 = &*avec![0.0; size];
			let o0 = &mut *avec![0.0; size];
			let o1 = &mut *avec![0.0; size];

			bencher.bench_local(|| {
				simd.vectorize(
					#[inline(always)]
					|| {
						let u0: &[f64x8] = bytemuck::cast_slice(u0);
						let u1: &[f64x8] = bytemuck::cast_slice(u1);
						let v0: &[f64x8] = bytemuck::cast_slice(v0);
						let v1: &[f64x8] = bytemuck::cast_slice(v1);
						let o0: &mut [f64x8] = bytemuck::cast_slice_mut(o0);
						let o1: &mut [f64x8] = bytemuck::cast_slice_mut(o1);

						for ((o0, o1), ((u0, u1), (v0, v1))) in zip(zip(o0, o1), zip(zip(u0, u1), zip(v0, v1))) {
							let u = Double(*u0, *u1);
							let v = Double(*v0, *v1);
							let o = double::simd_add(simd, u, v);
							*o0 = o.0;
							*o1 = o.1;
						}
					},
				);
			});
		}
	}

	#[divan::bench(args = 8..25)]
	pub fn bench_mul(bencher: divan::Bencher, n: u32) {
		if let Some(simd) = pulp::x86::V4::try_new() {
			use pulp::f64x8;
			let size = 1 << n;
			let u0 = &*avec![0.0; size];
			let u1 = &*avec![0.0; size];
			let v0 = &*avec![0.0; size];
			let v1 = &*avec![0.0; size];
			let o0 = &mut *avec![0.0; size];
			let o1 = &mut *avec![0.0; size];

			bencher.bench_local(|| {
				simd.vectorize(
					#[inline(always)]
					|| {
						let u0: &[f64x8] = bytemuck::cast_slice(u0);
						let u1: &[f64x8] = bytemuck::cast_slice(u1);
						let v0: &[f64x8] = bytemuck::cast_slice(v0);
						let v1: &[f64x8] = bytemuck::cast_slice(v1);
						let o0: &mut [f64x8] = bytemuck::cast_slice_mut(o0);
						let o1: &mut [f64x8] = bytemuck::cast_slice_mut(o1);

						for ((o0, o1), ((u0, u1), (v0, v1))) in zip(zip(o0, o1), zip(zip(u0, u1), zip(v0, v1))) {
							let u = Double(*u0, *u1);
							let v = Double(*v0, *v1);
							let o = double::simd_mul(simd, u, v);
							*o0 = o.0;
							*o1 = o.1;
						}
					},
				);
			});
		}
	}
}

fn main() {
	divan::main();
}
