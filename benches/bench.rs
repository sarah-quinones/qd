use aligned_vec::avec;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;
use qd::*;

pub fn bench_f128(c: &mut Criterion) {
    use pulp::f64x4;
    if let Some(simd) = pulp::x86::V3::try_new() {
        for n in 10..25 {
            let size = 1 << n;
            let mut v0 = avec![1.0; size];
            let mut v1 = avec![1.0; size];
            let mut v = avec![1.0; size * 2];

            c.bench_function(&format!("split-{n}"), |bencher| {
                bencher.iter(|| {
                    simd.vectorize(
                        #[inline(always)]
                        || {
                            let v0: &mut [f64x4] = bytemuck::cast_slice_mut(&mut v0);
                            let v1: &mut [f64x4] = bytemuck::cast_slice_mut(&mut v1);

                            for (v0, v1) in core::iter::zip(&mut *v0, &mut *v1) {
                                let mut x = Double(*v0, *v1);
                                x = double::simd_add(simd, x, x);
                                x = double::simd_add(simd, x, x);
                                x = double::simd_add(simd, x, x);
                                *v0 = x.0;
                                *v1 = x.1;
                            }

                            core::hint::black_box((v0, v1));
                        },
                    )
                })
            });

            c.bench_function(&format!("packed-{n}"), |bencher| {
                bencher.iter(|| {
                    simd.vectorize(
                        #[inline(always)]
                        || {
                            let v: &mut [Double<f64x4>] = bytemuck::cast_slice_mut(&mut v);

                            for v in &mut *v {
                                let mut x = double::unpack(simd, *v);
                                x = double::simd_add(simd, x, x);
                                x = double::simd_add(simd, x, x);
                                x = double::simd_add(simd, x, x);
                                *v = double::pack(simd, x);
                            }

                            core::hint::black_box(v);
                        },
                    )
                })
            });
        }
    }
}

criterion_group!(bench, bench_f128);
criterion_main!(bench);
