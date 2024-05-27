use super::*;
use faer_entity::*;
use num_traits::ParseFloatError;
pub struct DoubleGroup {
    __private: (),
}

impl ForType for DoubleGroup {
    type FaerOf<T> = Double<T>;
}
impl ForCopyType for DoubleGroup {
    type FaerOfCopy<T: Copy> = Double<T>;
}
impl ForDebugType for DoubleGroup {
    type FaerOfDebug<T: core::fmt::Debug> = Double<T>;
}

unsafe impl Entity for Double<f64> {
    type Unit = f64;
    type Index = u64;

    type SimdUnit<S: Simd> = S::f64s;
    type SimdMask<S: Simd> = S::m64s;
    type SimdIndex<S: Simd> = S::u64s;

    type Group = DoubleGroup;
    type Iter<I: Iterator> = Double<I>;

    type PrefixUnit<'a, S: Simd> = pulp::Prefix<'a, f64, S, S::m64s>;
    type SuffixUnit<'a, S: Simd> = pulp::Suffix<'a, f64, S, S::m64s>;
    type PrefixMutUnit<'a, S: Simd> = pulp::PrefixMut<'a, f64, S, S::m64s>;
    type SuffixMutUnit<'a, S: Simd> = pulp::SuffixMut<'a, f64, S, S::m64s>;

    const N_COMPONENTS: usize = 2;
    const UNIT: GroupCopyFor<Self, ()> = Double((), ());

    #[inline(always)]
    fn faer_first<T>(group: GroupFor<Self, T>) -> T {
        group.0
    }

    #[inline(always)]
    fn faer_from_units(group: GroupFor<Self, Self::Unit>) -> Self {
        group
    }

    #[inline(always)]
    fn faer_into_units(self) -> GroupFor<Self, Self::Unit> {
        self
    }

    #[inline(always)]
    fn faer_as_ref<T>(group: &GroupFor<Self, T>) -> GroupFor<Self, &T> {
        Double(&group.0, &group.1)
    }

    #[inline(always)]
    fn faer_as_mut<T>(group: &mut GroupFor<Self, T>) -> GroupFor<Self, &mut T> {
        Double(&mut group.0, &mut group.1)
    }

    #[inline(always)]
    fn faer_as_ptr<T>(group: *mut GroupFor<Self, T>) -> GroupFor<Self, *mut T> {
        unsafe {
            Double(
                core::ptr::addr_of_mut!((*group).0),
                core::ptr::addr_of_mut!((*group).1),
            )
        }
    }

    #[inline(always)]
    fn faer_map_impl<T, U>(
        group: GroupFor<Self, T>,
        f: &mut impl FnMut(T) -> U,
    ) -> GroupFor<Self, U> {
        Double((*f)(group.0), (*f)(group.1))
    }

    #[inline(always)]
    fn faer_zip<T, U>(
        first: GroupFor<Self, T>,
        second: GroupFor<Self, U>,
    ) -> GroupFor<Self, (T, U)> {
        Double((first.0, second.0), (first.1, second.1))
    }

    #[inline(always)]
    fn faer_unzip<T, U>(zipped: GroupFor<Self, (T, U)>) -> (GroupFor<Self, T>, GroupFor<Self, U>) {
        (
            Double(zipped.0 .0, zipped.1 .0),
            Double(zipped.0 .1, zipped.1 .1),
        )
    }

    #[inline(always)]
    fn faer_map_with_context<Ctx, T, U>(
        ctx: Ctx,
        group: GroupFor<Self, T>,
        f: &mut impl FnMut(Ctx, T) -> (Ctx, U),
    ) -> (Ctx, GroupFor<Self, U>) {
        let (ctx, x0) = (*f)(ctx, group.0);
        let (ctx, x1) = (*f)(ctx, group.1);
        (ctx, Double(x0, x1))
    }

    #[inline(always)]
    fn faer_into_iter<I: IntoIterator>(iter: GroupFor<Self, I>) -> Self::Iter<I::IntoIter> {
        Double(iter.0.into_iter(), iter.1.into_iter())
    }
}

impl num_traits::Zero for Double<f64> {
    #[inline]
    fn zero() -> Self {
        Self(0.0, 0.0)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self(0.0, 0.0)
    }
}

impl num_traits::One for Double<f64> {
    #[inline]
    fn one() -> Self {
        Self(1.0, 0.0)
    }
}
impl num_traits::Num for Double<f64> {
    type FromStrRadixErr = ParseFloatError;

    fn from_str_radix(_: &str, _: u32) -> Result<Self, Self::FromStrRadixErr> {
        todo!()
    }
}

unsafe impl Conjugate for Double<f64> {
    type Conj = Double<f64>;
    type Canonical = Double<f64>;
    #[inline(always)]
    fn canonicalize(self) -> Self::Canonical {
        self
    }
}

impl RealField for Double<f64> {
    #[inline(always)]
    fn faer_epsilon() -> Self {
        Self::EPSILON
    }
    #[inline(always)]
    fn faer_zero_threshold() -> Self {
        Self::MIN_POSITIVE
    }

    #[inline(always)]
    fn faer_div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn faer_usize_to_index(a: usize) -> Self::Index {
        a as _
    }

    #[inline(always)]
    fn faer_index_to_usize(a: Self::Index) -> usize {
        a as _
    }

    #[inline(always)]
    fn faer_max_index() -> Self::Index {
        Self::Index::MAX
    }

    #[inline(always)]
    fn faer_simd_less_than<S: Simd>(
        simd: S,
        a: SimdGroupFor<Self, S>,
        b: SimdGroupFor<Self, S>,
    ) -> Self::SimdMask<S> {
        double::simd_less_than(simd, a, b)
    }

    #[inline(always)]
    fn faer_simd_less_than_or_equal<S: Simd>(
        simd: S,
        a: SimdGroupFor<Self, S>,
        b: SimdGroupFor<Self, S>,
    ) -> Self::SimdMask<S> {
        double::simd_less_than_or_equal(simd, a, b)
    }

    #[inline(always)]
    fn faer_simd_greater_than<S: Simd>(
        simd: S,
        a: SimdGroupFor<Self, S>,
        b: SimdGroupFor<Self, S>,
    ) -> Self::SimdMask<S> {
        double::simd_greater_than(simd, a, b)
    }

    #[inline(always)]
    fn faer_simd_greater_than_or_equal<S: Simd>(
        simd: S,
        a: SimdGroupFor<Self, S>,
        b: SimdGroupFor<Self, S>,
    ) -> Self::SimdMask<S> {
        double::simd_greater_than_or_equal(simd, a, b)
    }

    #[inline(always)]
    fn faer_simd_select<S: Simd>(
        simd: S,
        mask: Self::SimdMask<S>,
        if_true: SimdGroupFor<Self, S>,
        if_false: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        double::simd_select(simd, mask, if_true, if_false)
    }

    #[inline(always)]
    fn faer_simd_index_select<S: Simd>(
        simd: S,
        mask: Self::SimdMask<S>,
        if_true: Self::SimdIndex<S>,
        if_false: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        simd.m64s_select_u64s(mask, if_true, if_false)
    }

    #[inline(always)]
    fn faer_simd_index_seq<S: Simd>(simd: S) -> Self::SimdIndex<S> {
        let _ = simd;
        pulp::cast_lossy([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15_u64])
    }

    #[inline(always)]
    fn faer_simd_index_splat<S: Simd>(simd: S, value: Self::Index) -> Self::SimdIndex<S> {
        simd.u64s_splat(value)
    }

    #[inline(always)]
    fn faer_simd_index_add<S: Simd>(
        simd: S,
        a: Self::SimdIndex<S>,
        b: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        simd.u64s_add(a, b)
    }

    #[inline(always)]
    fn faer_min_positive() -> Self {
        Self::MIN_POSITIVE
    }

    #[inline(always)]
    fn faer_min_positive_inv() -> Self {
        Self::MIN_POSITIVE.recip()
    }

    #[inline(always)]
    fn faer_min_positive_sqrt() -> Self {
        Self::MIN_POSITIVE.sqrt()
    }

    #[inline(always)]
    fn faer_min_positive_sqrt_inv() -> Self {
        Self::MIN_POSITIVE.sqrt().recip()
    }

    #[inline(always)]
    fn faer_simd_index_rotate_left<S: Simd>(
        simd: S,
        values: SimdIndexFor<Self, S>,
        amount: usize,
    ) -> SimdIndexFor<Self, S> {
        simd.u64s_rotate_left(values, amount)
    }

    #[inline(always)]
    fn faer_simd_abs<S: Simd>(simd: S, values: SimdGroupFor<Self, S>) -> SimdGroupFor<Self, S> {
        double::simd_abs(simd, values)
    }
}

impl ComplexField for Double<f64> {
    type Real = Double<f64>;
    type Simd = pulp::Arch;
    type ScalarSimd = pulp::ScalarArch;
    type PortableSimd = pulp::Arch;

    #[inline(always)]
    fn faer_sqrt(self) -> Self {
        self.sqrt()
    }

    #[inline(always)]
    fn faer_from_f64(value: f64) -> Self {
        Self(value, 0.0)
    }

    #[inline(always)]
    fn faer_add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline(always)]
    fn faer_sub(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline(always)]
    fn faer_mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn faer_neg(self) -> Self {
        -self
    }

    #[inline(always)]
    fn faer_inv(self) -> Self {
        self.recip()
    }

    #[inline(always)]
    fn faer_conj(self) -> Self {
        self
    }

    #[inline(always)]
    fn faer_scale_real(self, rhs: Self::Real) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn faer_scale_power_of_two(self, rhs: Self::Real) -> Self {
        Self(self.0 * rhs.0, self.1 * rhs.0)
    }

    #[inline(always)]
    fn faer_score(self) -> Self::Real {
        self.abs()
    }

    #[inline(always)]
    fn faer_abs(self) -> Self::Real {
        self.abs()
    }

    #[inline(always)]
    fn faer_abs2(self) -> Self::Real {
        self * self
    }

    #[inline(always)]
    fn faer_nan() -> Self {
        Self::NAN
    }

    #[inline(always)]
    fn faer_from_real(real: Self::Real) -> Self {
        real
    }

    #[inline(always)]
    fn faer_real(self) -> Self::Real {
        self
    }

    #[inline(always)]
    fn faer_imag(self) -> Self::Real {
        Self::ZERO
    }

    #[inline(always)]
    fn faer_zero() -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn faer_one() -> Self {
        Self(1.0, 0.0)
    }

    #[inline(always)]
    fn faer_slice_as_simd<S: Simd>(slice: &[Self::Unit]) -> (&[Self::SimdUnit<S>], &[Self::Unit]) {
        S::f64s_as_simd(slice)
    }

    #[inline(always)]
    fn faer_slice_as_simd_mut<S: Simd>(
        slice: &mut [Self::Unit],
    ) -> (&mut [Self::SimdUnit<S>], &mut [Self::Unit]) {
        S::f64s_as_mut_simd(slice)
    }

    #[inline(always)]
    fn faer_partial_load_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
        simd.f64s_partial_load(slice)
    }

    #[inline(always)]
    fn faer_partial_store_unit<S: Simd>(
        simd: S,
        slice: &mut [Self::Unit],
        values: Self::SimdUnit<S>,
    ) {
        simd.f64s_partial_store(slice, values)
    }

    #[inline(always)]
    fn faer_partial_load_last_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
        simd.f64s_partial_load_last(slice)
    }

    #[inline(always)]
    fn faer_partial_store_last_unit<S: Simd>(
        simd: S,
        slice: &mut [Self::Unit],
        values: Self::SimdUnit<S>,
    ) {
        simd.f64s_partial_store_last(slice, values)
    }

    #[inline(always)]
    fn faer_simd_splat_unit<S: Simd>(simd: S, unit: Self::Unit) -> Self::SimdUnit<S> {
        simd.f64s_splat(unit)
    }

    #[inline(always)]
    fn faer_simd_neg<S: Simd>(simd: S, values: SimdGroupFor<Self, S>) -> SimdGroupFor<Self, S> {
        double::simd_neg(simd, values)
    }

    #[inline(always)]
    fn faer_simd_conj<S: Simd>(simd: S, values: SimdGroupFor<Self, S>) -> SimdGroupFor<Self, S> {
        let _ = simd;
        values
    }

    #[inline(always)]
    fn faer_simd_add<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        double::simd_add(simd, lhs, rhs)
    }

    #[inline(always)]
    fn faer_simd_sub<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        double::simd_sub(simd, lhs, rhs)
    }

    #[inline(always)]
    fn faer_simd_mul<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        double::simd_mul(simd, lhs, rhs)
    }

    #[inline(always)]
    fn faer_simd_scale_real<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        double::simd_mul(simd, lhs, rhs)
    }

    #[inline(always)]
    fn faer_simd_conj_mul<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        double::simd_mul(simd, lhs, rhs)
    }

    #[inline(always)]
    fn faer_simd_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        double::simd_add(simd, acc, double::simd_mul(simd, lhs, rhs))
    }

    #[inline(always)]
    fn faer_simd_conj_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        double::simd_add(simd, acc, double::simd_mul(simd, lhs, rhs))
    }

    #[inline(always)]
    fn faer_simd_score<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self::Real, S> {
        double::simd_abs(simd, values)
    }

    #[inline(always)]
    fn faer_simd_abs2_adde<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self::Real, S>,
    ) -> SimdGroupFor<Self::Real, S> {
        Self::faer_simd_add(simd, acc, Self::faer_simd_mul(simd, values, values))
    }

    #[inline(always)]
    fn faer_simd_abs2<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self::Real, S> {
        Self::faer_simd_mul(simd, values, values)
    }

    #[inline(always)]
    fn faer_simd_scalar_mul<S: Simd>(simd: S, lhs: Self, rhs: Self) -> Self {
        let _ = simd;
        lhs * rhs
    }

    #[inline(always)]
    fn faer_simd_scalar_conj_mul<S: Simd>(simd: S, lhs: Self, rhs: Self) -> Self {
        let _ = simd;
        lhs * rhs
    }

    #[inline(always)]
    fn faer_simd_scalar_mul_adde<S: Simd>(simd: S, lhs: Self, rhs: Self, acc: Self) -> Self {
        let _ = simd;
        lhs * rhs + acc
    }

    #[inline(always)]
    fn faer_simd_scalar_conj_mul_adde<S: Simd>(simd: S, lhs: Self, rhs: Self, acc: Self) -> Self {
        let _ = simd;
        lhs * rhs + acc
    }

    #[inline(always)]
    fn faer_slice_as_aligned_simd<S: Simd>(
        simd: S,
        slice: &[UnitFor<Self>],
        offset: pulp::Offset<SimdMaskFor<Self, S>>,
    ) -> (
        pulp::Prefix<'_, UnitFor<Self>, S, SimdMaskFor<Self, S>>,
        &[SimdUnitFor<Self, S>],
        pulp::Suffix<'_, UnitFor<Self>, S, SimdMaskFor<Self, S>>,
    ) {
        simd.f64s_as_aligned_simd(slice, offset)
    }
    #[inline(always)]
    fn faer_slice_as_aligned_simd_mut<S: Simd>(
        simd: S,
        slice: &mut [UnitFor<Self>],
        offset: pulp::Offset<SimdMaskFor<Self, S>>,
    ) -> (
        pulp::PrefixMut<'_, UnitFor<Self>, S, SimdMaskFor<Self, S>>,
        &mut [SimdUnitFor<Self, S>],
        pulp::SuffixMut<'_, UnitFor<Self>, S, SimdMaskFor<Self, S>>,
    ) {
        simd.f64s_as_aligned_mut_simd(slice, offset)
    }

    #[inline(always)]
    fn faer_simd_rotate_left<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
        amount: usize,
    ) -> SimdGroupFor<Self, S> {
        Double(
            simd.f64s_rotate_left(values.0, amount),
            simd.f64s_rotate_left(values.1, amount),
        )
    }

    #[inline(always)]
    fn faer_align_offset<S: Simd>(
        simd: S,
        ptr: *const UnitFor<Self>,
        len: usize,
    ) -> pulp::Offset<SimdMaskFor<Self, S>> {
        simd.f64s_align_offset(ptr, len)
    }
}
