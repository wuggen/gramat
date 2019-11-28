//! Helpers for performing floating-point arithmetic.

/// The default threshold for approximate equality for `f32`s.
///
/// This value is based on (but larger than) the host system's machine epsilon for 32-bit floating
/// point numbers.
///
/// The machine epsilon is an upper bound on the relative rounding error incurred by a single
/// arithmetic operation. In contrast, `EQ_THRESHOLD_F32` is intended to provide an upper bound on
/// the _absolute_ error incurred by a moderate-length series of operations, and so is several
/// times the machine epsilon.
///
/// For very long, iterative, or recursive calculations, or calculations that involve very large
/// numbers, this bound may be too tight. For such cases, [`ApproxEq`] provides the
/// [`within_threshold`] method to allow users to provide a more appropriate bound.
///
/// See the [`ApproxEq`] trait for further information.
///
/// [`ApproxEq`]: trait.ApproxEq.html
/// [`within_threshold`]: trait.ApproxEq.html#tymethod.within_threshold
pub const EQ_THRESHOLD_F32: f32 = 16.0 * std::f32::EPSILON;

/// Trait for types that can be compared for _approximate_ equality.
///
/// Round-off error is a pervasive problem for floating-point arithemtic. Limited precision means
/// that many expressions that are mathematically equal evaluate as unqual when computed using
/// floating-point. A simple workaround is to, instead of comparing for exact equality, test
/// whether values are within some small absolute difference of each other. The `ApproxEq` trait is
/// intended to make this pattern a little more ergonomic by supplying a default difference
/// threshold and encapsulating the comparison in a single method.
///
/// `gramat` implements `ApproxEq` for `f32` and all of the `gramat` mathematical objects (vectors,
/// matrices, etc). All of these implementations use the default threshold [`EQ_THRESHOLD_F32`].
/// There are also assertion macros [`assert_approx_eq`] and [`assert_within_threshold`]
/// corresponding to `ApproxEq`'s trait methods.
///
/// [`EQ_THRESHOLD_F32`]: constant.EQ_THRESHOLD_F32.html
/// [`assert_approx_eq`]: ../macro.assert_approx_eq.html
/// [`assert_within_threshold`]: ../macro.assert_within_threshold.html
pub trait ApproxEq<T = Self> {
    /// Compare two values for approximate equality, using a default difference treshold.
    fn approx_eq(&self, rhs: &T) -> bool;

    /// Compare two values for approximate equality, using a user-defined difference threshold.
    fn within_threshold(&self, rhs: &T, threshold: &T) -> bool;
}

impl ApproxEq for f32 {
    #[inline(always)]
    fn approx_eq(&self, rhs: &f32) -> bool {
        Self::within_threshold(self, rhs, &EQ_THRESHOLD_F32)
    }

    #[inline(always)]
    fn within_threshold(&self, rhs: &f32, threshold: &f32) -> bool {
        f32::abs(*self - *rhs) <= *threshold
    }
}
