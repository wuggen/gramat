//! Mathematical utility traits, constants, and functions.

/// The default threshold for approximate equality for `f32`s.
///
/// See the [`ApproxEq`] trait for further information.
///
/// [`ApproxEq`]: trait.ApproxEq.html
pub const EQ_THRESHOLD_F32: f32 = 16.0 * std::f32::EPSILON;

/// Trait for types that can be compared for _approximate_ equality.
///
/// Round-off error is a pervasive problem for floating-point arithemtic. Limited precision means
/// that many expressions that are mathematically equal evaluate as unqual when computed using
/// floating-point. A common workaround for this problem is to instead test whether values are
/// within some small absolute difference of each other. The `ApproxEq` trait is intended to make
/// this pattern a little more ergonomic by supplying a default difference threshold and
/// encapsulating the comparison in a single method.
///
/// `gramat` implements `ApproxEq` for `f32` and all of its mathematical objects (vectors,
/// matrices, quaternions). All of these implementations use the default threshold
/// [`EQ_THRESHOLD_F32`]. There are also assertion macros [`assert_approx_eq`] and
/// [`assert_within_threshold`] corresponding to `ApproxEq`'s trait methods.
///
/// [`EQ_THRESHOLD_F32`]: constant.EQ_THRESHOLD_F32.html
/// [`assert_approx_eq`]: ../macro.assert_approx_eq.html
/// [`assert_within_threshold`]: ../macro.assert_within_threshold.html
pub trait ApproxEq<T = Self> {
    /// Compare two values for approximate equality, using a default treshold.
    fn approx_eq(&self, rhs: &T) -> bool;

    /// Compare two values, returning true if they are within the given threshold of one another.
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

#[cfg(test)]
#[test]
#[should_panic]
fn fail_eq_thresh() {
    panic!("EQ_THRESHOLD_F32 = {}", EQ_THRESHOLD_F32)
}
