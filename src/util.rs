//! Mathematical utility traits, constants, and functions.

/// The default threshold for approximate equality for two `f32`s: 64 times the machine epsilon.
pub const EQ_THRESHOLD: f32 = 64.0 * std::f32::EPSILON;

/// Trait for types that can be compared for _approximate_ equality.
pub trait ApproxEq<T = Self> {
    /// Compare two values for approximate equality, using a default treshold.
    fn approx_eq(&self, rhs: &T) -> bool;

    /// Compare two values, returning true if they are within the given threshold of one another.
    fn within_threshold(&self, rhs: &T, threshold: &T) -> bool;
}

impl ApproxEq for f32 {
    #[inline(always)]
    fn approx_eq(&self, rhs: &f32) -> bool {
        Self::within_threshold(self, rhs, &EQ_THRESHOLD)
    }

    #[inline(always)]
    fn within_threshold(&self, rhs: &f32, threshold: &f32) -> bool {
        f32::abs(*self - *rhs) <= *threshold
    }
}
