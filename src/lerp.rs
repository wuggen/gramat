use crate::fp::ApproxEq;

use std::convert::TryFrom;
use std::fmt::{self, Display, Formatter};
use std::ops::{Add, Mul};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A value in the range [0, 1].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[repr(transparent)]
pub struct Fraction(f32);

impl ApproxEq for Fraction {
    fn approx_eq(self, rhs: Self) -> bool {
        self.0.approx_eq(rhs.0)
    }

    fn within_threshold(self, rhs: Self, threshold: Self) -> bool {
        self.0.within_threshold(rhs.0, threshold.0)
    }
}

impl Fraction {
    /// Create a new `Fraction`.
    ///
    /// Returns `None` if the given value is less than 0 or greater than 1.
    pub fn new(val: f32) -> Option<Fraction> {
        if val < 0.0 || val > 1.0 {
            None
        } else {
            Some(Fraction(val))
        }
    }

    /// Get the `f32` representation of this `Fraction`.
    pub fn get(self) -> f32 {
        self.0
    }
}

/// The error type for `Fraction`'s `TryFrom<f32>` implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MakeFractionError {
    /// The given value was less than 0.
    ValueLessThanZero,

    /// The given value was greater than 1.
    ValueGreaterThanOne,
}

use MakeFractionError::*;

impl Display for MakeFractionError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            ValueLessThanZero => write!(f, "provided value was less than zero"),
            ValueGreaterThanOne => write!(f, "provided value was greater than one"),
        }
    }
}

impl std::error::Error for MakeFractionError {}

impl TryFrom<f32> for Fraction {
    type Error = MakeFractionError;

    fn try_from(value: f32) -> Result<Fraction, MakeFractionError> {
        Fraction::new(value).ok_or(if value < 0.0 {
            ValueLessThanZero
        } else {
            ValueGreaterThanOne
        })
    }
}

/// Linearly interpolate two values.
///
/// The type `T` must support multiplication by `f32`s, and the output type of that multiplication
/// must support addition with itself.
pub fn lerp<T>(a: T, b: T, t: Fraction) -> <<T as Mul<f32>>::Output as Add>::Output
where
    T: Mul<f32>,
    <T as Mul<f32>>::Output: Add,
{
    let t = t.0;
    (a * (1.0_f32 - t)) + (b * t)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Lerp<T> {
    a: T,
    b: T,
    steps: u32,
    max_steps: u32,
}

impl<T> Lerp<T> {
    pub fn new(a: T, b: T, divisions: u32) -> Lerp<T> {
        Lerp {
            a,
            b,
            steps: 0,
            max_steps: divisions + 1,
        }
    }
}

impl<T> Iterator for Lerp<T>
where
    T: Mul<f32> + Clone,
    <T as Mul<f32>>::Output: Add,
{
    type Item = <<T as Mul<f32>>::Output as Add>::Output;

    fn next(&mut self) -> Option<Self::Item> {
        if self.steps > self.max_steps {
            None
        } else {
            let t = Fraction::new((self.steps as f32) / (self.max_steps as f32)).unwrap();
            self.steps += 1;
            Some(lerp(self.a.clone(), self.b.clone(), t))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = (self.max_steps + 1 - self.steps) as usize;
        (rem, Some(rem))
    }
}

impl<T> std::iter::FusedIterator for Lerp<T>
where
    T: Mul<f32> + Clone,
    <T as Mul<f32>>::Output: Add,
{
}

impl<T> std::iter::ExactSizeIterator for Lerp<T>
where
    T: Mul<f32> + Clone,
    <T as Mul<f32>>::Output: Add,
{
}
