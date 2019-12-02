//! Consistent angle representations and conversions.
//!
//! This module defines the [`Angle`] struct to provide a single, unified angle representation
//! independent of either degrees or radians.
//!
//! [`Angle`]: struct.Angle.html

use std::ops::*;
use std::f32;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::fp::ApproxEq;

/// An angle struct independent of representation as either degrees or radians.
///
/// See the [module-level documentation](index.html) for more details.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(transparent)]
pub struct Angle {
    radians: f32,
}

impl Angle {
    /// Construct an `Angle` of zero.
    #[inline(always)]
    pub fn zero() -> Angle {
        Angle { radians: 0.0 }
    }

    /// Construct an `Angle` of 360 degrees (`2*PI` radians).
    #[inline(always)]
    pub fn full_circle() -> Angle {
        Angle { radians: 2.0 * f32::consts::PI }
    }

    /// Construct an `Angle` of 180 degrees (`PI` radians).
    #[inline(always)]
    pub fn half_circle() -> Angle {
        Angle { radians: f32::consts::PI }
    }

    /// Construct an `Angle` from a value given in radians.
    #[inline(always)]
    pub fn from_radians(radians: f32) -> Angle {
        Angle { radians }
    }

    /// Constuct an `Angle` from a value given in degrees.
    #[inline(always)]
    pub fn from_degrees(degrees: f32) -> Angle {
        Angle {
            radians: degrees.to_radians(),
        }
    }

    /// Get the value of this `Angle` in radians.
    #[inline(always)]
    pub fn radians(self) -> f32 {
        self.radians
    }

    /// Get the value of this `Angle` in degrees.
    #[inline(always)]
    pub fn degrees(self) -> f32 {
        self.radians.to_degrees()
    }

    /// Get the sine of this `Angle`.
    #[inline(always)]
    pub fn sin(self) -> f32 {
        self.radians.sin()
    }

    /// Get the cosine of this `Angle`.
    #[inline(always)]
    pub fn cos(self) -> f32 {
        self.radians.cos()
    }

    /// Get the tangent of this `Angle`.
    #[inline(always)]
    pub fn tan(self) -> f32 {
        self.radians.tan()
    }

    /// Compute the arc sine of the given value.
    ///
    /// For values in the range [-1, 1], produces an `Angle` in the range [-pi/2, pi/2] (radians).
    /// For other values, produces `None`.
    pub fn asin(x: f32) -> Option<Angle> {
        let radians = x.asin();
        if radians.is_nan() {
            None
        } else {
            Some(Angle { radians })
        }
    }

    /// Compute the arc cosine of the given value.
    ///
    /// For values in the range [-1, 1], produces an `Angle` in the range [0, pi] (radians). For
    /// other values, produces `None`.
    pub fn acos(x: f32) -> Option<Angle> {
        let radians = x.acos();
        if radians.is_nan() {
            None
        } else {
            Some(Angle { radians })
        }
    }

    /// Compute the arc tangent of the given value.
    #[inline(always)]
    pub fn atan(x: f32) -> Angle {
        Angle { radians: x.atan() }
    }

    /// Compute the four-quadrant arc tangent of `y` and `x`.
    #[inline(always)]
    pub fn atan2(y: f32, x: f32) -> Angle {
        Angle {
            radians: f32::atan2(y, x),
        }
    }

    /// Simultaneously compute the sine and cosine of this `Angle`. Returns `(sin(x), cos(x))`.
    #[inline(always)]
    pub fn sin_cos(self) -> (f32, f32) {
        self.radians.sin_cos()
    }
}

impl Default for Angle {
    /// Construct a zero `Angle`.
    #[inline(always)]
    fn default() -> Angle {
        Angle::zero()
    }
}

impl ApproxEq for Angle {
    fn approx_eq(self, rhs: Angle) -> bool {
        self.radians.approx_eq(rhs.radians)
    }

    fn within_threshold(self, rhs: Angle, threshold: Angle) -> bool {
        self.radians.within_threshold(rhs.radians, threshold.radians)
    }
}

impl Add for Angle {
    type Output = Angle;

    #[inline(always)]
    fn add(self, rhs: Angle) -> Angle {
        Angle {
            radians: self.radians + rhs.radians,
        }
    }
}

impl AddAssign for Angle {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Angle) {
        self.radians += rhs.radians;
    }
}

impl Sub for Angle {
    type Output = Angle;

    #[inline(always)]
    fn sub(self, rhs: Angle) -> Angle {
        Angle {
            radians: self.radians - rhs.radians,
        }
    }
}

impl SubAssign for Angle {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Angle) {
        self.radians -= rhs.radians;
    }
}

impl Mul<f32> for Angle {
    type Output = Angle;

    #[inline(always)]
    fn mul(self, rhs: f32) -> Angle {
        Angle {
            radians: self.radians * rhs,
        }
    }
}

impl Mul<Angle> for f32 {
    type Output = Angle;

    #[inline(always)]
    fn mul(self, rhs: Angle) -> Angle {
        rhs * self
    }
}

impl MulAssign<f32> for Angle {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f32) {
        self.radians *= rhs;
    }
}

impl Div<f32> for Angle {
    type Output = Angle;

    #[inline(always)]
    fn div(self, rhs: f32) -> Angle {
        Angle {
            radians: self.radians / rhs,
        }
    }
}

impl DivAssign<f32> for Angle {
    #[inline(always)]
    fn div_assign(&mut self, rhs: f32) {
        self.radians /= rhs;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ApproxEq;
    use std::f32::consts::PI;

    #[test]
    fn test_defaults() {
        let z = Angle::zero();
        let h = Angle::half_circle();
        let f = Angle::full_circle();
        let d = Angle::default();

        assert_approx_eq!(z.degrees(), 0.0);
        assert_approx_eq!(z.radians(), 0.0);

        assert_approx_eq!(h.degrees(), 180.0);
        assert_approx_eq!(h.radians(), PI);

        assert_approx_eq!(f.degrees(), 360.0);
        assert_approx_eq!(f.radians(), 2.0*PI);

        assert_approx_eq!(z, d);
    }

    #[test]
    fn test_from_radians() {
        let a = Angle::from_radians(PI/2.0);
        assert_approx_eq!(a.radians(), PI/2.0);
        assert_approx_eq!(a.degrees(), 90.0);

        let a = Angle::from_radians(PI/6.0);
        assert_approx_eq!(a.radians(), PI/6.0);
        assert_approx_eq!(a.degrees(), 30.0);
    }

    #[test]
    fn test_from_degrees() {
        let a = Angle::from_degrees(90.0);
        assert_approx_eq!(a.radians(), PI/2.0);
        assert_approx_eq!(a.degrees(), 90.0);

        let a = Angle::from_degrees(30.0);
        assert_approx_eq!(a.radians(), PI/6.0);
        assert_approx_eq!(a.degrees(), 30.0);
    }

    #[test]
    fn test_asin() {
        let a = Angle::asin(0.5).unwrap();
        assert_approx_eq!(a.degrees(), 30.0);

        let a = Angle::asin(-f32::sqrt(3.0)/2.0).unwrap();
        assert_approx_eq!(a.degrees(), -60.0, "a.degrees() -> {} ({})", a.degrees(), crate::fp::EQ_THRESHOLD_F32);
    }

    #[test]
    fn test_asin_none() {
        let a = Angle::asin(1.5);
        assert!(a.is_none());

        let a = Angle::asin(-1.1);
        assert!(a.is_none());
    }

    #[test]
    fn test_acos() {
        let a = Angle::acos(0.5).unwrap();
        assert_approx_eq!(a.degrees(), 60.0);

        let a = Angle::acos(-f32::sqrt(3.0)/2.0).unwrap();
        assert_approx_eq!(a.degrees(), 150.0);
    }
}
