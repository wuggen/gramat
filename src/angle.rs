//! Consistent angle representations and conversions.
//!
//! This module defines the [`Angle`] struct to provide a single, unified angle representation
//! independent of either degrees or radians.
//!
//! [`Angle`]: struct.Angle.html

use std::ops::*;

/// An angle struct independent of representation as either degrees or radians.
///
/// See the [module-level documentation](index.html) for more details.
#[derive(Debug, Clone, Copy, PartialOrd, PartialEq)]
#[repr(transparent)]
pub struct Angle {
    radians: f32,
}

impl Angle {
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
    pub fn radians(&self) -> f32 {
        self.radians
    }

    /// Get the value of this `Angle` in degrees.
    #[inline(always)]
    pub fn degrees(&self) -> f32 {
        self.radians.to_degrees()
    }

    /// Get the sine of this `Angle`.
    #[inline(always)]
    pub fn sin(&self) -> f32 {
        self.radians.sin()
    }

    /// Get the cosine of this `Angle`.
    #[inline(always)]
    pub fn cos(&self) -> f32 {
        self.radians.cos()
    }

    /// Get the tangent of this `Angle`.
    #[inline(always)]
    pub fn tan(&self) -> f32 {
        self.radians.tan()
    }

    /// Compute the arc sine of the given value.
    ///
    /// For values in the range [-1, 1], produces an `Angle` in the range [-pi/2, pi/2] (radians).
    /// For other values, produces `None`.
    pub fn asin(x: f32) -> Option<Angle> {
        let radians = x.asin();
        if x.is_nan() {
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
        if x.is_nan() {
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
    pub fn sin_cos(&self) -> (f32, f32) {
        self.radians.sin_cos()
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
