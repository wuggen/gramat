//! Quaternions with `f32` components.

use super::*;
use std::convert::From;
use std::ops::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Quaternion {
    pub r: f32,
    pub i: f32,
    pub j: f32,
    pub k: f32,
}

impl Quaternion {
    /// Construct a `Quaternion` with the given components.
    #[inline(always)]
    pub fn new(r: f32, i: f32, j: f32, k: f32) -> Quaternion {
        Quaternion { r, i, j, k }
    }

    /// Construct a `Quaternion` with only a real part.
    #[inline(always)]
    pub fn real(r: f32) -> Quaternion {
        Quaternion::new(r, 0.0, 0.0, 0.0)
    }

    /// Construct a `Quaternion` with only a vector part.
    #[inline(always)]
    pub fn vector(vec: Vec3) -> Quaternion {
        Quaternion::new(0.0, vec.x, vec.y, vec.z)
    }

    /// Construct a `Quaternion` with the given real and vector parts.
    #[inline(always)]
    pub fn real_vector(r: f32, vec: Vec3) -> Quaternion {
        Quaternion::new(r, vec.x, vec.y, vec.z)
    }

    /// Get the imaginary components of this `Quaternion` as a `Vec3`.
    #[inline(always)]
    pub fn vector_part(&self) -> Vec3 {
        Vec3::new(self.i, self.j, self.k)
    }

    /// Conjugate this `Quaternion` by `q`.
    pub fn conjugate_by(&self, q: &Quaternion) -> Quaternion {
        q * self * q.inverse()
    }

    /// Conjugate this `Quaternion` by `q`, which is assumed to be of unit magnitude.
    ///
    /// This function takes advantage of the fact that, for unit quaternions, the inverse is equal
    /// to the complex conjugate, and so omits several floating point operations in the calculation
    /// of `q`'s inverse.
    ///
    /// # Usage Warning
    /// This function _does not check_ that `q` is of unit magnitude! Doing so would require the
    /// same operations that the unit-magnitude assumption avoids. The user should be sure that `q`
    /// is actually a unit quaternion, or simply use `conjugate_by`, which performs the full
    /// calculation for inverting `q`.
    pub fn conjugate_by_unit(&self, q: &Quaternion) -> Quaternion {
        q * self * q.neg_vector()
    }

    /// Get the multiplicative inverse of this `Quaternion`.
    pub fn inverse(&self) -> Quaternion {
        let q = self.r * self.r + self.i * self.i + self.j * self.j + self.k * self.k;
        Quaternion {
            r: self.r / q,
            i: -self.i / q,
            j: -self.j / q,
            k: -self.k / q,
        }
    }

    /// Get a `Quaternion` that is equal to this `Quaternion` with its vector part negated
    /// (i.e. the complex conjugate).
    ///
    /// For a unit quaternion, this is equal to the inverse.
    pub fn neg_vector(&self) -> Quaternion {
        Quaternion::real_vector(self.r, -self.vector_part())
    }

    /// Get the magnitude of this `Quaternion`.
    pub fn magnitude(&self) -> f32 {
        f32::sqrt(self.r * self.r + self.i * self.i + self.j * self.j + self.k * self.k)
    }

    /// Get the normalized (unit-magnitude) version of this `Quaternion`.
    pub fn unit(&self) -> Quaternion {
        let m = self.magnitude();
        Quaternion {
            r: self.r / m,
            i: self.i / m,
            j: self.j / m,
            k: self.k / m,
        }
    }
}

impl ApproxEq for Quaternion {
    fn approx_eq(self, rhs: Quaternion) -> bool {
        self.r.approx_eq(rhs.r)
            & self.i.approx_eq(rhs.i)
            & self.j.approx_eq(rhs.j)
            & self.k.approx_eq(rhs.k)
    }

    /// Compare two [`Quaternion`]s for approximate equality.
    ///
    /// Uses a third [`Quaternion`] for element-wise thresholds.
    ///
    /// [`Quaternion`]: ../quaternion/struct.Quaternion.html
    fn within_threshold(self, rhs: Quaternion, threshold: Quaternion) -> bool {
        self.r.within_threshold(rhs.r, threshold.r)
            & self.i.within_threshold(rhs.i, threshold.i)
            & self.j.within_threshold(rhs.j, threshold.j)
            & self.k.within_threshold(rhs.k, threshold.k)
    }
}

impl From<f32> for Quaternion {
    /// Converts an `f32` into a [`Quaternion`] with only a real part.
    ///
    /// This conversion is identical to that performed by [`Quaternion::real`].
    ///
    /// [`Quaternion`]: ../quaternion/struct.Quaternion.html
    /// [`Quaternion::real`]: ../quaternion/struct.Quaternion.html#method.real
    #[inline(always)]
    fn from(r: f32) -> Quaternion {
        Quaternion::real(r)
    }
}

impl From<Vec3> for Quaternion {
    /// Converts a [`Vec3`] into a [`Quaternion`] with only a vector part.
    ///
    /// This conversion is identical to that performed by [`Quaternion::vector`].
    ///
    /// [`Vec3`]: ../vec/struct.Vec3.html
    /// [`Quaternion`]: ../quaternion/struct.Quaternion.html
    /// [`Quaternion::vector`]: ../quaternion/struct.Quaternion.html#method.vector
    #[inline(always)]
    fn from(vec: Vec3) -> Quaternion {
        Quaternion::vector(vec)
    }
}

macro_rules! quatop_additive {
    ($trait:ident, $func:ident, $op:tt) => {
        quatop_additive!(@SINGLE $trait, $func, $op, Quaternion, Quaternion);
        quatop_additive!(@SINGLE $trait, $func, $op, Quaternion, &Quaternion);
        quatop_additive!(@SINGLE $trait, $func, $op, &Quaternion, Quaternion);
        quatop_additive!(@SINGLE $trait, $func, $op, &Quaternion, &Quaternion);
    };

    (@SINGLE $trait:ident, $func:ident, $op:tt, $lhs:ty, $rhs:ty) => {
        impl $trait<$rhs> for $lhs {
            type Output = Quaternion;

            fn $func(self, rhs: $rhs) -> Quaternion {
                Quaternion {
                    r: self.r $op rhs.r,
                    i: self.i $op rhs.i,
                    j: self.j $op rhs.j,
                    k: self.k $op rhs.k,
                }
            }
        }
    };
}

quatop_additive!(Add, add, +);
quatop_additive!(Sub, sub, -);

macro_rules! quatop_additive_assign {
    ($trait:ident, $func:ident, $op:tt) => {
        quatop_additive_assign!(@SINGLE $trait, $func, $op, Quaternion);
        quatop_additive_assign!(@SINGLE $trait, $func, $op, &Quaternion);
    };

    (@SINGLE $trait:ident, $func:ident, $op:tt, $rhs:ty) => {
        impl $trait<$rhs> for Quaternion {
            fn $func(&mut self, rhs: $rhs) {
                self.r $op rhs.r;
                self.i $op rhs.i;
                self.j $op rhs.j;
                self.k $op rhs.k;
            }
        }
    };
}

quatop_additive_assign!(AddAssign, add_assign, +=);
quatop_additive_assign!(SubAssign, sub_assign, -=);

macro_rules! quatop_mult {
    ($lhs:ty, $rhs:ty) => {
        impl Mul<$rhs> for $lhs {
            type Output = Quaternion;

            fn mul(self, rhs: $rhs) -> Quaternion {
                Quaternion {
                    r: self.r * rhs.r - self.i * rhs.i - self.j * rhs.j - self.k * rhs.k,
                    i: self.r * rhs.i + self.i * rhs.r + self.j * rhs.k - self.k * rhs.j,
                    j: self.r * rhs.j - self.i * rhs.k + self.j * rhs.r + self.k * rhs.i,
                    k: self.r * rhs.k + self.i * rhs.j - self.j * rhs.i + self.k * rhs.r,
                }
            }
        }
    };
}

quatop_mult!(Quaternion, Quaternion);
quatop_mult!(Quaternion, &Quaternion);
quatop_mult!(&Quaternion, Quaternion);
quatop_mult!(&Quaternion, &Quaternion);

macro_rules! quatop_mult_assign {
    ($rhs:ty) => {
        impl MulAssign<$rhs> for Quaternion {
            fn mul_assign(&mut self, rhs: $rhs) {
                self.r = self.r * rhs.r - self.i * rhs.i - self.j * rhs.j - self.k * rhs.k;
                self.i = self.r * rhs.i + self.i * rhs.r + self.j * rhs.k - self.k * rhs.j;
                self.j = self.r * rhs.j - self.i * rhs.k + self.j * rhs.r + self.k * rhs.i;
                self.k = self.r * rhs.k + self.i * rhs.j - self.j * rhs.i + self.k * rhs.r;
            }
        }
    };
}

quatop_mult_assign!(Quaternion);
quatop_mult_assign!(&Quaternion);

macro_rules! quatop_div {
    ($lhs:ty, $rhs:ty) => {
        #[allow(clippy::suspicious_arithmetic_impl)]
        impl Div<$rhs> for $lhs {
            type Output = Quaternion;

            fn div(self, rhs: $rhs) -> Quaternion {
                self * rhs.inverse()
            }
        }
    };
}

quatop_div!(Quaternion, Quaternion);
quatop_div!(Quaternion, &Quaternion);
quatop_div!(&Quaternion, Quaternion);
quatop_div!(&Quaternion, &Quaternion);

macro_rules! quatop_div_assign {
    ($rhs:ty) => {
        impl DivAssign<$rhs> for Quaternion {
            fn div_assign(&mut self, rhs: $rhs) {
                *self *= rhs.inverse();
            }
        }
    };
}

quatop_div_assign!(Quaternion);
quatop_div_assign!(&Quaternion);

macro_rules! quatop_neg {
    ($type:ty) => {
        impl Neg for $type {
            type Output = Quaternion;

            #[inline(always)]
            fn neg(self) -> Quaternion {
                Quaternion {
                    r: -self.r,
                    i: -self.i,
                    j: -self.j,
                    k: -self.k,
                }
            }
        }
    };
}

quatop_neg!(Quaternion);
quatop_neg!(&Quaternion);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn basic() {
        let r = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let i = Quaternion::new(0.0, 1.0, 0.0, 0.0);
        let j = Quaternion::new(0.0, 0.0, 1.0, 0.0);
        let k = Quaternion::new(0.0, 0.0, 0.0, 1.0);

        // Multiplication by unit real
        assert_approx_eq!(r * r, r);
        assert_approx_eq!(r * i, i);
        assert_approx_eq!(i * r, i);
        assert_approx_eq!(r * j, j);
        assert_approx_eq!(j * r, j);
        assert_approx_eq!(r * k, k);
        assert_approx_eq!(k * r, k);

        // Squares of imaginary components
        assert_approx_eq!(i * i, -r);
        assert_approx_eq!(j * j, -r);
        assert_approx_eq!(k * k, -r);

        // Circle of imaginaries
        assert_approx_eq!(i * j, k);
        assert_approx_eq!(j * k, i);
        assert_approx_eq!(k * i, j);

        assert_approx_eq!(j * i, -k);
        assert_approx_eq!(i * k, -j);
        assert_approx_eq!(k * j, -i);

        assert_approx_eq!(i * j * k, -r);
    }
}
