//! Two, three, and four-dimensional real vectors with `f32` components.
//!
//! This module defines a trait [`Vector`] for generic mathematical vectors, as well as three
//! default vector types useful for graphics programming: [`Vec2`], [`Vec3`], and [`Vec4`], vectors
//! in the 2, 3, and 4 dimensional real spaces respectively. To facilitate interfaces with common
//! graphics APIs, all three are represented as contiguous tuples of 32-bit floating point numbers.
//!
//! The vector types defined here implement a full set of arithmetic operators on both scalars and
//! other vectors:
//!
//! ```
//! # #[macro_use] extern crate gramit;
//! # use gramit::*;
//! let v1 = vec2!(1.0, 2.0);
//! let v2 = vec2!(3.0, 4.0);
//!
//! // Between a vector and an `f32` scalar, we may:
//!
//! // Multiply (on either side) to scale the vector:
//! assert_approx_eq!(2.0 * v1, vec2!(2.0, 4.0));
//! assert_approx_eq!(v1 * 2.0, vec2!(2.0, 4.0));
//!
//! // ... or divide by the scalar:
//! assert_approx_eq!(v1 / 2.0, vec2!(0.5, 1.0));
//!
//! // Between two vectors, we can perform component-wise arithmetic:
//! assert_approx_eq!(v1 + v2, vec2!(4.0, 6.0));
//! assert_approx_eq!(v1 - v2, vec2!(-2.0, -2.0));
//! assert_approx_eq!(v1 * v2, vec2!(3.0, 8.0));
//! assert_approx_eq!(v2 / v1, vec2!(3.0, 2.0));
//!
//! // Arithmetic assignment operations are also defined for mutable vectors:
//! let mut v1 = vec2!(4.0, 5.0);
//! let mut v2 = vec2!(6.0, 7.0);
//!
//! v1 *= 2.0;
//! assert_approx_eq!(v1, vec2!(8.0, 10.0));
//!
//! v2 /= 2.0;
//! assert_approx_eq!(v2, vec2!(3.0, 3.5));
//!
//! v1 += v2;
//! assert_approx_eq!(v1, vec2!(11.0, 13.5));
//!
//! v2 -= v1;
//! assert_approx_eq!(v2, vec2!(-8.0, -10.0));
//!
//! v1 *= vec2!(0.5, 10.0);
//! assert_approx_eq!(v1, vec2!(5.5, 135.0));
//!
//! v2 /= vec2!(-4.0, 2.0);
//! assert_approx_eq!(v2, vec2!(2.0, -5.0));
//!
//! // Similar for `Vec3` and `Vec4`.
//! ```
//!
//! Variants of the arithmetic operators are defined for all (sane) combinations of references and
//! moves for their arguments, and all three vector types are `Copy`. In most cases, it should be
//! uneccessary to explicitly reference or dereference vectors in order to perform arithmetic.
//!
//! For convenience, `gramit` defines the macros [`vec2`](../macro.vec2.html),
//! [`vec3`](../macro.vec3.html), and [`vec4`](../macro.vec4.html) for creating new vectors.
//!
//! [`Vector`]: trait.Vector.html
//! [`Vec2`]: struct.Vec2.html
//! [`Vec3`]: struct.Vec3.html
//! [`Vec4`]: struct.Vec4.html

use std::convert::*;
use std::ops::*;

use super::*;

/// Generic vector operations.
///
/// This trait defines vectors very generally; in fact, it does not even constrain implementors to
/// be true vector spaces. In particular, it requires that a scalar type be associated with the
/// implementing vector type, and that scalar multiplication and vector addition are defined, but
/// does not require that the scalar type is a mathematical field.
///
/// In most cases, however, implementors will want to define common mathematical operations, at
/// least for scalar multiplication and vector addition.
pub trait Vector {
    /// The scalar type over which this vector type is defined.
    ///
    /// This should ideally be a field in the mathematical sense. The real numbers, rational
    /// numbers, and complex numbers are common examples.
    ///
    /// Notably, the integers are not a field, since only 1 and -1 have multiplicative inverses.
    /// Hence, vectors with integer components do not constitute a true vector field. Defining such
    /// a vector may be useful for practical purposes, however, and so this type is unconstrained.
    type Scalar;

    /// The dimension of this vector type.
    ///
    /// For most vector types, this should be the number of components in a vector.
    const DIMS: usize;

    /// The length (norm) of this vector.
    ///
    /// This function should compute the most reasonable norm for the vector type, ideally the norm
    /// induced by the inner product defined by the [`dot`] function.
    ///
    /// Most types will implement a Euclidean space, (e.g. two, three, and four-dimensional real
    /// vector spaces,) and so should define this function to compute the Euclidean length (2-norm)
    /// of the vector.
    ///
    /// More obscure vector types may make better use of different norms for this function (e.g. a
    /// vector type with integer components may wish to use Manhattan distance). In such a case,
    /// the implementation should be sure to document exactly which norm is computed by this
    /// function.
    ///
    /// [`dot`]: #tymethod.dot
    fn length(&self) -> Self::Scalar;

    /// The dot (scalar) product of this vector with another.
    ///
    /// Ideally this is a true inner product, whose induced norm is implemented by the [`length`]
    /// function.
    ///
    /// [`length`]: #tymethod.length
    fn dot(&self, other: &Self) -> Self::Scalar;

    /// Returns a vector of all 1's.
    fn ones() -> Self;

    /// Returns a vector of all 0's.
    fn zeros() -> Self;

    /// Returns the normalized (unit-length) version of this vector.
    fn unit(&self) -> Self;

    /// The scalar multiplication operation.
    fn scalar_mul(&self, s: &Self::Scalar) -> Self;

    /// The vector addition operation.
    fn vector_add(&self, v: &Self) -> Self;
}

macro_rules! decl_vec {
    ($name:ident, $($dims:ident),+) => {
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub struct $name {
            $(pub $dims: f32),+
        }

        impl $name {
            #[doc = "Create a new vector with the given components."]
            #[inline(always)]
            pub fn new($($dims: f32),+) -> Self {
                $name {
                    $($dims),+
                }
            }

            $(#[doc = "Create a unit vector along this axis."]
            pub fn $dims() -> Self {
                let mut v = Self::zeros();
                v.$dims = 1.0;
                v
            })+

            #[doc = "Get the angle between this vector and another."]
            pub fn angle_between(&self, other: &$name) -> Angle {
                Angle::acos(self.dot(&other) / self.length() / other.length()).unwrap()
            }
        }

        impl Vector for $name {
            type Scalar = f32;

            const DIMS: usize = count_args!($($dims),+);

            fn length(&self) -> f32 {
                self.dot(self).sqrt()
            }

            fn dot(&self, other: &Self) -> f32 {
                0.0 $(+ self.$dims * other.$dims)+
            }

            #[inline(always)]
            fn ones() -> Self {
                $name::new($(replace_tt!($dims, 1.0)),+)
            }

            #[inline(always)]
            fn zeros() -> Self {
                $name::new($(replace_tt!($dims, 0.0)),+)
            }

            fn unit(&self) -> Self {
                self / self.length()
            }

            #[inline(always)]
            fn scalar_mul(&self, s: &f32) -> Self {
                s * self
            }

            #[inline(always)]
            fn vector_add(&self, v: &Self) -> Self {
                self + v
            }
        }

        impl AsRef<[f32]> for $name {
            #[inline(always)]
            fn as_ref(&self) -> &[f32] {
                unsafe { &*(self as *const $name as *const [f32; count_args!($($dims),+)]) }
            }
        }

        impl AsMut<[f32]> for $name {
            #[inline(always)]
            fn as_mut(&mut self) -> &mut [f32] {
                unsafe { &mut *(self as *mut $name as *mut [f32; count_args!($($dims),+)]) }
            }
        }

        impl Into<Vec<f32>> for $name {
            fn into(self) -> Vec<f32> {
                Vec::from(self.as_ref())
            }
        }

        impl Index<usize> for $name {
            type Output = f32;

            #[inline(always)]
            fn index(&self, idx: usize) -> &f32 {
                &self.as_ref()[idx]
            }
        }

        impl IndexMut<usize> for $name {
            #[inline(always)]
            fn index_mut(&mut self, idx: usize) -> &mut f32 {
                &mut self.as_mut()[idx]
            }
        }

        impl ApproxEq for $name {
            fn approx_eq(&self, rhs: &$name) -> bool {
                $(self.$dims.approx_eq(&rhs.$dims))&+
            }

            #[doc = "Compare two vectors for approximate equality.\n\n"]
            #[doc = "Uses a third vector for component-wise thresholds."]
            fn within_threshold(&self, rhs: &$name, threshold: &$name) -> bool {
                $(self.$dims.within_threshold(&rhs.$dims, &threshold.$dims))&+
            }
        }

        decl_vec!(@VECTOP $name, Add, add, +, $($dims),+);
        decl_vec!(@VECTOP $name, Sub, sub, -, $($dims),+);
        decl_vec!(@VECTOP $name, Mul, mul, *, $($dims),+);
        decl_vec!(@VECTOP $name, Div, div, /, $($dims),+);

        decl_vec!(@VECTASGN $name, AddAssign, add_assign, +=, $($dims),+);
        decl_vec!(@VECTASGN $name, SubAssign, sub_assign, -=, $($dims),+);
        decl_vec!(@VECTASGN $name, MulAssign, mul_assign, *=, $($dims),+);
        decl_vec!(@VECTASGN $name, DivAssign, div_assign, /=, $($dims),+);

        decl_vec!(@SCALAR_MUL $name, $($dims),+);
        decl_vec!(@SCALAR_DIV $name, $($dims),+);

        decl_vec!(@MULASGN $name, f32, $($dims),+);
        decl_vec!(@MULASGN $name, &f32, $($dims),+);

        decl_vec!(@DIVASGN $name, f32, $($dims),+);
        decl_vec!(@DIVASGN $name, &f32, $($dims),+);

        decl_vec!(@NEG $name, $name, $($dims),+);
        decl_vec!(@NEG $name, &$name, $($dims),+);
    };

    (@VECTOP $name:ident, $trait:ident, $func:ident, $op:tt, $($dims:ident),+) => {
        decl_vec!(@VECTOP_SINGLE $name, $trait, $func, $op, $name, $name, $($dims),+);
        decl_vec!(@VECTOP_SINGLE $name, $trait, $func, $op, $name, &$name, $($dims),+);
        decl_vec!(@VECTOP_SINGLE $name, $trait, $func, $op, &$name, $name, $($dims),+);
        decl_vec!(@VECTOP_SINGLE $name, $trait, $func, $op, &$name, &$name, $($dims),+);
    };
    (@VECTOP_SINGLE $name:ident, $trait:ident, $func:ident, $op:tt, $lhs:ty, $rhs:ty, $($dims:ident),+) => {
        impl $trait<$rhs> for $lhs {
            type Output = $name;

            #[doc = "Component-wise arithemetic operation."]
            fn $func(self, rhs: $rhs) -> $name {
                $name {
                    $($dims: self.$dims $op rhs.$dims),+
                }
            }
        }
    };

    (@VECTASGN $name:ident, $trait:ident, $func:ident, $op:tt, $($dims:ident),+) => {
        decl_vec!(@VECTASGN_SINGLE $name, $trait, $func, $op, $name, $($dims),+);
        decl_vec!(@VECTASGN_SINGLE $name, $trait, $func, $op, &$name, $($dims),+);
    };
    (@VECTASGN_SINGLE $name:ident, $trait:ident, $func:ident, $op:tt, $rhs:ty, $($dims:ident),+) => {
        impl $trait<$rhs> for $name {
            #[doc = "Component-wise arithmetic assignment operation."]
            fn $func(&mut self, rhs: $rhs) {
                $(self.$dims $op rhs.$dims;)+
            }
        }
    };

    (@SCALAR_MUL $name:ident, $($dims:ident),+) => {
        decl_vec!(@SCALAR_MUL_SINGLE $name, $name, f32, $($dims),+);
        decl_vec!(@SCALAR_MUL_SINGLE $name, $name, &f32, $($dims),+);
        decl_vec!(@SCALAR_MUL_SINGLE $name, &$name, f32, $($dims),+);
        decl_vec!(@SCALAR_MUL_SINGLE $name, &$name, &f32, $($dims),+);
    };
    (@SCALAR_MUL_SINGLE $name:ident, $lhs:ty, $rhs:ty, $($dims:ident),+) => {
        impl Mul<$rhs> for $lhs {
            type Output = $name;

            fn mul(self, rhs: $rhs) -> $name {
                $name {
                    $($dims: self.$dims * rhs),+
                }
            }
        }

        impl Mul<$lhs> for $rhs {
            type Output = $name;

            fn mul(self, rhs: $lhs) -> $name {
                $name {
                    $($dims: rhs.$dims * self),+
                }
            }
        }
    };

    (@MULASGN $name:ident, $rhs:ty, $($dims:ident),+) => {
        impl MulAssign<$rhs> for $name {
            fn mul_assign(&mut self, rhs: $rhs) {
                $(self.$dims *= rhs;)+
            }
        }
    };

    (@SCALAR_DIV $name:ident, $($dims:ident),+) => {
        decl_vec!(@SCALAR_DIV_SINGLE $name, $name, f32, $($dims),+);
        decl_vec!(@SCALAR_DIV_SINGLE $name, $name, &f32, $($dims),+);
        decl_vec!(@SCALAR_DIV_SINGLE $name, &$name, f32, $($dims),+);
        decl_vec!(@SCALAR_DIV_SINGLE $name, &$name, &f32, $($dims),+);
    };
    (@SCALAR_DIV_SINGLE $name:ident, $lhs:ty, $rhs:ty, $($dims:ident),+) => {
        impl Div<$rhs> for $lhs {
            type Output = $name;

            fn div(self, rhs: $rhs) -> $name {
                $name {
                    $($dims: self.$dims / rhs),+
                }
            }
        }
    };

    (@DIVASGN $name:ident, $rhs:ty, $($dims:ident),+) => {
        impl DivAssign<$rhs> for $name {
            fn div_assign(&mut self, rhs: $rhs) {
                $(self.$dims /= rhs;)+
            }
        }
    };

    (@NEG $name:ident, $type:ty, $($dims:ident),+) => {
        impl Neg for $type {
            type Output = $name;

            fn neg(self) -> $name {
                $name {
                    $($dims: -self.$dims),+
                }
            }
        }
    };
}

decl_vec!(Vec2, x, y);
decl_vec!(Vec3, x, y, z);
decl_vec!(Vec4, x, y, z, w);

impl Vec2 {
    /// Extend this `Vec2` to a `Vec3`, with the given _z_ component.
    #[inline(always)]
    pub fn extend(self, z: f32) -> Vec3 {
        Vec3 {
            x: self.x,
            y: self.y,
            z,
        }
    }
}

impl Vec3 {
    /// Compute the cross product `self` &times; `other`.
    pub fn cross(self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Extend this `Vec3` to a `Vec4`, with the given _w_ component.
    #[inline(always)]
    pub fn extend(self, w: f32) -> Vec4 {
        Vec4 {
            x: self.x,
            y: self.y,
            z: self.z,
            w,
        }
    }

    /// Get a homogeneous (point) representation of this `Vec3`.
    ///
    /// This is equivalent to calling `vec3.extend(1.0)`.
    #[inline(always)]
    pub fn homogeneous(self) -> Vec4 {
        self.extend(1.0)
    }

    /// Truncate the _z_ component of this `Vec3` to produce a `Vec2`.
    #[inline(always)]
    pub fn truncate(self) -> Vec2 {
        Vec2 {
            x: self.x,
            y: self.y,
        }
    }
}

impl Vec4 {
    /// Truncate the _w_ component of this `Vec4` to produce a `Vec3`.
    #[inline(always)]
    pub fn truncate(self) -> Vec3 {
        Vec3 {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }

    /// Convert a homogeneous 4-vector to the corresponding point in 3-space.
    pub fn homogenize(self) -> Vec3 {
        self.truncate() / self.w
    }
}

/// Project one vector onto another.
///
/// Returns the projection of `from` onto `onto`.
pub fn project<V: Vector>(from: &V, onto: &V) -> V {
    onto.scalar_mul(&from.unit().dot(&onto.unit()))
}

/// A free function version of [`Vector::length`](trait.Vector.html#tymethod.length).
#[inline(always)]
pub fn length<V: Vector>(v: &V) -> V::Scalar {
    v.length()
}

/// A free function version of [`Vector::dot`](trait.Vector.html#tymethod.dot).
#[inline(always)]
pub fn dot<V: Vector>(a: &V, b: &V) -> V::Scalar {
    a.dot(b)
}

/// A free function version of [`Vector::unit`](trait.Vector.html#tymethod.unit).
#[inline(always)]
pub fn normalize<V: Vector>(v: &V) -> V {
    v.unit()
}

/// A free function version of [`Vec3::cross`](struct.Vec3.html#method.cross).
pub fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    a.cross(b)
}

#[cfg(test)]
macro_rules! test_vec {
    ($name:ident, $($dims:ident),+) => {
        #[test]
        fn dimension() {
            assert_eq!($name::DIMS, count_args!($($dims),+));
        }

        #[test]
        fn ones() {
            let v = $name::ones();
            $(assert!(v.$dims.approx_eq(&1.0));)+
        }

        #[test]
        fn zeros() {
            let v = $name::zeros();
            $(assert!(v.$dims.approx_eq(&0.0));)+
        }

        #[test]
        fn add() {
            let v = $name::ones() + $name::ones();
            $(assert!(v.$dims.approx_eq(&2.0));)+

            let mut v = $name::ones();
            v += $name::ones();
            $(assert!(v.$dims.approx_eq(&2.0));)+
        }

        #[test]
        fn sub() {
            let v = $name::ones() - $name::ones();
            $(assert!(v.$dims.approx_eq(&0.0));)+

            let mut v = $name::ones();
            v -= $name::ones();
            $(assert!(v.$dims.approx_eq(&0.0));)+
        }

        #[test]
        fn scalar_mul() {
            let v = $name::ones() * 2.0;
            $(assert!(v.$dims.approx_eq(&2.0));)+

            let v = 2.0_f32 * $name::ones();
            $(assert!(v.$dims.approx_eq(&2.0));)+

            let mut v = $name::ones();
            v *= 2.0;
            $(assert!(v.$dims.approx_eq(&2.0));)+
        }

        #[test]
        fn scalar_div() {
            let v = $name::ones() / 2.0;
            $(assert!(v.$dims.approx_eq(&0.5));)+

            let mut v = $name::ones();
            v /= 2.0;
            $(assert!(v.$dims.approx_eq(&0.5));)+
        }

        #[test]
        fn unit_axes() {
            $(let v = $name::$dims();
              assert!(v.$dims.approx_eq(&1.0));
              assert!(v.length().approx_eq(&1.0));)+
        }

        #[test]
        fn dot() {
            assert!($name::ones().dot(&$name::ones()).approx_eq(&($name::DIMS as f32)));

            let v = $name::ones() * 2.0;
            assert!(v.dot(&v).approx_eq(&(count_args!($($dims),+) as f32 * 4.0)));

            $(let v = $name::$dims() * 3.0;
              assert!(v.dot(&v).approx_eq(&9.0));)+
        }

        #[test]
        fn length() {
            assert!($name::zeros().length().approx_eq(&0.0));
            assert!($name::ones().length().approx_eq(&(count_args!($($dims),+) as f32).sqrt()));
            $(assert!($name::$dims().length().approx_eq(&1.0));)+
        }

        #[test]
        fn index() {
            let mut v = $name::zeros();
            for i in 0..count_args!($($dims),+) {
                v[i] = i as f32;
            }

            for i in 0..count_args!($($dims),+) {
                assert!(v[i].approx_eq(&(i as f32)));
            }
        }

        #[test]
        fn into_vec() {
            let v: Vec<_> = $name::zeros().into();
            assert_eq!(v.len(), count_args!($($dims),+));
            for n in v {
                assert!(n.approx_eq(&0.0));
            }
        }
    };
}

#[cfg(test)]
mod test_vec2 {
    use super::*;

    test_vec!(Vec2, x, y);
}

#[cfg(test)]
mod test_vec3 {
    use super::*;

    test_vec!(Vec3, x, y, z);
}

#[cfg(test)]
mod test_vec4 {
    use super::*;

    test_vec!(Vec4, x, y, z, w);
}
