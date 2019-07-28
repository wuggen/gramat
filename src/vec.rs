//! Two, three, and four-dimensional real vectors with `f32` components.

use std::convert::*;
use std::ops::*;

use super::*;

/// Generic vector operations.
pub trait Vector {
    /// The dimension of this vector type.
    fn dimension() -> usize;

    /// The Euclidean length (2-norm) of this vector.
    fn length(&self) -> f32;

    /// The dot (inner) product of this vector with another.
    fn dot(&self, other: &Self) -> f32;

    /// Returns a vector of all 1's.
    fn ones() -> Self;

    /// Returns a vector of all 0's.
    fn zeros() -> Self;

    /// Returns the normalized (unit-length) version of this vector.
    fn unit(&self) -> Self;
}

macro_rules! decl_vec {
    ($name:ident, $($dims:ident),+) => {
        #[repr(C)]
        #[derive(Debug, Clone)]
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
            #[inline(always)]
            fn dimension() -> usize {
                count_args!($($dims),+)
            }

            fn length(&self) -> f32 {
                self.dot(&self).sqrt()
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

            #[doc = "Compare two vectors for approximate equality.\n\nUses a third"]
            #[doc = "vector for component-wise thresholds."]
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
    pub fn extend(&self, z: f32) -> Vec3 {
        Vec3 {
            x: self.x,
            y: self.y,
            z,
        }
    }
}

impl Vec3 {
    /// Compute the cross product `self` &times; `other`.
    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Extend this `Vec3` to a `Vec4`, with the given _w_ component.
    #[inline(always)]
    pub fn extend(&self, w: f32) -> Vec4 {
        Vec4 {
            x: self.x,
            y: self.y,
            z: self.z,
            w,
        }
    }

    /// Truncate the _z_ component of this `Vec3` to produce a `Vec2`.
    #[inline(always)]
    pub fn truncate(&self) -> Vec2 {
        Vec2 {
            x: self.x,
            y: self.y,
        }
    }
}

impl Vec4 {
    /// Truncate the _w_ component of this `Vec4` to produce a `Vec3`.
    #[inline(always)]
    pub fn truncate(&self) -> Vec3 {
        Vec3 {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }

    /// Convert a homogeneous 4-vector to the corresponding point in 3-space.
    pub fn homogenize(&self) -> Vec3 {
        self.truncate() / self.w
    }
}

#[cfg(test)]
macro_rules! test_vec {
    ($name:ident, $($dims:ident),+) => {
        #[test]
        fn dimension() {
            assert_eq!($name::dimension(), count_args!($($dims),+));
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
            assert!($name::ones().dot(&$name::ones()).approx_eq(&($name::dimension() as f32)));

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
