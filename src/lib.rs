//! A simple, `f32`-based <b>gra</b>phics <b>mat</b>h library.
//!
//! Includes two, three, and four-dimensional real vectors and matrices, quaternions, and
//! assorted utilities for transformations and projections.

#![doc(html_root_url = "https://docs.rs/gramit/0.1.0")]

macro_rules! replace_tt {
    ($_t:tt, $sub:expr) => {
        $sub
    };
}

macro_rules! count_args {
    ($($args:ident),*) => { 0 $(+ replace_tt!($args, 1))* };
}

/// Assert that two expressions are approximately equal according to their [`ApproxEq`]
/// implementations.
///
/// Passes additional parameters on to the standard library `assert!` macro, so custom format
/// messages are accepted as per usual.
///
/// # Example
/// ```
/// # #[macro_use] extern crate gramit;
/// use gramit::ApproxEq;
/// # fn main() {
/// assert_approx_eq!(1.0_f32, 1.0_f32);
///
/// assert_approx_eq!(1.0, 1.0 + 0.5 * gramit::fp::EQ_THRESHOLD_F32,
///     "{} and {} were not approximately equal; this shouldn't happen!",
///     1.0,
///     1.0 + 0.5 * gramit::fp::EQ_THRESHOLD_F32);
/// # }
/// ```
///
/// [`ApproxEq`]: fp/trait.ApproxEq.html
#[macro_export]
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr, $($msg:expr),*) => {
        assert!($a.approx_eq(&$b), $($msg),*)
    };

    ($a:expr, $b:expr) => { assert_approx_eq!($a, $b,) };
}

/// Assert that two expressions are within a threshold specified by the third parameter, according
/// to their [`ApproxEq`] implementations.
///
/// Passes additional parameters on to the standard library `assert!` macro, so custom format
/// messages are accepted as per usual.
///
/// # Example
/// ```
/// # #[macro_use] extern crate gramit;
/// use gramit::ApproxEq;
/// # fn main() {
/// assert_within_threshold!(1.0, 1.5, 0.75);
/// assert_within_threshold!(0.0, 0.1, 0.2,
///     "{} and {} were not within distance {} from each other; this shouldn't happen!",
///     0.0, 0.1, 0.2);
/// # }
/// ```
///
/// ```should_panic
/// # #[macro_use] extern crate gramit;
/// use gramit::ApproxEq;
/// # fn main() {
/// // Panics!
/// assert_within_threshold!(1.0, 2.0, 0.5);
/// # }
/// ```
///
/// [`ApproxEq`]: fp/trait.ApproxEq.html
#[macro_export]
macro_rules! assert_within_threshold {
    ($a:expr, $b:expr, $thresh:expr, $($msg:expr),*) => {
        assert!($a.within_threshold(&$b, &$thresh), $($msg),*)
    };

    ($a:expr, $b:expr, $thresh:expr) => { assert_within_threshold!($a, $b, $thresh,) };
}

/// Construct a [`Vec2`] with the given components.
///
/// The components are consumed and can be given as any type for which `Into<f32>` is implemented.
///
/// [`Vec2`]: vec/struct.Vec2.html
#[macro_export]
macro_rules! vec2 {
    ($x:expr, $y:expr) => {
        Vec2::new(f32::from($x), f32::from($y))
    };
}

/// Construct a [`Vec3`] with the given components.
///
/// The components are consumed and can be given as any type for which `Into<f32>` is implemented.
///
/// [`Vec3`]: vec/struct.Vec2.html
#[macro_export]
macro_rules! vec3 {
    ($x:expr, $y:expr, $z:expr) => {
        Vec3::new(f32::from($x), f32::from($y), f32::from($z))
    };
}

/// Construct a [`Vec4`] with the given components.
///
/// The components are consumed and can be given as any type for which `Into<f32>` is implemented.
///
/// [`Vec4`]: vec/struct.Vec2.html
#[macro_export]
macro_rules! vec4 {
    ($x:expr, $y:expr, $z:expr, $w:expr) => {
        Vec4::new(f32::from($x), f32::from($y), f32::from($z), f32::from($w))
    };
}

pub mod angle;
pub mod mat;
pub mod quaternion;
pub mod transform;
pub mod fp;
pub mod vec;

pub use angle::Angle;
pub use mat::{SquareMatrix, Mat2, Mat3, Mat4};
pub use quaternion::Quaternion;
pub use fp::ApproxEq;
pub use vec::{Vector, Vec2, Vec3, Vec4};

#[cfg(test)]
mod test_util;
