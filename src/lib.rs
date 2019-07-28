//! A simple, `f32`-based <b>gra</b>phics <b>mat</b>h library.
//!
//! Includes two, three, and four-dimensional real vectors and matrices, quaternions, and
//! assorted utilities for transformations and projections.

macro_rules! replace_tt {
    ($_t:tt, $sub:expr) => {
        $sub
    };
}

macro_rules! count_args {
    ($($args:ident),*) => { 0 $(+ replace_tt!($args, 1))* };
}

pub mod angle;
pub mod mat;
pub mod quaternion;
pub mod transform;
pub mod util;
pub mod vec;

pub use angle::*;
pub use mat::*;
pub use quaternion::*;
pub use util::*;
pub use vec::*;
