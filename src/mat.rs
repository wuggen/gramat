//! Two, three, and four-dimensional real square matrices with `f32` components.

use super::*;
use std::convert::*;
use std::ops::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub trait SquareMatrix: Copy {
    /// The type used to represent the columns and rows of the matrix type.
    type VecType;

    /// The number of rows (or columns) of this matrix.
    const DIMS: usize;

    /// Produce a matrix whose elements are all 1's.
    fn ones() -> Self;

    /// Produce a matrix whose elements are all 0's.
    fn zeros() -> Self;

    /// Produce an identity matrix, i.e. a matrix with 1's on the main diagonal and 0's elsewhere.
    fn identity() -> Self;

    /// Get the transpose of this matrix.
    fn transpose(&self) -> Self;

    /// Get the determinant of this matrix.
    fn determinant(&self) -> f32;

    /// Get the inverse of this matrix.
    fn inverse(&self) -> Self;

    /// Get the `row`-`col` minor of this matrix.
    ///
    /// This is the determinant of the matrix produced by omitting the `row`'th row and the
    /// `col`'th column of this matrix.
    ///
    /// # Panics
    ///
    /// This function panics if either `col` or `row` is out of bounds of the matrix.
    fn minor(&self, col: usize, row: usize) -> f32;

    /// Get the `row`-`col` cofactor of this matrix.
    ///
    /// This is the `row`-`col` minor, multiplied by 1 or -1 depending on the parity of
    /// `row`+`col`.
    ///
    /// # Panics
    ///
    /// This function panics if either `col` or `row` is out of bounds of the matrix.
    fn cofactor(&self, col: usize, row: usize) -> f32;

    /// Get the `i`'th column of this matrix.
    ///
    /// # Panics
    ///
    /// This function panics if `i` is out of bounds.
    fn get_col(&self, i: usize) -> Self::VecType;

    /// Get the `i`'th row of this matrix.
    ///
    /// # Panics
    ///
    /// This function panics if `i` is out of bounds.
    fn get_row(&self, i: usize) -> Self::VecType;

    /// Set the `i`'th column of this matrix to match the given vector.
    ///
    /// # Panics
    ///
    /// This function panics if `i` is out of bounds.
    fn set_col(&mut self, i: usize, c: Self::VecType);

    /// Set the `i`'th row of this matrix to match the given vector.
    ///
    /// # Panics
    ///
    /// This function panics if `i` is out of bounds.
    fn set_row(&mut self, i: usize, r: Self::VecType);
}

macro_rules! decl_mat {
    ($name:ident, $coltype:ident, $($cols:ident),+ | $($dims:ident),+) => {
        #[repr(C)]
        #[derive(Debug, PartialEq, Clone, Copy)]
        #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
        pub struct $name {
            $($cols: $coltype),+
        }

        impl $name {
            #[doc = "Construct a new matrix with the given vectors as columns."]
            #[inline(always)]
            pub fn new($($cols: $coltype),+) -> $name {
                $name {
                    $($cols),+
                }
            }
        }

        impl SquareMatrix for $name {
            type VecType = $coltype;

            const DIMS: usize = count_args!($($cols),+);

            fn ones() -> Self {
                $name {
                    $($cols: $coltype::ones()),+
                }
            }

            fn zeros() -> Self {
                $name {
                    $($cols: $coltype::zeros()),+
                }
            }

            fn identity() -> Self {
                $name {
                    $($cols: $coltype::$dims()),+
                }
            }

            fn transpose(&self) -> Self {
                let mut m = $name::zeros();
                for (i, col) in self.as_ref().iter().enumerate() {
                    m.set_row(i, col.clone());
                }
                m
            }

            #[inline(always)]
            fn determinant(&self) -> f32 {
                self.matrix_determinant()
            }

            #[inline(always)]
            fn inverse(&self) -> Self {
                self.matrix_inverse()
            }

            #[inline(always)]
            fn minor(&self, col: usize, row: usize) -> f32 {
                self.matrix_minor(col, row)
            }

            fn cofactor(&self, col: usize, row: usize) -> f32 {
                let neg = ((col + row) & 1) as f32 * (-2.0) + 1.0;
                neg * self.minor(col, row)
            }

            fn get_col(&self, i: usize) -> $coltype {
                assert!(i < $name::DIMS, "Column index out of bounds");
                self[i].clone()
            }

            fn get_row(&self, i: usize) -> $coltype {
                assert!(i < $name::DIMS, "Row index out of bounds");
                $coltype {
                    $($dims: self.$cols[i]),+
                }
            }

            fn set_col(&mut self, i: usize, c: $coltype) {
                self[i] = c;
            }

            fn set_row(&mut self, i: usize, r: $coltype) {
                $(self.$cols[i] = r.$dims;)+
            }
        }

        impl AsRef<[$coltype]> for $name {
            #[doc = "View this matrix as a slice containing its column vectors."]
            #[inline(always)]
            fn as_ref(&self) -> &[$coltype] {
                unsafe { &*(self as *const $name as *const [$coltype; count_args!($($cols),+)]) }
            }
        }

        impl AsMut<[$coltype]> for $name {
            #[doc = "View this matrix as a slice containing its column vectors."]
            #[inline(always)]
            fn as_mut(&mut self) -> &mut [$coltype] {
                unsafe { &mut *(self as *mut $name as *mut [$coltype; count_args!($($cols),+)]) }
            }
        }

        impl Index<usize> for $name {
            type Output = $coltype;

            #[doc = "Index this matrix as a slice containing its column vectors."]
            #[inline(always)]
            fn index(&self, idx: usize) -> &$coltype {
                &self.as_ref()[idx]
            }
        }

        impl IndexMut<usize> for $name {
            #[doc = "Index this matrix as a slice containing its column vectors."]
            #[inline(always)]
            fn index_mut(&mut self, idx: usize) -> &mut $coltype {
                &mut self.as_mut()[idx]
            }
        }

        impl ApproxEq for $name {
            fn approx_eq(&self, rhs: &$name) -> bool {
                $(self.$cols.approx_eq(&rhs.$cols))&+
            }

            #[doc = "Compare two matrices for approximate equality, using a third matrix"]
            #[doc = "for component-wise thresholds."]
            fn within_threshold(&self, rhs: &$name, threshold: &$name) -> bool {
                $(self.$cols.within_threshold(&rhs.$cols, &threshold.$cols))&+
            }
        }

        decl_mat!(@MATOP $name, Add, add, +, $($cols),+);
        decl_mat!(@MATOP $name, Sub, sub, -, $($cols),+);

        decl_mat!(@MATASGN $name, AddAssign, add_assign, +=, $($cols),+);
        decl_mat!(@MATASGN $name, SubAssign, sub_assign, -=, $($cols),+);

        decl_mat!(@VECMUL $name, $coltype, $($cols),+);

        decl_mat!(@MATMUL $name, $($cols),+);

        decl_mat!(@SCALAR_MUL $name, $name, $($cols),+);
        decl_mat!(@SCALAR_MUL $name, &$name, $($cols),+);

        decl_mat!(@SCALAR_DIV $name, $name, $($cols),+);
        decl_mat!(@SCALAR_DIV $name, &$name, $($cols),+);

        decl_mat!(@MULASGN $name, f32, $($cols),+);
        decl_mat!(@MULASGN $name, &f32, $($cols),+);

        decl_mat!(@DIVASGN $name, f32, $($cols),+);
        decl_mat!(@DIVASGN $name, &f32, $($cols),+);

        decl_mat!(@NEG $name, $name, $($cols),+);
        decl_mat!(@NEG $name, &$name, $($cols),+);
    };

    (@MATOP $name:ident, $trait:ident, $func:ident, $op:tt, $($cols:ident),+) => {
        decl_mat!(@MATOP_SINGLE $name, $trait, $func, $op, $name, $name, $($cols),+);
        decl_mat!(@MATOP_SINGLE $name, $trait, $func, $op, $name, &$name, $($cols),+);
        decl_mat!(@MATOP_SINGLE $name, $trait, $func, $op, &$name, $name, $($cols),+);
        decl_mat!(@MATOP_SINGLE $name, $trait, $func, $op, &$name, &$name, $($cols),+);
    };
    (@MATOP_SINGLE $name:ident, $trait:ident, $func:ident, $op:tt, $lhs:ty, $rhs:ty, $($cols:ident),+) => {
        impl $trait<$rhs> for $lhs {
            type Output = $name;

            #[doc = "Component-wise arithmetic operation."]
            fn $func(self, rhs: $rhs) -> $name {
                $name {
                    $($cols: &self.$cols $op &rhs.$cols),+
                }
            }
        }
    };

    (@VECMUL $name:ident, $vectype:ident, $($cols:ident),+) => {
        decl_mat!(@VECMUL_SINGLE $name, $vectype, $name, $vectype, $($cols),+);
        decl_mat!(@VECMUL_SINGLE $name, $vectype, $name, &$vectype, $($cols),+);
        decl_mat!(@VECMUL_SINGLE $name, $vectype, &$name, $vectype, $($cols),+);
        decl_mat!(@VECMUL_SINGLE $name, $vectype, &$name, &$vectype, $($cols),+);
    };
    (@VECMUL_SINGLE $name:ident, $vectype:ident, $lhs:ty, $rhs:ty, $($cols:ident),+) => {
        impl Mul<$rhs> for $lhs {
            type Output = $vectype;

            #[doc = "Matrix-vector multiplication operation."]
            fn mul(self, rhs: $rhs) -> $vectype {
                let mut res = $vectype::zeros();
                for col in 0..$name::DIMS {
                    res += &rhs[col] * &self[col];
                }
                res
            }
        }
    };

    (@MATMUL $name:ident, $($cols:ident),+) => {
        decl_mat!(@MATMUL_SINGLE $name, $name, $name, $($cols),+);
        decl_mat!(@MATMUL_SINGLE $name, $name, &$name, $($cols),+);
        decl_mat!(@MATMUL_SINGLE $name, &$name, $name, $($cols),+);
        decl_mat!(@MATMUL_SINGLE $name, &$name, &$name, $($cols),+);
    };
    (@MATMUL_SINGLE $name:ident, $lhs:ty, $rhs:ty, $($cols:ident),+) => {
        impl Mul<$rhs> for $lhs {
            type Output = $name;

            #[doc = "Matrix-matrix multiplication operation."]
            fn mul(self, rhs: $rhs) -> $name {
                $name {
                    $($cols: self * &rhs.$cols),+
                }
            }
        }
    };

    (@MATASGN $name:ident, $trait:ident, $func:ident, $op:tt, $($cols:ident),+) => {
        decl_mat!(@MATASGN_SINGLE $name, $trait, $func, $op, $name, $($cols),+);
        decl_mat!(@MATASGN_SINGLE $name, $trait, $func, $op, &$name, $($cols),+);
    };
    (@MATASGN_SINGLE $name:ident, $trait:ident, $func:ident, $op:tt, $rhs:ty, $($cols:ident),+) => {
        impl $trait<$rhs> for $name {
            #[doc = "Component-wise arithmetic assignment operation."]
            fn $func(&mut self, rhs: $rhs) {
                $(self.$cols $op &rhs.$cols;)+
            }
        }
    };

    (@SCALAR_MUL $name:ident, $type:ty, $($cols:ident),+) => {
        impl Mul<f32> for $type {
            type Output = $name;

            fn mul(self, rhs: f32) -> $name {
                $name {
                    $($cols: &self.$cols * rhs),+
                }
            }
        }

        impl Mul<$type> for f32 {
            type Output = $name;

            fn mul(self, rhs: $type) -> $name {
                $name {
                    $($cols: &rhs.$cols * self),+
                }
            }
        }
    };

    (@MULASGN $name:ident, $rhs:ty, $($cols:ident),+) => {
        impl MulAssign<$rhs> for $name {
            fn mul_assign(&mut self, rhs: $rhs) {
                $(self.$cols *= rhs;)+
            }
        }
    };

    (@SCALAR_DIV $name:ident, $type:ty, $($cols:ident),+) => {
        impl Div<f32> for $type {
            type Output = $name;

            fn div(self, rhs: f32) -> $name {
                $name {
                    $($cols: &self.$cols / rhs),+
                }
            }
        }
    };

    (@DIVASGN $name:ident, $rhs:ty, $($cols:ident),+) => {
        impl DivAssign<$rhs> for $name {
            fn div_assign(&mut self, rhs: $rhs) {
                $(self.$cols /= rhs;)+
            }
        }
    };

    (@NEG $name:ident, $type:ty, $($cols:ident),+) => {
        impl Neg for $type {
            type Output = $name;

            fn neg(self) -> $name {
                $name {
                    $($cols: -&self.$cols),+
                }
            }
        }
    };
}

decl_mat!(Mat2, Vec2, col1, col2 | x, y);
decl_mat!(Mat3, Vec3, col1, col2, col3 | x, y, z);
decl_mat!(Mat4, Vec4, col1, col2, col3, col4 | x, y, z, w);

impl Mat2 {
    fn matrix_determinant(&self) -> f32 {
        self[0][0] * self[1][1] - self[1][0] * self[0][1]
    }

    fn matrix_inverse(&self) -> Mat2 {
        Mat2::new(
            Vec2::new(self[1][1], -self[0][1]),
            Vec2::new(-self[1][0], self[0][0]),
        ) / self.determinant()
    }

    fn matrix_minor(&self, col: usize, row: usize) -> f32 {
        assert!(col < 2, "[Mat2::matrix_minor] Column index out of bounds");
        assert!(row < 2, "[Mat2::matrix_minor] Row index out of bounds");

        self[col ^ 1][row ^ 1]
    }
}

impl Mat3 {
    fn matrix_determinant(&self) -> f32 {
        self[0][0] * (self[1][1] * self[2][2] - self[2][1] * self[1][2])
            + self[1][0] * (self[2][1] * self[0][2] - self[0][1] * self[2][2])
            + self[2][0] * (self[0][1] * self[1][2] - self[1][1] * self[0][2])
    }

    fn matrix_minor(&self, col: usize, row: usize) -> f32 {
        assert!(col < 3, "[Mat3::minor] Column index out of bounds");
        assert!(row < 3, "[Mat3::minor] Row index out of bounds");

        let mut m = Mat2::zeros();
        for i in 0..2 {
            let c = if i >= col { i + 1 } else { i };
            for j in 0..2 {
                let r = if j >= row { j + 1 } else { j };

                m[i][j] = self[c][r];
            }
        }

        m.determinant()
    }

    fn matrix_inverse(&self) -> Mat3 {
        let c00 = self.cofactor(0, 0);
        let c01 = self.cofactor(0, 1);
        let c02 = self.cofactor(0, 2);
        let c10 = self.cofactor(1, 0);
        let c11 = self.cofactor(1, 1);
        let c12 = self.cofactor(1, 2);
        let c20 = self.cofactor(2, 0);
        let c21 = self.cofactor(2, 1);
        let c22 = self.cofactor(2, 2);

        Mat3::new(
            Vec3::new(c00, c10, c20),
            Vec3::new(c01, c11, c21),
            Vec3::new(c02, c12, c22),
        ) / self.determinant()
    }
}

impl Mat4 {
    fn matrix_minor(&self, col: usize, row: usize) -> f32 {
        assert!(col < 4, "[Mat4::minor] Column index out of bounds");
        assert!(row < 4, "[Mat4::minor] Row index out of bounds");

        let mut m = Mat3::zeros();
        for i in 0..3 {
            let c = if i >= col { i + 1 } else { i };
            for j in 0..3 {
                let r = if j >= row { j + 1 } else { j };

                m[i][j] = self[c][r];
            }
        }

        m.determinant()
    }

    fn first_row_minors(&self) -> (f32, f32, f32, f32) {
        (
            Mat3::new(
                Vec3::new(self[1][1], self[1][2], self[1][3]),
                Vec3::new(self[2][1], self[2][2], self[2][3]),
                Vec3::new(self[3][1], self[3][2], self[3][3]),
            )
            .determinant(),
            Mat3::new(
                Vec3::new(self[0][1], self[0][2], self[0][3]),
                Vec3::new(self[2][1], self[2][2], self[2][3]),
                Vec3::new(self[3][1], self[3][2], self[3][3]),
            )
            .determinant(),
            Mat3::new(
                Vec3::new(self[0][1], self[0][2], self[0][3]),
                Vec3::new(self[1][1], self[1][2], self[1][3]),
                Vec3::new(self[3][1], self[3][2], self[3][3]),
            )
            .determinant(),
            Mat3::new(
                Vec3::new(self[0][1], self[0][2], self[0][3]),
                Vec3::new(self[1][1], self[1][2], self[1][3]),
                Vec3::new(self[2][1], self[2][2], self[2][3]),
            )
            .determinant(),
        )
    }

    fn det_minors(&self, m0: f32, m1: f32, m2: f32, m3: f32) -> f32 {
        self[0][0] * m0 - self[1][0] * m1 + self[2][0] * m2 - self[3][0] * m3
    }

    fn matrix_determinant(&self) -> f32 {
        let (m0, m1, m2, m3) = self.first_row_minors();
        self.det_minors(m0, m1, m2, m3)
    }

    fn matrix_inverse(&self) -> Mat4 {
        let c00 = self.cofactor(0, 0);
        let c01 = self.cofactor(0, 1);
        let c02 = self.cofactor(0, 2);
        let c03 = self.cofactor(0, 3);
        let c10 = self.cofactor(1, 0);
        let c11 = self.cofactor(1, 1);
        let c12 = self.cofactor(1, 2);
        let c13 = self.cofactor(1, 3);
        let c20 = self.cofactor(2, 0);
        let c21 = self.cofactor(2, 1);
        let c22 = self.cofactor(2, 2);
        let c23 = self.cofactor(2, 3);
        let c30 = self.cofactor(3, 0);
        let c31 = self.cofactor(3, 1);
        let c32 = self.cofactor(3, 2);
        let c33 = self.cofactor(3, 3);

        Mat4::new(
            Vec4::new(c00, c10, c20, c30),
            Vec4::new(c01, c11, c21, c31),
            Vec4::new(c02, c12, c22, c32),
            Vec4::new(c03, c13, c23, c33),
        ) / self.determinant()
    }
}

#[cfg(test)]
macro_rules! test_mat {
    ($name:ident, $vec:ident, $($dims:ident),+) => {
        #[test]
        fn dimension() {
            assert_eq!($name::DIMS, count_args!($($dims),+));
        }

        #[test]
        fn ones() {
            let m = $name::ones();
            for i in 0..$name::DIMS {
                for j in 0..$name::DIMS {
                    assert!(m[i][j].approx_eq(&1.0));
                }
            }
        }

        #[test]
        fn zeros() {
            let m = $name::zeros();
            for i in 0..$name::DIMS {
                for j in 0..$name::DIMS {
                    assert!(m[i][j].approx_eq(&0.0));
                }
            }
        }

        #[test]
        fn identity() {
            let m = $name::identity();
            for i in 0..$name::DIMS {
                for j in 0..$name::DIMS {
                    if i == j {
                        assert!(m[i][j].approx_eq(&1.0));
                    } else {
                        assert!(m[i][j].approx_eq(&0.0));
                    }
                }
            }
        }

        #[test]
        fn transpose() {
            let mut m = $name::zeros();
            for i in 0..$name::DIMS {
                for j in 0..$name::DIMS {
                    if i >= j {
                        m[i][j] = 1.0;
                    }
                }
            }

            let m = m.transpose();
            for i in 0..$name::DIMS {
                for j in 0..$name::DIMS {
                    if i <= j {
                        assert!(m[i][j].approx_eq(&1.0));
                    } else {
                        assert!(m[i][j].approx_eq(&0.0));
                    }
                }
            }
        }

        #[test]
        fn identity_determinant() {
            let i = $name::identity();
            assert!(i.determinant().approx_eq(&1.0), "Det = {}", i.determinant());

            let i = 3.0 * i;
            assert!(i.determinant().approx_eq(&(3.0_f32).powi(count_args!($($dims),+))),
                "Det = {}", i.determinant());
        }

        #[test]
        fn identity_inverse() {
            let i = $name::identity();
            let i_inv = i.inverse();

            assert!(i.approx_eq(&i_inv));
            assert!((&i * &i_inv).approx_eq(&$name::identity()));
            assert!((&i_inv * &i).approx_eq(&$name::identity()));

            let i = 2.0 * i;
            let i_inv = i.inverse();

            assert!(i_inv.approx_eq(&(0.5 * $name::identity())), "i_inv: {:?}", i_inv);
            assert!((&i * &i_inv).approx_eq(&$name::identity()));
            assert!((&i_inv * &i).approx_eq(&$name::identity()));
        }

        #[test]
        fn zero_determinant() {
            let mut v = $vec::ones();
            let mut m = $name::zeros();
            for i in 0..count_args!($($dims),+) {
                m.set_col(i, v.clone());
                v += $vec::ones();
            }

            assert!(m.determinant().approx_eq(&0.0));
        }

        #[test]
        fn inverse() {
            let mut v = $vec::ones();
            let mut m = $name::zeros();
            let n = count_args!($($dims),+);
            for i in 0..n {
                m.set_col(i, v.clone());
                v *= 2.0;
            }

            m.set_row(0, &v + m.get_row(0));
            m.set_row(1, &v - m.get_col(1));
            if n > 2 {
                m.set_row(2, &v + m.get_col(2) - m.get_row(1));
            }

            assert!(!m.determinant().approx_eq(&0.0), "det: {}", m.determinant());

            let m_inv = m.inverse();
            assert!((&m * &m_inv).approx_eq(&$name::identity()),
                "m: {:?}\ni: {:?}\nm * i: {:?}", m, m_inv, &m * &m_inv);
            assert!((&m_inv * &m).approx_eq(&$name::identity()),
                "m: {:?}\ni: {:?}\ni * m: {:?}", m, m_inv, &m_inv * &m);
        }
    };
}

#[cfg(test)]
mod test_mat2 {
    use super::*;

    test_mat!(Mat2, Vec2, x, y);
}

#[cfg(test)]
mod test_mat3 {
    use super::*;

    test_mat!(Mat3, Vec3, x, y, z);

    const MAT: Mat3 = Mat3 {
        col1: Vec3 {
            x: 3.0,
            y: 2.0,
            z: 0.0,
        },
        col2: Vec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
        col3: Vec3 {
            x: 2.0,
            y: -2.0,
            z: 1.0,
        },
    };

    #[test]
    fn minors() {
        let expected = [2.0, -2.0, 0.0, 2.0, 3.0, -10.0, 2.0, 3.0, 0.0];

        for i in 0..3 {
            for j in 0..3 {
                let k = i * 3 + j;
                assert!(
                    expected[k].approx_eq(&MAT.minor(i, j)),
                    "Minor {},{}: expected {}, actually {}",
                    i,
                    j,
                    expected[k],
                    MAT.minor(i, j)
                );
            }
        }
    }

    #[test]
    fn matrix_inverse_specific() {}
}

#[cfg(test)]
mod test_mat4 {
    use super::*;

    test_mat!(Mat4, Vec4, x, y, z, w);

    const MAT: Mat4 = Mat4 {
        col1: Vec4 {
            x: 1.0,
            y: 0.0,
            z: 2.0,
            w: 2.0,
        },
        col2: Vec4 {
            x: 0.0,
            y: 2.0,
            z: 1.0,
            w: 0.0,
        },
        col3: Vec4 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
            w: 1.0,
        },
        col4: Vec4 {
            x: 1.0,
            y: 2.0,
            z: 1.0,
            w: 4.0,
        },
    };

    #[test]
    fn minors_specific() {
        let expected = [
            -4.0, 1.0, 2.0, -1.0, -2.0, 1.0, 0.0, -1.0, -16.0, 2.0, 4.0, -4.0, -6.0, 1.0, 2.0, -1.0,
        ];

        for i in 0..4 {
            for j in 0..4 {
                let k = i * 4 + j;
                assert!(
                    expected[k].approx_eq(&MAT.minor(i, j)),
                    "Minor {},{}: expected {}, actually {}",
                    i,
                    j,
                    expected[k],
                    MAT.minor(i, j)
                );
            }
        }
    }

    #[test]
    fn cofactors_specific() {
        let expected = [
            -4.0, -1.0, 2.0, 1.0, 2.0, 1.0, 0.0, -1.0, -16.0, -2.0, 4.0, 4.0, 6.0, 1.0, -2.0, -1.0,
        ];

        for i in 0..4 {
            for j in 0..4 {
                let k = i * 4 + j;
                assert!(
                    expected[k].approx_eq(&MAT.cofactor(i, j)),
                    "Cofactor {},{}: expected {}, actually {}",
                    i,
                    j,
                    expected[k],
                    MAT.cofactor(i, j)
                );
            }
        }
    }

    #[test]
    fn determinant_specific() {
        assert!(
            MAT.determinant().approx_eq(&2.0),
            "Determinant: expected {}, actually {}",
            2.0,
            MAT.determinant()
        );
    }

    #[test]
    fn inverse_specific() {
        let expected = Mat4::new(
            Vec4::new(-2.0, 1.0, -8.0, 3.0),
            Vec4::new(-0.5, 0.5, -1.0, 0.5),
            Vec4::new(1.0, 0.0, 2.0, -1.0),
            Vec4::new(0.5, -0.5, 2.0, -0.5),
        );
        let actual = MAT.inverse();

        assert!(
            actual.approx_eq(&expected),
            "Inverse: expected {:?}, actually {:?}",
            expected,
            actual
        );
        assert!((&MAT * &actual).approx_eq(&Mat4::identity()));
        assert!((&actual * &MAT).approx_eq(&Mat4::identity()));
    }
}
