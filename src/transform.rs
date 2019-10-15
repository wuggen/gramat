use super::*;

/// A builder struct for homogeneous transformation matrices.
///
/// A `Transform` is used to construct arbitrary affine transformations starting from the identity
/// transformation, primarily by composing translation, scaling, shear, and rotation
/// transformations. The final matrix is obtained via the `finish()` method.
///
/// # Example
/// ```rust
/// # extern crate gramat;
/// # use gramat::*;
/// use gramat::transform::Transform;
///
/// // `mat` is a matrix that represents the effect of _first_ shearing the x axis 1 unit in the
/// // y direction, _then_ rotating about the y axis 180 degrees.
/// let mat: Mat4 = Transform::new()
///     .shear_x(1.0, 0.0)
///     .rotate(Vec3::y(), Angle::from_degrees(180.0))
///     .finish();
///
/// //assert_approx_eq!(
/// //    (mat * vec3!(1.0, 0.0, 0.0).homogeneous()).homogenize(),
/// //    vec3!(1.0, -1.0, 0.0));
/// ```
///
/// Note that each builder method applies its transformation _after_ those of preceding builder
/// methods. In matrix terms, this corresponds to multiplying the new transformation on the _left_,
/// rather than the right. In other words, the above code computes the following matrix:
///
/// ```plaintext
/// ROT((1, 0, 0), 180°) * SHEAR_X(1)
/// ```
///
/// where `ROT(axis, angle)` is a rotation about axis `axis` by angle `angle` and `SHEAR_X(dist)`
/// is a shear by distance `dist` that fixes the x-axis.
pub struct Transform {
    mat: Mat4,
}

impl Transform {
    /// Create a new `Transform`.
    ///
    /// `Tranform`s initially represent the identity transformation (i.e. no transformation at all),
    /// and are built into more useful transformations via other methods on the struct.
    #[inline(always)]
    pub fn new() -> Transform {
        Transform {
            mat: Mat4::identity(),
        }
    }

    /// Translate by the given offset.
    #[inline(always)]
    pub fn translate(self, offset: Vec3) -> Transform {
        Transform {
            mat: &translate(offset) * &self.mat,
        }
    }

    /// Scale by the given factors.
    ///
    /// The scaling is performed independently per-axis, using the corresponding factor from the
    /// factor vector.
    #[inline(always)]
    pub fn scale(self, factor: Vec3) -> Transform {
        Transform {
            mat: &scale(factor) * &self.mat,
        }
    }

    /// Shear by the given amount, fixing the _yz_ plane.
    ///
    /// This will shear the _x_ axis by the given amounts along the _y_ and _z_ axes.
    #[inline(always)]
    pub fn shear_x(self, y_amount: f32, z_amount: f32) -> Transform {
        Transform {
            mat: &shear_x(y_amount, z_amount) * &self.mat,
        }
    }

    /// Shear by the given amount, fixing the _xz_ plane.
    ///
    /// This will shear the _y_ axis by the given amounts along the _x_ and _z_ axes.
    #[inline(always)]
    pub fn shear_y(self, x_amount: f32, z_amount: f32) -> Transform {
        Transform {
            mat: &shear_y(x_amount, z_amount) * &self.mat,
        }
    }

    /// Shear by the given amount, fixing the _xy_ plane.
    ///
    /// This will shear the _z_ axis by the given amounts along the _x_ and _y_ axes.
    #[inline(always)]
    pub fn shear_z(self, x_amount: f32, y_amount: f32) -> Transform {
        Transform {
            mat: &shear_z(x_amount, y_amount) * &self.mat,
        }
    }

    /// Rotate about the given axis by the given angle.
    #[inline(always)]
    pub fn rotate(self, axis: Vec3, angle: Angle) -> Transform {
        Transform {
            mat: &rotate(axis, angle) * &self.mat,
        }
    }

    /// Apply an arbitrary affine transformation, represented by a homogenous matrix.
    #[inline(always)]
    pub fn arbitrary(self, transform: Mat4) -> Transform {
        Transform {
            mat: &transform * &self.mat,
        }
    }

    /// Consume the `Transform` and acquire the resulting homogeneous transformation matrix.
    #[inline(always)]
    pub fn finish(&self) -> Mat4 {
        self.mat
    }
}

/// Get the homogeneous transformation matrix of a translation by the given offset.
pub fn translate(offset: Vec3) -> Mat4 {
    let offset = offset.extend(1.0);
    let mut mat = Mat4::identity();
    mat.set_col(3, offset);
    mat
}

/// Get the homogeneous transformation matrix of a scale by the given factors.
///
/// Scaling is computed independently per-axis, using the corresponding factors in the given
/// vector.
pub fn scale(factor: Vec3) -> Mat4 {
    let mut mat = Mat4::identity();
    mat[0][0] = factor[0];
    mat[1][1] = factor[1];
    mat[2][2] = factor[2];
    mat
}

/// Get the homogeneous transformation matrix of a shear fixing the _yz_ plane by the given amounts
/// parallel to the _y_ and _z_ axes.
pub fn shear_x(y_amount: f32, z_amount: f32) -> Mat4 {
    let mut m = Mat4::identity();
    m[0][1] = y_amount;
    m[0][2] = z_amount;
    m
}

/// Get the homogeneous transformation matrix of a shear fixing the _xz_ plane by the given amounts
/// parallel to the _x_ and _z_ axes.
pub fn shear_y(x_amount: f32, z_amount: f32) -> Mat4 {
    let mut m = Mat4::identity();
    m[1][0] = x_amount;
    m[1][2] = z_amount;
    m
}

/// Get the homogeneous transformation matrix of a shear fixing the _xy_ plane by the given amounts
/// parallel to the _x_ and _y_ axes.
pub fn shear_z(x_amount: f32, y_amount: f32) -> Mat4 {
    let mut m = Mat4::identity();
    m[2][0] = x_amount;
    m[2][1] = y_amount;
    m
}

/// Get the homogeneous transformation matrix of a rotation about the given axis by the given
/// angle.
pub fn rotate(axis: Vec3, angle: Angle) -> Mat4 {
    let half = angle / 2.0;
    let w = half.cos();
    let v = half.sin() * axis.unit();

    let xy = v.x * v.y;
    let xz = v.x * v.z;
    let xw = v.x * w;
    let x2 = v.x * v.x;
    let yz = v.y * v.z;
    let yw = v.y * w;
    let y2 = v.y * v.y;
    let zw = v.z * w;
    let z2 = v.z * v.z;

    Mat4::new(
        Vec4::new(1.0 - 2.0 * (y2 + z2), 2.0 * (xy + zw), 2.0 * (xz - yw), 0.0),
        Vec4::new(2.0 * (xy - zw), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz + xw), 0.0),
        Vec4::new(2.0 * (xz + yw), 2.0 * (yz - xw), 1.0 - 2.0 * (x2 + y2), 0.0),
        Vec4::w(),
    )
}

/// Build a look-at view matrix.
///
/// # Parameters
/// * `eye` The position of the camera.
/// * `center` The point towards which the camera is facing.
/// * `up` A vector in the upwards direction, usually `vec3!(0.0, 0.0, 1.0)`.
///
/// # Errors and Panics
/// There is a singularity that results in a divide by zero when the (`eye` - `center`) vector is
/// parallel to the `up` vector.
#[allow(unused_variables)]
pub fn look_at(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
    let facing = center - eye;
    let facing_unit = facing.unit();
    let up_unit = up.unit();
    // TODO
    unimplemented!()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_util::*;

    #[test]
    fn test_translate() {
        let test_func = |v: Vec3, offset| {
            let expected = v + offset;
            let v = v.homogeneous();

            let m = translate(offset);
            let vt = m * v;
            let vt = vt.homogenize();
            assert_approx_eq!(
                vt,
                expected,
                "Failure with v = {:?}, offset = {:?}. Expected {:?}, got {:?}.",
                v,
                offset,
                expected,
                vt
            );
        };

        for v in GenVec3::new(-4, 4) {
            for o in GenVec3::new(-4, 4) {
                test_func(v, o);
            }
        }
    }

    #[test]
    fn test_scale() {
        let test_func = |v: Vec3, s: Vec3| {
            let expected = vec3!(v[0] * s[0], v[1] * s[1], v[2] * s[2]);
            let v = v.homogeneous();

            let m = scale(s);
            let vt = m * v;
            let vt = vt.homogenize();
            assert_approx_eq!(
                vt,
                expected,
                "Failure with v = {:?}, scale = {:?}. Expected {:?}, got {:?}.",
                v,
                s,
                expected,
                vt
            );
        };

        for v in GenVec3::new(-4, 4) {
            for s in GenVec3::new(-4, 4) {
                test_func(v, s);
            }
        }
    }

    #[test]
    fn test_shear() {
        let test_func = |v: Vec3, amt1, amt2| {
            let expectedx = vec3!(v[0], v[1] + v[0]*amt1, v[2] + v[0]*amt2);
            let expectedy = vec3!(v[0] + v[1]*amt1, v[1], v[2] + v[1]*amt2);
            let expectedz = vec3!(v[0] + v[2]*amt1, v[1] + v[2]*amt2, v[2]);

            let v = v.homogeneous();

            let mx = shear_x(amt1, amt2);
            let my = shear_y(amt1, amt2);
            let mz = shear_z(amt1, amt2);

            let vtx = (mx * v).homogenize();
            let vty = (my * v).homogenize();
            let vtz = (mz * v).homogenize();

            assert_approx_eq!(vtx, expectedx,
                "Failure with v = {:?}, shear_x({}, {}). Expected {:?}, got {:?}.",
                v, amt1, amt2, expectedx, vtx);
            assert_approx_eq!(vty, expectedy,
                "Failure with v = {:?}, shear_y({}, {}). Expected {:?}, got {:?}.",
                v, amt1, amt2, expectedy, vty);
            assert_approx_eq!(vtz, expectedz,
                "Failure with v = {:?}, shear_z({}, {}). Expected {:?}, got {:?}.",
                v, amt1, amt2, expectedz, vtz);
        };

        for v in GenVec3::new(-4, 4) {
            for amt1 in -4..=4 {
                for amt2 in -4..=4 {
                    test_func(v, amt1 as f32, amt2 as f32);
                }
            }
        }
    }

    #[test]
    #[cfg(dont_compile_this_lol)]
    fn test_rotate() {
        let _test_func = |v: Vec3, a: Angle| {
            // TODO
        };
    }
}
