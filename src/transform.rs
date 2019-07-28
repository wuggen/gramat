use super::*;

pub struct Transform {
    mat: Mat4,
}

impl Transform {
    #[inline(always)]
    pub fn new() -> Transform {
        Transform {
            mat: Mat4::identity(),
        }
    }

    #[inline(always)]
    pub fn translate(self, offset: Vec3) -> Transform {
        Transform {
            mat: &translate(offset) * &self.mat,
        }
    }

    #[inline(always)]
    pub fn scale(self, factor: Vec3) -> Transform {
        Transform {
            mat: &scale(factor) * &self.mat,
        }
    }

    #[inline(always)]
    pub fn rotate(self, axis: Vec3, angle: Angle) -> Transform {
        Transform {
            mat: &rotate(axis, angle) * &self.mat,
        }
    }

    #[inline(always)]
    pub fn arbitrary(self, transform: Mat4) -> Transform {
        Transform {
            mat: &transform * &self.mat,
        }
    }

    #[inline(always)]
    pub fn finish(self) -> Mat4 {
        self.mat
    }
}

pub fn translate(offset: Vec3) -> Mat4 {
    let offset = offset.extend(1.0);
    let mut mat = Mat4::identity();
    mat.set_col(3, offset);
    mat
}

pub fn scale(factor: Vec3) -> Mat4 {
    let mut mat = Mat4::identity();
    mat[0][0] = factor[0];
    mat[1][1] = factor[1];
    mat[2][2] = factor[2];
    mat
}

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
