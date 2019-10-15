#![allow(dead_code)]

use crate::vec::{Vec2, Vec3, Vec4};

use std::ops::RangeInclusive;

type Range = RangeInclusive<i16>;

pub struct GenVec2 {
    range: Range,
    x: i16,
    y: i16,
}

impl GenVec2 {
    pub fn new(lower: i16, upper: i16) -> GenVec2 {
        let range = Range::new(lower, upper);
        let x = *range.start();
        let y = *range.start();

        GenVec2 { range, x, y }
    }
}

impl Iterator for GenVec2 {
    type Item = Vec2;

    fn next(&mut self) -> Option<Vec2> {
        let (lower, upper) = (*self.range.start(), *self.range.end());
        if self.x > upper {
            return None;
        }

        let v = vec2!(self.x, self.y);

        if self.y <= upper {
            self.y += 1;
        } else {
            self.y = lower;
            self.x += 1;
        }

        Some(v)
    }
}

pub struct GenVec3 {
    range: Range,
    x: i16,
    y: i16,
    z: i16,
}

impl GenVec3 {
    pub fn new(lower: i16, upper: i16) -> GenVec3 {
        let range = Range::new(lower, upper);
        let x = *range.start();
        let y = *range.start();
        let z = *range.start();

        GenVec3 { range, x, y, z }
    }
}

impl Iterator for GenVec3 {
    type Item = Vec3;

    fn next(&mut self) -> Option<Vec3> {
        let (lower, upper) = (*self.range.start(), *self.range.end());
        if self.x > upper {
            return None;
        }

        let v = vec3!(self.x, self.y, self.z);

        if self.z <= upper {
            self.z += 1;
        } else if self.y <= upper {
            self.z = lower;
            self.y += 1;
        } else {
            self.z = lower;
            self.y = lower;
            self.x += 1;
        }

        Some(v)
    }
}

pub struct GenVec4 {
    range: Range,
    x: i16,
    y: i16,
    z: i16,
    w: i16,
}

impl GenVec4 {
    pub fn new(lower: i16, upper: i16) -> GenVec4 {
        let range = Range::new(lower, upper);
        let x = *range.start();
        let y = *range.start();
        let z = *range.start();
        let w = *range.start();

        GenVec4 { range, x, y, z, w }
    }
}

impl Iterator for GenVec4 {
    type Item = Vec4;

    fn next(&mut self) -> Option<Vec4> {
        let (lower, upper) = (*self.range.start(), *self.range.end());
        if self.x > upper {
            return None;
        }

        let v = vec4!(self.x, self.y, self.z, self.w);

        if self.w <= upper {
            self.w += 1;
        } else if self.z <= upper {
            self.w = lower;
            self.z += 1;
        } else if self.y <= upper {
            self.w = lower;
            self.z = lower;
            self.y += 1;
        } else {
            self.w = lower;
            self.z = lower;
            self.y = lower;
            self.x += 1;
        }

        Some(v)
    }
}
