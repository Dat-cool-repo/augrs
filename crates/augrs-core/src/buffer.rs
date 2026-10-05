//! Pixel buffers: HWC `ndarray` arrays of a few element types.

use ndarray::{Array3, ArrayView3};

/// A pixel element type. Conversions to/from `f32` are used by interpolating
/// operations; integer conversions round half up and saturate.
pub trait Element: Copy + Send + Sync + Default + PartialEq + std::fmt::Debug + 'static {
    const NAME: &'static str;
    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
    fn from_f64(v: f64) -> Self;
}

impl Element for u8 {
    const NAME: &'static str = "uint8";
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        // `as` saturates (and maps NaN to 0); +0.5 rounds half up for v >= 0.
        (v + 0.5) as u8
    }
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        (v + 0.5) as u8
    }
}

impl Element for u16 {
    const NAME: &'static str = "uint16";
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        (v + 0.5) as u16
    }
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        (v + 0.5) as u16
    }
}

impl Element for i32 {
    const NAME: &'static str = "int32";
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        v.round() as i32
    }
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v.round() as i32
    }
}

impl Element for f32 {
    const NAME: &'static str = "float32";
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        v
    }
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

/// An HWC pixel buffer. Images are `U8` or `F32`; masks may use any variant.
/// Arrays held in a `Buf` are always in standard (C-contiguous) layout.
#[derive(Clone, Debug, PartialEq)]
pub enum Buf {
    U8(Array3<u8>),
    U16(Array3<u16>),
    I32(Array3<i32>),
    F32(Array3<f32>),
}

/// Run `$body` with `$a` bound to the inner array of a `Buf` (by value/ref as given).
#[macro_export]
macro_rules! with_buf {
    ($buf:expr, $a:ident => $body:expr) => {
        match $buf {
            $crate::buffer::Buf::U8($a) => $body,
            $crate::buffer::Buf::U16($a) => $body,
            $crate::buffer::Buf::I32($a) => $body,
            $crate::buffer::Buf::F32($a) => $body,
        }
    };
}

/// Like [`with_buf!`] but re-wraps the result in the same `Buf` variant.
#[macro_export]
macro_rules! map_buf {
    ($buf:expr, $a:ident => $body:expr) => {
        match $buf {
            $crate::buffer::Buf::U8($a) => $crate::buffer::Buf::U8($body),
            $crate::buffer::Buf::U16($a) => $crate::buffer::Buf::U16($body),
            $crate::buffer::Buf::I32($a) => $crate::buffer::Buf::I32($body),
            $crate::buffer::Buf::F32($a) => $crate::buffer::Buf::F32($body),
        }
    };
}

impl Buf {
    /// `(height, width, channels)`.
    pub fn dims(&self) -> (usize, usize, usize) {
        with_buf!(self, a => a.dim())
    }

    pub fn dtype_name(&self) -> &'static str {
        match self {
            Buf::U8(_) => u8::NAME,
            Buf::U16(_) => u16::NAME,
            Buf::I32(_) => i32::NAME,
            Buf::F32(_) => f32::NAME,
        }
    }

    /// Make sure the inner array is in standard layout (copies only if needed).
    pub fn into_standard(self) -> Buf {
        map_buf!(self, a => standard(a))
    }
}

/// Return `a` in standard (C-contiguous) layout.
pub fn standard<T: Element>(a: Array3<T>) -> Array3<T> {
    if a.is_standard_layout() {
        a
    } else {
        a.as_standard_layout().into_owned()
    }
}

/// Copy a (possibly strided) view into a new standard-layout array.
pub fn to_standard<T: Element>(v: ArrayView3<'_, T>) -> Array3<T> {
    if v.is_standard_layout() {
        v.to_owned()
    } else {
        v.as_standard_layout().into_owned()
    }
}

/// Contiguous data of a standard-layout array.
#[inline]
pub(crate) fn data<T: Element>(a: &Array3<T>) -> &[T] {
    a.as_slice().expect("augrs: array must be in standard layout")
}

/// Build an `Array3` from a vector (shape `(h, w, c)`).
#[inline]
pub(crate) fn from_vec<T: Element>(h: usize, w: usize, c: usize, v: Vec<T>) -> Array3<T> {
    Array3::from_shape_vec((h, w, c), v).expect("augrs: shape/vec length mismatch")
}
