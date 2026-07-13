//! Quantizers: sign-bit binary packing and symmetric linear int8.
//!
//! Both target L2-normalized embedding corpora (zero-centered, so int8 uses
//! no zero-point — a zero-point would cost a correction term in every
//! kernel for nothing on this data).
//!
//! Binary codes follow the layout contract in [`crate::kernels`]: MSB-first
//! within each byte, padding bits zeroed. The bit rule is `x > 0.0` strictly
//! (both `+0.0` and `-0.0` pack as 0; NaN packs as 0), matching the numpy
//! ecosystem's `packbits(embeddings > 0)`.

mod binary;
mod int8;
mod parity;

pub use binary::{pack_sign_bits, pack_sign_bits_vec, unpack_sign_bits};
pub use int8::{dequantize_i8, fixed_scale, max_abs_scale, quantize_i8, quantize_i8_vec};
pub use parity::{
    BinaryParity, Int8Parity, binary_parity, int8_parity, voyage_offset_binary_to_codes,
};
