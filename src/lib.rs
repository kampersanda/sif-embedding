//! # sif-embedding
//!
//! This crate provides simple but powerful sentence embedding techniques based on
//! *Smooth Inverse Frequency (SIF)* described in the paper:
//!
//! > Sanjeev Arora, Yingyu Liang, and Tengyu Ma,
//! > [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/forum?id=SyK00v5xx),
//! > ICLR 2017.
//!
//! ## Basic usage
//!
//! See the document of [`Sif`].
//!
//! ## Backend specifications
//!
//! This crate depends on [ndarray-linalg](https://github.com/rust-ndarray/ndarray-linalg) and
//! allows you to specify any backend supported by ndarray-linalg through `features`.
//! The feature names in sif-embedding are the same as those in [ndarray-linalg v0.16.0](https://github.com/rust-ndarray/ndarray-linalg/tree/ndarray-linalg-v0.16.0).
//! The default is `openblas` and uses the [OpenBLAS](https://www.openblas.net/) backend.
//! **Note that you must specify one backend.**
//!
//! If you are having problems compiling this library due to the backend, [my tips](https://github.com/kampersanda/sif-embedding/wiki/Trouble-shooting) may help.
#![deny(missing_docs)]

// These declarations are required so that finalfusion recognizes the backend.
// c.f. https://github.com/finalfusion/finalfusion-utils
#[cfg(any(
    feature = "intel-mkl",
    feature = "intel-mkl-static",
    feature = "intel-mkl-system"
))]
extern crate intel_mkl_src;
#[cfg(any(
    feature = "netlib",
    feature = "netlib-static",
    feature = "netlib-system"
))]
extern crate netlib_src;
#[cfg(any(
    feature = "openblas",
    feature = "openblas-static",
    feature = "openblas-system"
))]
extern crate openblas_src;

pub mod sif;
pub mod unigram;
pub mod util;

pub use sif::Sif;
pub use unigram::UnigramLM;

/// Common type of floating numbers.
pub type Float = f32;
