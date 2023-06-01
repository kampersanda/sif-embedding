//! # sif-embedding
//!
//! This crate provides simple but powerful sentence embedding techniques based on
//! *Smooth Inverse Frequency (SIF)* described in the paper:
//!
//! > Sanjeev Arora, Yingyu Liang, and Tengyu Ma,
//! > [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/forum?id=SyK00v5xx),
//! > ICLR 2017.
//!
//! ## Examples
//!
//! See the document of [`Sif`].
//!
//! ## Instructions
//!
//! This crate depends on [ndarray-linalg](https://github.com/rust-ndarray/ndarray-linalg) and
//! allows you to specify any backend supported by ndarray-linalg.
//! **You must always specify one of the features** from:
//!
//! - `openblas-static` (or alias `openblas`)
//! - `openblas-system`
//! - `netlib-static` (or alias `netlib`)
//! - `netlib-system`
//! - `intel-mkl-static` (or alias `intel-mkl`)
//! - `intel-mkl-system`
//!
//! The feature names correspond to those of ndarray-linalg (v0.16.0).
//! Refer to [the documentation](https://github.com/rust-ndarray/ndarray-linalg/tree/ndarray-linalg-v0.16.0)
//! for the specifications.
//!
//! For example, if you want to use the [OpenBLAS](https://www.openblas.net/) backend with static linking,
//! specify the dependencies as follows:
//!
//! ```toml
//! # Cargo.toml
//!
//! [features]
//! default = ["openblas-static"]
//! openblas-static = ["sif-embedding/openblas-static", "openblas-src/static"]
//!
//! [dependencies]
//! sif-embedding = "0.3"
//!
//! [dependencies.openblas-src]
//! version = "0.10.4"
//! optional = true
//! default-features = false
//! features = ["cblas"]
//! ```
//!
//! In addition, declare `openblas-src` at the root of your crate as follows:
//!
//! ```
//! // main.rs / lib.rs
//! #[cfg(feature = "openblas-static")]
//! extern crate openblas_src as _src;
//! ```
//!
//! See [`examples/toy`](https://github.com/kampersanda/sif-embedding/tree/main/examples/toy) for a complete example.
//!
//! ### Tips
//!
//! If you are having problems compiling this library due to the backend,
//! [my tips](https://github.com/kampersanda/sif-embedding/wiki/Trouble-shooting) may help.
#![deny(missing_docs)]

// These declarations are required to recognize the backend.
// https://github.com/rust-ndarray/ndarray-linalg/blob/ndarray-linalg-v0.16.0/lax/src/lib.rs
#[cfg(any(feature = "intel-mkl-static", feature = "intel-mkl-system"))]
extern crate intel_mkl_src as _src;
#[cfg(any(feature = "netlib-static", feature = "netlib-system"))]
extern crate netlib_src as _src;
#[cfg(any(feature = "openblas-static", feature = "openblas-system"))]
extern crate openblas_src as _src;

pub mod sif;
pub mod unigram;
pub mod util;

pub use sif::Sif;
pub use unigram::UnigramLM;

/// Common type of floating numbers.
pub type Float = f32;
