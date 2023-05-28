//! # sif-embedding
//!
//! This crate provides simple but powerful sentence embedding techniques based on
//! *Smooth Inverse Frequency (SIF)* described in the paper:
//!
//! > Sanjeev Arora, Yingyu Liang, and Tengyu Ma,
//! > [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/forum?id=SyK00v5xx),
//! > ICLR 2017.
//!
//! ## Specifications
//!
//! This crate depends on [ndarray-linalg](https://github.com/rust-ndarray/ndarray-linalg).
//! You must *always* specify which backend will be used with `features`, following the specifications of ndarray-linalg.
//! See [README of ndarray-linalg v0.16.0](https://github.com/rust-ndarray/ndarray-linalg/tree/ndarray-linalg-v0.16.0) since the feature names of sif-embedding are the same.
//!
//! For example, you can specify the [OpenBLAS](https://www.openblas.net/) backend as follows:
//!
//! ```toml
//! # Cargo.toml
//!
//! [dependencies]
//! sif-embedding = { version = "0.1", features = ["openblas"] }
//! ```
//!
//! If you are having problems compiling this library due to the backend, [my tips](https://github.com/kampersanda/sif-embedding/wiki/Trouble-shooting) may help.
//!
//! ## Basic usage
//!
//! See the document of [`Sif`].
#![deny(missing_docs)]

pub mod lexicon;
pub mod sif;
pub mod util;
// pub mod word_embeddings;

pub use lexicon::Lexicon;
pub use sif::FreezedSif;
pub use sif::Sif;
// pub use word_embeddings::WordEmbeddings;

/// Common type of floating numbers.
pub type Float = f32;
