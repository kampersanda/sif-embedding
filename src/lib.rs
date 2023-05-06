//! # sif-embedding
//!
//! It provides simple but powerful sentence embedding techniques based on
//! *Smooth Inverse Frequency (SIF)* described in the paper:
//!
//! > Sanjeev Arora, Yingyu Liang, and Tengyu Ma,
//! > [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/forum?id=SyK00v5xx),
//! > ICLR 2017.
//!
//! ## Usage
//!
//! See [README](https://github.com/kampersanda/sif-embedding) for information
//! on how to specify this crate in your dependencies; the backend to be used in
//! [ndarray-linalg](https://github.com/rust-ndarray/ndarray-linalg) must be properly specified.
//!
//! See the document of [`Sif`] for an example on how to compute sentence embeddings.
#![deny(missing_docs)]

pub mod lexicon;
pub mod sif;
pub mod util;
pub mod word_embeddings;

pub use lexicon::Lexicon;
pub use sif::FreezedSif;
pub use sif::Sif;
pub use word_embeddings::WordEmbeddings;

/// Common type of floating numbers.
pub type Float = f32;
