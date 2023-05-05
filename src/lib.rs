//! # sif-embedding
//!
//! It provides simple but powerful sentence embedding techniques based on
//! *Smooth Inverse Frequency (SIF)* described in the paper:
//!
//! > Sanjeev Arora, Yingyu Liang, and Tengyu Ma,
//! > [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/forum?id=SyK00v5xx),
//! > ICLR 2017.
//!
//! See the document of [`Sif`] for the basic usage.
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
