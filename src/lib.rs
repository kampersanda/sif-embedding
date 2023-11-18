//! # sif-embedding
//!
//! This crate provides simple but powerful sentence embedding algorithms based on
//! *Smooth Inverse Frequency* and *Common Component Removal* described in the following papers:
//!
//! - Sanjeev Arora, Yingyu Liang, and Tengyu Ma,
//!   [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/forum?id=SyK00v5xx),
//!   ICLR 2017
//! - Kawin Ethayarajh,
//!   [Unsupervised Random Walk Sentence Embeddings: A Strong but Simple Baseline](https://aclanthology.org/W18-3012/),
//!   RepL4NLP 2018
//!
//! This library will help you if
//!
//! - DNN-based sentence embeddings are too slow for your application, or
//! - you do not have an option using GPUs.
//!
//! ## Getting started
//!
//! Given models of word embeddings and probabilities,
//! sif-embedding can immediately compute sentence embeddings.
//!
//! This crate does not have any dependency limitations on using the input models;
//! however, using [finalfusion](https://docs.rs/finalfusion/) and [wordfreq](https://docs.rs/wordfreq/)
//! would be the easiest and most reasonable way
//! because these libraries can handle various pre-trained models and are pluged into this crate.
//! See [the instructions](#instructions-pre-trained-models) to install the libraries in this crate.
//!
//! [`Sif`] or [`USif`] implements the algorithms of sentence embeddings.
//! [`SentenceEmbedder`] defines the behavior of sentence embeddings.
//! The following code shows an example to compute sentence embeddings using finalfusion and wordfreq.
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use std::io::BufReader;
//!
//! use finalfusion::compat::text::ReadText;
//! use finalfusion::embeddings::Embeddings;
//! use wordfreq::WordFreq;
//!
//! use sif_embedding::{Sif, SentenceEmbedder};
//!
//! // Loads word embeddings from a pretrained model.
//! let word_embeddings_text = "las 0.0 1.0 2.0\nvegas -3.0 -4.0 -5.0\n";
//! let mut reader = BufReader::new(word_embeddings_text.as_bytes());
//! let word_embeddings = Embeddings::read_text(&mut reader)?;
//!
//! // Loads word probabilities from a pretrained model.
//! let word_probs = WordFreq::new([("las", 0.4), ("vegas", 0.6)]);
//!
//! // Prepares input sentences.
//! let sentences = ["las vegas", "mega vegas"];
//!
//! // Fits the model with input sentences.
//! let model = Sif::new(&word_embeddings, &word_probs);
//! let model = model.fit(&sentences)?;
//!
//! // Computes sentence embeddings in shape (n, m),
//! // where n is the number of sentences and m is the number of dimensions.
//! let sent_embeddings = model.embeddings(sentences)?;
//! assert_eq!(sent_embeddings.shape(), &[2, 3]);
//! # Ok(())
//! # }
//! ```
//!
//! `model.embeddings` requires memory of `O(n_sentences * n_dimensions)`.
//! If your input sentences are too large to fit in memory,
//! you can compute sentence embeddings in a batch manner.
//!
//! ```ignore
//! for batch in sentences.chunks(batch_size) {
//!     let sent_embeddings = model.embeddings(batch)?;
//!     ...
//! }
//! ```
//!
//! ## Feature specifications
//!
//! This crate provides the following features:
//!
//! - Backend features
//!   - `openblas-static` (or alias `openblas`)
//!   - `openblas-system`
//!   - `netlib-static` (or alias `netlib`)
//!   - `netlib-system`
//!   - `intel-mkl-static` (or alias `intel-mkl`)
//!   - `intel-mkl-system`
//! - Pre-trained model features
//!   - `finalfusion`
//!   - `wordfreq`
//!
//! No feature is enabled by default.
//! The descriptions of the features can be found below.
//!
//! ## Instructions: Backend specifications
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
//! [dependencies.sif-embedding]
//! version = "0.6"
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
//!
//! #[cfg(feature = "openblas-static")]
//! extern crate openblas_src as _src;
//! ```
//!
//! ## Instructions: Pre-trained models
//!
//! The embedding techniques require two pre-trained models as input:
//!
//! - Word embeddings
//! - Word probabilities
//!
//! You can use arbitrary models through the [`WordEmbeddings`] and [`WordProbabilities`] traits.
//!
//! This crate already implements these traits for the two external libraries:
//!
//! - [finalfusion (v0.17)](https://docs.rs/finalfusion/): Library to handle different types of word embeddings such as Glove and fastText.
//! - [wordfreq (v0.2)](https://docs.rs/wordfreq/): Library to look up the frequencies of words in many languages.
//!
//! To enable the features, specify the dependencies as follows:
//!
//! ```toml
//! # Cargo.toml
//!
//! [dependencies.sif-embedding]
//! version = "0.6"
//! features = ["finalfusion", "wordfreq"]
//! ```
//!
//! A tutorial to learn how to use external pre-trained models in finalfusion and wordfreq can be found
//! [here](https://github.com/kampersanda/sif-embedding/tree/main/tutorial).
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
pub mod usif;
pub mod util;

#[cfg(feature = "finalfusion")]
pub mod finalfusion;
#[cfg(feature = "wordfreq")]
pub mod wordfreq;

pub use sif::Sif;
pub use usif::USif;

use anyhow::Result;
use ndarray::{Array2, CowArray, Ix1};

/// Common type of floating numbers.
pub type Float = f32;

/// Default separator for splitting sentences into words.
pub const DEFAULT_SEPARATOR: char = ' ';

/// Default number of samples to fit.
pub const DEFAULT_N_SAMPLES_TO_FIT: usize = 1 << 16;

/// Word embeddings.
pub trait WordEmbeddings {
    /// Returns the embedding of a word.
    fn embedding(&self, word: &str) -> Option<CowArray<Float, Ix1>>;

    /// Returns the number of dimension of the word embeddings.
    fn embedding_size(&self) -> usize;
}

/// Word probabilities.
pub trait WordProbabilities {
    /// Returns the probability of a word.
    fn probability(&self, word: &str) -> Float;

    /// Returns the number of words in the vocabulary.
    fn n_words(&self) -> usize;

    /// Returns an iterator over words and probabilities in the vocabulary.
    fn entries(&self) -> Box<dyn Iterator<Item = (String, Float)> + '_>;
}

/// Common behavior of our models for sentence embeddings.
pub trait SentenceEmbedder: Sized {
    /// Returns the number of dimensions for sentence embeddings.
    fn embedding_size(&self) -> usize;

    /// Fits the model with input sentences.
    fn fit<S>(self, sentences: &[S]) -> Result<Self>
    where
        S: AsRef<str>;

    /// Computes embeddings for input sentences using the fitted model.
    fn embeddings<I, S>(&self, sentences: I) -> Result<Array2<Float>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>;
}
