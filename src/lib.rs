//! # sif-embedding
//!
//! This crate provides simple but powerful sentence embedding techniques based on
//! *Smooth Inverse Frequency* and *Common Component Removal* described in the following papers:
//!
//! - Sanjeev Arora, Yingyu Liang, and Tengyu Ma,
//!   [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/forum?id=SyK00v5xx),
//!   ICLR 2017
//! - Kawin Ethayarajh,
//!   [Unsupervised Random Walk Sentence Embeddings: A Strong but Simple Baseline](https://aclanthology.org/W18-3012/),
//!   RepL4NLP 2018
//!
//! ## Getting started
//!
//! Given models of word embeddings and unigram probabilities,
//! sif-embedding can immediately compute sentence embeddings.
//!
//! This crate does not have any dependency limitations on using the input models;
//! however, using [finalfusion](https://docs.rs/finalfusion/) and [wordfreq](https://docs.rs/wordfreq/)
//! would be the easiest and most reasonable way
//! because these libraries can handle various pre-trained models and are pluged into this crate.
//! See [the instructions](#instructions-pre-trained-models) to install the libraries in this crate.
//!
//! [`Sif`] or [`USif`] implements the techniques of sentence embeddings.
//! [`SentenceEmbedder`] defines the behavior of sentence embeddings.
//! The following code shows an example to compute sentence embeddings using finalfusion and wordfreq.
//!
//! ```
//! use std::io::BufReader;
//!
//! use finalfusion::compat::text::ReadText;
//! use finalfusion::embeddings::Embeddings;
//! use wordfreq::WordFreq;
//!
//! use sif_embedding::{Sif, SentenceEmbedder};
//!
//! // Creates word embeddings from a pretrained model.
//! let word_model = "las 0.0 1.0 2.0\nvegas -3.0 -4.0 -5.0\n";
//! let mut reader = BufReader::new(word_model.as_bytes());
//! let word_embeddings = Embeddings::read_text(&mut reader).unwrap();
//!
//! // Creates a unigram language model.
//! let word_weights = [("las", 10.), ("vegas", 20.)];
//! let unigram_lm = WordFreq::new(word_weights);
//!
//! // Computes sentence embeddings in shape (n, m),
//! // where n is the number of sentences and m is the number of dimensions.
//! let sif = Sif::new(&word_embeddings, &unigram_lm);
//! let (sent_embeddings, _) = sif.fit_embeddings(&["go to las vegas", "mega vegas"]).unwrap();
//! assert_eq!(sent_embeddings.shape(), &[2, 3]);
//! ```
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
//! version = "0.4"
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
//! sif-embedding, or the SIF algorithm, requires two pre-trained models as input:
//!
//! - Word embeddings
//! - Unigram language models
//!
//! You can use arbitrary models in this crate through the [`WordEmbeddings`] and [`UnigramLanguageModel`] traits.
//!
//! This crate already implements the traits for the two external libraries:
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
//! version = "0.4"
//! features = ["finalfusion", "wordfreq"]
//! ```
//!
//! A tutorial to learn how to use external pre-trained models in finalfusion and wordfreq can be found
//! [here](https://github.com/kampersanda/sif-embedding/tree/main/examples/tutorial).
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

/// Common type of floating numbers.
pub type Float = f32;

use anyhow::Result;
use ndarray::{Array2, CowArray, Ix1};

/// Word embeddings.
pub trait WordEmbeddings {
    /// Returns the embedding of a word.
    fn embedding(&self, word: &str) -> Option<CowArray<Float, Ix1>>;

    /// Returns the number of dimension of the word embeddings.
    fn embedding_size(&self) -> usize;
}

/// Unigram language model.
pub trait UnigramLanguageModel {
    /// Returns the probability of a word.
    fn probability(&self, word: &str) -> Float;

    /// Returns the number of words in the vocabulary.
    fn n_words(&self) -> usize;

    /// Returns the iterator over words and probabilities in the vocabulary.
    fn entries(&self) -> Box<dyn Iterator<Item = (String, Float)> + '_>;
}

/// Sentence Embeddings.
pub trait SentenceEmbedder: Sized {
    /// Returns the number of dimensions for sentence embeddings.
    fn embedding_size(&self) -> usize;

    ///
    fn fit<S>(self, sentences: &[S]) -> Result<Self>
    where
        S: AsRef<str>;

    /// Computes embeddings for input sentences,
    /// returning a 2D-array of shape `(n_sentences, embedding_size)`, where
    ///
    /// - `n_sentences` is the number of input sentences, and
    /// - `embedding_size` is [`Self::embedding_size()`].
    fn embeddings<I, S>(&self, sentences: I) -> Result<Array2<Float>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>;

    ///
    fn fit_embeddings<S>(self, sentences: &[S]) -> Result<(Array2<Float>, Self)>
    where
        S: AsRef<str>,
    {
        let model = self.fit(sentences)?;
        let embeddings = model.embeddings(sentences)?;
        Ok((embeddings, model))
    }
}
