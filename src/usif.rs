//! Smooth Inverse Frequency (SIF).
use anyhow::Ok;
use anyhow::{anyhow, Result};
use ndarray::Array1;
use ndarray::Array2;

use crate::util;
use crate::Float;
use crate::UnigramLanguageModel;
use crate::WordEmbeddings;

const N_COMPONENTS: usize = 5;

/// uSIF
///
/// Unsupervised Random Walk Sentence Embeddings: A Strong but Simple Baseline
/// https://aclanthology.org/W18-3012/
#[derive(Clone)]
pub struct USif<'w, 'u, W, U> {
    word_embeddings: &'w W,
    unigram_lm: &'u U,
    separator: char,
    param_a: Option<Float>,
    singular_vectors: Option<Array2<Float>>,
    singular_weights: Option<Array1<Float>>,
}

impl<'w, 'u, W, U> USif<'w, 'u, W, U>
where
    W: WordEmbeddings,
    U: UnigramLanguageModel,
{
    /// Creates a new instance.
    pub const fn new(word_embeddings: &'w W, unigram_lm: &'u U) -> Self {
        Self {
            word_embeddings,
            unigram_lm,
            separator: ' ',
            param_a: None,
            singular_vectors: None,
            singular_weights: None,
        }
    }

    /// Sets a separator for sentence segmentation (default: ASCII whitespace).
    pub const fn separator(mut self, separator: char) -> Self {
        self.separator = separator;
        self
    }

    ///
    pub fn fit<S>(mut self, sentences: &[S]) -> Result<Self>
    where
        S: AsRef<str>,
    {
        if sentences.is_empty() {
            return Err(anyhow!("no sentences"));
        }
        let sent_len = self.average_sentence_length(sentences);
        self.param_a = Some(self.estimate_param_a(sent_len));
        Ok(self)
    }

    /// Computes embeddings for input sentences,
    /// returning a 2D-array of shape `(n_sentences, embedding_size)`, where
    ///
    /// - `n_sentences` is the number of input sentences, and
    /// - `embedding_size` is [`Self::embedding_size()`].
    pub fn embeddings<I, S>(&self, sentences: I) -> Result<Array2<Float>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let sent_embeddings = self.weighted_embeddings(sentences)?;
        Ok(sent_embeddings)
    }

    /// Compute the average sentence length.
    /// (Line 3 in Algorithm 1)
    fn average_sentence_length<S>(&self, sentences: &[S]) -> Float
    where
        S: AsRef<str>,
    {
        let mut n_words = 0;
        for sent in sentences {
            let sent = sent.as_ref();
            n_words += sent.split(self.separator).count();
        }
        n_words as Float / sentences.len() as Float
    }

    /// Estimate the parameter `a` for the weight function.
    /// (Lines 5--7 in Algorithm 1)
    fn estimate_param_a(&self, sent_len: Float) -> Float {
        let n_words = self.word_embeddings.n_words() as Float;
        let threshold = 1. - (1. - (1. / n_words)).powf(sent_len);
        let n_greater = self
            .word_embeddings
            .words()
            .iter()
            .filter(|word| self.unigram_lm.probability(word) > threshold)
            .count() as Float;
        let alpha = n_greater / n_words;
        let partiion = n_words / 2.;
        (1. - alpha) / (alpha * partiion)
    }

    /// Line 8 in Algorithm 1
    fn weighted_embeddings<I, S>(&self, sentences: I) -> Result<Array2<Float>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        if self.param_a.is_none() {
            return Err(anyhow!("not fitted"));
        }
        let param_a = self.param_a.unwrap();
        let mut sent_embeddings = vec![];
        let mut n_sentences = 0;
        for sent in sentences {
            let sent = sent.as_ref();
            let mut n_words = 0;
            let mut sent_embedding = Array1::zeros(self.embedding_size());
            for word in sent.split(self.separator) {
                if let Some(word_embedding) = self.word_embeddings.embedding(word) {
                    let pw = self.unigram_lm.probability(word);
                    let weight = param_a / (pw + 0.5 * param_a);
                    sent_embedding += &(word_embedding.to_owned() * weight);
                    n_words += 1;
                }
            }
            if n_words != 0 {
                sent_embedding /= n_words as Float;
            }
            sent_embeddings.extend(sent_embedding.iter());
            n_sentences += 1;
        }
        Ok(Array2::from_shape_vec((n_sentences, self.embedding_size()), sent_embeddings).unwrap())
    }

    /// Returns the number of dimensions for sentence embeddings,
    /// which is equivalent to that of the input word embeddings.
    pub fn embedding_size(&self) -> usize {
        self.word_embeddings.embedding_size()
    }
}
