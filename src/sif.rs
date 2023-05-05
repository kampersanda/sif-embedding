//! Smooth Inverse Frequency (SIF).
use ndarray::{Array1, Array2};

use crate::util;
use crate::{Float, Lexicon};

/// An implementation of *Smooth Inverse Frequency (SIF)* that is a simple but pewerful
/// embedding technique for sentences, described in the paper:
///
/// > Sanjeev Arora, Yingyu Liang, and Tengyu Ma,
/// > [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/forum?id=SyK00v5xx),
/// > ICLR 2017.
///
/// # Examples
///
/// ```
/// use sif_embedding::{Lexicon, Sif, WordEmbeddings};
///
/// // Load word embeddings from a pretrained model.
/// let word_model = "las 0.0 1.0 2.0\nvegas -3.0 -4.0 -5.0\n";
/// let word_embeddings = WordEmbeddings::from_text(word_model.as_bytes()).unwrap();
///
/// // Prepare word-frequency pairs.
/// let word_weights = [("las", 10.), ("vegas", 20.)];
///
/// // Create a lexicon instance.
/// let lexicon = Lexicon::new(word_embeddings, word_weights);
///
/// // Embed sentences using the lexicon.
/// //
/// let (sent_embeddings, freezed_model) = Sif::new(lexicon).embeddings(["go to las vegas", "mega vegas"]);
/// assert_eq!(sent_embeddings.shape(), &[2, 3]);
///
/// // freezed_model is a model compiled using the first input sentences.
/// let other_embeddings = freezed_model.embeddings(["vegas pro"]);
/// assert_eq!(other_embeddings.shape(), &[1, 3]);
/// assert_eq!(freezed_model.embedding_size(), 3);
/// ```
///
/// See [`FreezedSif`] for more details on `freezed_model`.
#[derive(Debug, Clone)]
pub struct Sif {
    inner: InnerSif,
}

impl Sif {
    /// Creates an instance from a lexicon.
    pub fn new(lexicon: Lexicon) -> Self {
        let inner = InnerSif {
            lexicon,
            separator: ' ',
            param_a: 1e-3,
            n_components: 1,
        };
        Self { inner }
    }

    /// Sets a separator for sentence segmentation (default: the ASCII whitespace ` `).
    pub fn separator(mut self, separator: char) -> Self {
        self.inner.separator = separator;
        self
    }

    /// Sets a weighting parameter `a` (default: `1e-3`).
    pub fn param_a(mut self, param_a: Float) -> Self {
        self.inner.param_a = param_a;
        self
    }

    /// Sets the number of principal components to estimate a common discourse vector `c_0` (default: `1`).
    ///
    /// # Notes
    ///
    /// The original idea uses only the first principal component.
    /// This parameter is for experimental purposes and will not need to be changed.
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.inner.n_components = n_components;
        self
    }

    /// Computes embeddings for the input sentences,
    /// returning a 2D-array of shape `(n_sentences, embedding_size)`
    /// as the first item, where
    ///
    /// - `n_sentences` is the number of input sentences, and
    /// - `embedding_size` is [`Self::embedding_size()`].
    ///
    /// This function consumes itself and, as the second item,
    /// returns a compiled model [`FreezedSif`] for subsequent embeddings.
    pub fn embeddings<I, S>(self, sentences: I) -> (Array2<Float>, FreezedSif)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let sent_embeddings = self.inner.weighted_average_embeddings(sentences);
        // principal_components has shape (embedding_size(), embedding_size())
        let principal_component =
            util::principal_component(&sent_embeddings, self.inner.n_components);
        let sent_embeddings =
            InnerSif::subtract_principal_components(sent_embeddings, &principal_component);
        let freezed_model = FreezedSif {
            inner: self.inner,
            principal_component,
        };
        (sent_embeddings, freezed_model)
    }

    /// Returns the number of dimensions for sentence embeddings,
    /// which is equivalent to that of the input word embedding, i.e., [`Lexicon::embedding_size()`].
    pub fn embedding_size(&self) -> usize {
        self.inner.embedding_size()
    }
}

/// A compiled model of [`Sif`] that maintains the common discourse vector
/// (`c_0` in the paper) estimated from the input sentences of [`Sif::embeddings()`].
///
/// # Notes
///
/// I do not know if the SIF algorithm assumes that a common discourse vector from
/// one corpus will be used for other corpora.
#[derive(Debug, Clone)]
pub struct FreezedSif {
    inner: InnerSif,
    principal_component: Array2<Float>,
}

impl FreezedSif {
    /// Computes embeddings for the input sentences,
    /// returning a 2D-array of shape `(n_sentences, embedding_size)`, where
    ///
    /// - `n_sentences` is the number of input sentences, and
    /// - `embedding_size` is [`Self::embedding_size()`].
    pub fn embeddings<I, S>(&self, sentences: I) -> Array2<Float>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let sent_embeddings = self.inner.weighted_average_embeddings(sentences);
        let sent_embeddings =
            InnerSif::subtract_principal_components(sent_embeddings, &self.principal_component);
        sent_embeddings
    }

    /// Returns the number of dimensions for sentence embeddings,
    /// which is equivalent to that of the input word embedding, i.e., [`Lexicon::embedding_size()`].
    pub fn embedding_size(&self) -> usize {
        self.inner.embedding_size()
    }
}

#[derive(Debug, Clone)]
struct InnerSif {
    lexicon: Lexicon,
    separator: char,
    param_a: Float,
    n_components: usize,
}

impl InnerSif {
    fn embedding_size(&self) -> usize {
        self.lexicon.embedding_size()
    }

    /// Lines 1--3
    fn weighted_average_embeddings<I, S>(&self, sentences: I) -> Array2<Float>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut sent_embeddings = vec![];
        let mut n_sentences = 0;
        for sent in sentences {
            let sent = sent.as_ref();
            let mut n_words = 0;
            let mut sent_embedding = Array1::zeros(self.lexicon.embedding_size());
            for word in sent.split(self.separator) {
                if let Some(word_embedding) = self.lexicon.embedding(word) {
                    let weight = self.param_a / (self.param_a + self.lexicon.probability(word));
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
        Array2::from_shape_vec(
            (n_sentences, self.lexicon.embedding_size()),
            sent_embeddings,
        )
        .unwrap()
    }

    /// Lines 5--7
    fn subtract_principal_components(
        sent_embeddings: Array2<Float>,
        principal_component: &Array2<Float>,
    ) -> Array2<Float> {
        sent_embeddings.to_owned() - &(sent_embeddings.dot(principal_component))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::WordEmbeddings;

    #[test]
    fn test_sif_basic() {
        let model = "A 0.0 1.0 2.0\nBB -3.0 -4.0 -5.0\nCCC 6.0 -7.0 8.0\nDDDD -9.0 10.0 -11.0\n";
        let embeddings = WordEmbeddings::from_text(model.as_bytes()).unwrap();
        let word_weights = [("A", 1.), ("BB", 2.), ("CCC", 3.), ("DDDD", 4.)];

        let lexicon = Lexicon::new(embeddings, word_weights);
        let (se, _) = Sif::new(lexicon).embeddings(&["A BB CCC DDDD", "BB CCC", "A B C", "Z", ""]);
        assert_eq!(se.shape(), &[5, 3]);
    }
}
