//! Smooth Inverse Frequency (SIF).
use finalfusion::embeddings::Embeddings;
use finalfusion::storage::Storage;
use finalfusion::vocab::Vocab;
use ndarray::{Array1, Array2};

use crate::{util, Float, UnigramLM};

const N_COMPONENTS: usize = 1;

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
/// use std::io::BufReader;
///
/// use finalfusion::compat::text::ReadText;
/// use finalfusion::embeddings::Embeddings;
///
/// use sif_embedding::{Sif, UnigramLM};
///
/// // Load word embeddings from a pretrained model.
/// let word_model = "las 0.0 1.0 2.0\nvegas -3.0 -4.0 -5.0\n";
/// let mut reader = BufReader::new(word_model.as_bytes());
/// let word_embeddings = Embeddings::read_text(&mut reader).unwrap();
///
/// // Create a unigram language model.
/// let word_weights = [("las", 10.), ("vegas", 20.)];
/// let unigram_lm = UnigramLM::new(word_weights);
///
/// // Compute sentence embeddings.
/// let sif = Sif::new(&word_embeddings, &unigram_lm);
/// let sent_embeddings = sif.embeddings(["go to las vegas", "mega vegas"]);
/// assert_eq!(sent_embeddings.shape(), &[2, 3]);
/// ```
#[derive(Debug, Clone)]
pub struct Sif<'w, 'u, V, T> {
    separator: char,
    param_a: Float,
    word_embeddings: &'w Embeddings<V, T>,
    unigram_lm: &'u UnigramLM,
}

impl<'w, 'u, V, T> Sif<'w, 'u, V, T>
where
    V: Vocab,
    T: Storage,
{
    /// Creates a new instance.
    pub const fn new(word_embeddings: &'w Embeddings<V, T>, unigram_lm: &'u UnigramLM) -> Self {
        Self {
            separator: ' ',
            param_a: 1e-3,
            word_embeddings,
            unigram_lm,
        }
    }

    /// Sets a separator for sentence segmentation (default: ASCII whitespace).
    pub const fn separator(mut self, separator: char) -> Self {
        self.separator = separator;
        self
    }

    /// Sets a SIF-weighting parameter `a` (default: `1e-3`).
    pub const fn param_a(mut self, param_a: Float) -> Self {
        self.param_a = param_a;
        self
    }

    /// Computes embeddings for input sentences,
    /// returning a 2D-array of shape `(n_sentences, embedding_size)`, where
    ///
    /// - `n_sentences` is the number of input sentences, and
    /// - `embedding_size` is [`Self::embedding_size()`].
    pub fn embeddings<I, S>(&self, sentences: I) -> Array2<Float>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let sent_embeddings = self.weighted_average_embeddings(sentences);
        if sent_embeddings.is_empty() {
            return sent_embeddings;
        }
        let common_component = util::principal_component(&sent_embeddings, N_COMPONENTS);
        Self::subtract_common_components(sent_embeddings, &common_component)
    }

    /// Returns the number of dimensions for sentence embeddings,
    /// which is equivalent to that of the input word embeddings.
    pub fn embedding_size(&self) -> usize {
        self.word_embeddings.dims()
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
            let mut sent_embedding = Array1::zeros(self.embedding_size());
            for word in sent.split(self.separator) {
                if let Some(word_embedding) = self.word_embeddings.embedding(word) {
                    let weight = self.param_a / (self.param_a + self.unigram_lm.probability(word));
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
        Array2::from_shape_vec((n_sentences, self.embedding_size()), sent_embeddings).unwrap()
    }

    /// Lines 5--7
    fn subtract_common_components(
        sent_embeddings: Array2<Float>,
        common_component: &Array2<Float>,
    ) -> Array2<Float> {
        sent_embeddings.to_owned() - &(sent_embeddings.dot(common_component))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::BufReader;

    use finalfusion::compat::text::ReadText;
    use finalfusion::embeddings::Embeddings;

    #[test]
    fn test_embeddings() {
        let model = "A 0.0 1.0 2.0\nBB -3.0 -4.0 -5.0\nCCC 6.0 -7.0 8.0\nDDDD -9.0 10.0 -11.0\n";
        let mut reader = BufReader::new(model.as_bytes());
        let word_embeddings = Embeddings::read_text(&mut reader).unwrap();

        let word_weights = [("A", 1.), ("BB", 2.), ("CCC", 3.), ("DDDD", 4.)];
        let unigram_lm = UnigramLM::new(word_weights);

        let sif = Sif::new(&word_embeddings, &unigram_lm);

        let sent_embeddings = sif.embeddings(["A BB CCC DDDD", "BB CCC", "A B C", "Z", ""]);
        assert_eq!(sent_embeddings.shape(), &[5, 3]);

        let sent_embeddings = sif.embeddings(Vec::<&str>::new());
        assert_eq!(sent_embeddings.shape(), &[0, 3]);

        let sent_embeddings = sif.embeddings(["", ""]);
        assert_eq!(sent_embeddings.shape(), &[2, 3]);
    }
}
