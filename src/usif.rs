//! Smooth Inverse Frequency (SIF).
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
pub struct Usif<'w, 'u, W, U> {
    separator: char,
    param_a: Float,
    word_embeddings: &'w W,
    unigram_lm: &'u U,
}

impl<'w, 'u, W, U> Usif<'w, 'u, W, U>
where
    W: WordEmbeddings,
    U: UnigramLanguageModel,
{
    /// Creates a new instance.
    pub const fn new(word_embeddings: &'w W, unigram_lm: &'u U) -> Self {
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
        self.word_embeddings.embedding_size()
    }

    fn estimate_param_a(&self, avg_sent_len: Float) -> Float {
        let n_words = self.word_embeddings.n_words() as Float;
        let threshold = 1. - (1. - (1. / n_words)).powf(avg_sent_len);
        let n_greater = self
            .word_embeddings
            .words()
            .iter()
            .filter(|word| self.unigram_lm.probability(word) > threshold)
            .count() as Float;
        let alpha = n_greater / n_words;
        let partiion = n_words / 2.;
        let param_a = (1. - alpha) / (alpha * partiion);
        param_a
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
