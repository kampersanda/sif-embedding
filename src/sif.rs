//! Smooth Inverse Frequency (SIF).
use ndarray::{Array1, Array2};

use crate::{util, Float, UnigramLanguageModel, WordEmbeddings};

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
/// use wordfreq::WordFreq;
///
/// use sif_embedding::Sif;
///
/// // Load word embeddings from a pretrained model.
/// let word_model = "las 0.0 1.0 2.0\nvegas -3.0 -4.0 -5.0\n";
/// let mut reader = BufReader::new(word_model.as_bytes());
/// let word_embeddings = Embeddings::read_text(&mut reader).unwrap();
///
/// // Create a unigram language model.
/// let word_weights = [("las", 10.), ("vegas", 20.)];
/// let unigram_lm = WordFreq::new(word_weights);
///
/// // Compute sentence embeddings.
/// let sif = Sif::new(&word_embeddings, &unigram_lm);
/// let sent_embeddings = sif.embeddings(["go to las vegas", "mega vegas"]);
/// assert_eq!(sent_embeddings.shape(), &[2, 3]);
/// ```
#[derive(Clone)]
pub struct Sif<'w, 'u, W, U> {
    separator: char,
    param_a: Float,
    word_embeddings: &'w W,
    unigram_lm: &'u U,
}

impl<'w, 'u, W, U> Sif<'w, 'u, W, U>
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

    use ndarray::{arr1, CowArray, Ix1};

    struct SimpleWordEmbeddings {}

    impl WordEmbeddings for SimpleWordEmbeddings {
        fn embedding(&self, word: &str) -> Option<CowArray<Float, Ix1>> {
            match word {
                "A" => Some(arr1(&[1., 2., 3.]).into()),
                "BB" => Some(arr1(&[4., 5., 6.]).into()),
                "CCC" => Some(arr1(&[7., 8., 9.]).into()),
                "DDDD" => Some(arr1(&[10., 11., 12.]).into()),
                _ => None,
            }
        }

        fn embedding_size(&self) -> usize {
            3
        }
    }

    struct SimpleUnigramLanguageModel {}

    impl UnigramLanguageModel for SimpleUnigramLanguageModel {
        fn probability(&self, word: &str) -> Float {
            match word {
                "A" => 1.,
                "BB" => 2.,
                "CCC" => 3.,
                "DDDD" => 4.,
                _ => 0.,
            }
        }
    }

    #[test]
    fn test_embeddings() {
        let word_embeddings = SimpleWordEmbeddings {};
        let unigram_lm = SimpleUnigramLanguageModel {};

        let sif = Sif::new(&word_embeddings, &unigram_lm);

        let sent_embeddings = sif.embeddings(["A BB CCC DDDD", "BB CCC", "A B C", "Z", ""]);
        assert_eq!(sent_embeddings.shape(), &[5, 3]);

        let sent_embeddings = sif.embeddings(Vec::<&str>::new());
        assert_eq!(sent_embeddings.shape(), &[0, 3]);

        let sent_embeddings = sif.embeddings(["", ""]);
        assert_eq!(sent_embeddings.shape(), &[2, 3]);
    }
}
