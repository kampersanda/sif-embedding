//! Unsupervised Smooth Inverse Frequency (uSIF).
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
    n_components: usize,
    param_a: Option<Float>,
    singular_weights: Option<Array1<Float>>,
    singular_vectors: Option<Array2<Float>>,
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
            n_components: N_COMPONENTS,
            param_a: None,
            singular_weights: None,
            singular_vectors: None,
        }
    }

    /// Returns the number of dimensions for sentence embeddings,
    /// which is equivalent to that of the input word embeddings.
    pub fn embedding_size(&self) -> usize {
        self.word_embeddings.embedding_size()
    }

    /// Sets a separator for sentence segmentation (default: ASCII whitespace).
    pub const fn separator(mut self, separator: char) -> Self {
        self.separator = separator;
        self
    }

    ///
    pub fn is_fitted(&self) -> bool {
        self.param_a.is_some() || self.singular_weights.is_some() || self.singular_vectors.is_some()
    }

    ///
    pub fn fit<S>(mut self, sentences: &[S]) -> Result<Self>
    where
        S: AsRef<str>,
    {
        if sentences.is_empty() {
            return Err(anyhow!("no sentences"));
        }
        // SIF-weighting.
        let sent_len = self.average_sentence_length(sentences);
        self.param_a = Some(self.estimate_param_a(sent_len));
        let sent_embeddings = self.weighted_embeddings(sentences);
        // Common component removal.
        let (singular_weights, singular_vectors) =
            self.estimate_principal_components(&sent_embeddings);
        self.singular_weights = Some(singular_weights);
        self.singular_vectors = Some(singular_vectors);
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
        if !self.is_fitted() {
            return Err(anyhow!("not fitted"));
        }
        let sent_embeddings = self.weighted_embeddings(sentences);
        let singular_weights = self.singular_weights.as_ref().unwrap();
        let singular_vectors = self.singular_vectors.as_ref().unwrap();
        let sent_embeddings = util::remove_principal_components(
            &sent_embeddings,
            singular_vectors,
            Some(singular_weights),
        );
        Ok(sent_embeddings)
    }

    /// Computes the average length of sentences.
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

    /// Estimates the parameter `a` for the weight function.
    /// (Lines 5--7 in Algorithm 1)
    fn estimate_param_a(&self, sent_len: Float) -> Float {
        let n_words = self.word_embeddings.n_words() as Float;
        let threshold = 1. - (1. - (1. / n_words)).powf(sent_len);
        let n_greater = self
            .word_embeddings
            .words()
            .filter(|word| self.unigram_lm.probability(word) > threshold)
            .count() as Float;
        let alpha = n_greater / n_words;
        let partiion = n_words / 2.;
        (1. - alpha) / (alpha * partiion)
    }

    /// Applies SIF-weighting.
    /// (Line 8 in Algorithm 1)
    fn weighted_embeddings<I, S>(&self, sentences: I) -> Array2<Float>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
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
        Array2::from_shape_vec((n_sentences, self.embedding_size()), sent_embeddings).unwrap()
    }

    /// Estimates the principal components of sentence embeddings.
    /// (Lines 11--17 in Algorithm 1)
    fn estimate_principal_components(
        &self,
        sent_embeddings: &Array2<Float>,
    ) -> (Array1<Float>, Array2<Float>) {
        let (singular_values, singular_vectors) =
            util::principal_components(&sent_embeddings, self.n_components);
        let singular_weights = singular_values.mapv(|v| v.powf(2.0));
        let singular_weights = singular_weights.to_owned() / singular_weights.sum();
        (singular_weights, singular_vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::{arr1, CowArray, Ix1};

    struct SimpleWordEmbeddings {
        words: Vec<String>,
    }

    impl SimpleWordEmbeddings {
        fn new() -> Self {
            Self {
                words: vec![
                    "A".to_owned(),
                    "BB".to_owned(),
                    "CCC".to_owned(),
                    "DDDD".to_owned(),
                ],
            }
        }
    }

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

        fn n_words(&self) -> usize {
            4
        }

        fn words(&self) -> Box<dyn Iterator<Item = String> + '_> {
            Box::new(self.words.iter().cloned())
        }
    }

    struct SimpleUnigramLanguageModel {}

    impl SimpleUnigramLanguageModel {
        fn new() -> Self {
            Self {}
        }
    }

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
        let word_embeddings = SimpleWordEmbeddings::new();
        let unigram_lm = SimpleUnigramLanguageModel::new();

        let usif = USif::new(&word_embeddings, &unigram_lm)
            .fit(&["A BB CCC DDDD", "BB CCC", "A B C", "Z", ""])
            .unwrap();

        let sent_embeddings = usif
            .embeddings(["A BB CCC DDDD", "BB CCC", "A B C", "Z", ""])
            .unwrap();
        assert_eq!(sent_embeddings.shape(), &[5, 3]);

        let sent_embeddings = usif.embeddings(Vec::<&str>::new()).unwrap();
        assert_eq!(sent_embeddings.shape(), &[0, 3]);

        let sent_embeddings = usif.embeddings(["", ""]).unwrap();
        assert_eq!(sent_embeddings.shape(), &[2, 3]);
    }
}
