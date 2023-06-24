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
    weights: Option<Array1<Float>>,
    common_components: Option<Array2<Float>>,
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
            weights: None,
            common_components: None,
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
        self.param_a.is_some() || self.weights.is_some() || self.common_components.is_some()
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
        let param_a = self.estimate_param_a(sent_len);
        let sent_embeddings = self.weighted_embeddings(sentences, param_a);
        // Common component removal.
        let (weights, common_components) = self.estimate_principal_components(&sent_embeddings);
        // Set the fitted parameters.
        self.param_a = Some(param_a);
        self.weights = Some(weights);
        self.common_components = Some(common_components);
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
        // Get the fitted parameters.
        let param_a = self.param_a.unwrap();
        let weights = self.weights.as_ref().unwrap();
        let common_components = self.common_components.as_ref().unwrap();
        // Embedding.
        let sent_embeddings = self.weighted_embeddings(sentences, param_a);
        let sent_embeddings =
            util::remove_principal_components(&sent_embeddings, common_components, Some(weights));
        Ok(sent_embeddings)
    }

    ///
    pub fn fit_embeddings<S>(&mut self, sentences: &[S]) -> Result<Array2<Float>>
    where
        S: AsRef<str>,
    {
        if sentences.is_empty() {
            return Err(anyhow!("no sentences"));
        }
        // SIF-weighting.
        let sent_len = self.average_sentence_length(sentences);
        let param_a = self.estimate_param_a(sent_len);
        let sent_embeddings = self.weighted_embeddings(sentences, param_a);
        // Common component removal.
        let (weights, common_components) = self.estimate_principal_components(&sent_embeddings);
        let sent_embeddings =
            util::remove_principal_components(&sent_embeddings, &common_components, Some(&weights));
        // Set the fitted parameters.
        self.param_a = Some(param_a);
        self.weights = Some(weights);
        self.common_components = Some(common_components);
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
    fn weighted_embeddings<I, S>(&self, sentences: I, param_a: Float) -> Array2<Float>
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

    use approx::assert_relative_eq;
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

        let sif = USif::new(&word_embeddings, &unigram_lm)
            .fit(&["A BB CCC DDDD", "BB CCC", "A B C", "Z", ""])
            .unwrap();

        let sent_embeddings = sif
            .embeddings(["A BB CCC DDDD", "BB CCC", "A B C", "Z", ""])
            .unwrap();
        assert_eq!(sent_embeddings.shape(), &[5, 3]);

        let sent_embeddings = sif.embeddings(Vec::<&str>::new()).unwrap();
        assert_eq!(sent_embeddings.shape(), &[0, 3]);

        let sent_embeddings = sif.embeddings(["", ""]).unwrap();
        assert_eq!(sent_embeddings.shape(), &[2, 3]);
    }

    #[test]
    fn test_equality() {
        let word_embeddings = SimpleWordEmbeddings::new();
        let unigram_lm = SimpleUnigramLanguageModel::new();

        let sentences = &["A BB CCC DDDD", "BB CCC", "A B C", "Z", ""];

        let mut sif = USif::new(&word_embeddings, &unigram_lm);
        let embeddings_1 = sif.fit_embeddings(sentences).unwrap();
        let embeddings_2 = sif.embeddings(sentences).unwrap();
        assert_relative_eq!(embeddings_1, embeddings_2);

        let sif = USif::new(&word_embeddings, &unigram_lm);
        let sif = sif.fit(sentences).unwrap();
        let embeddings_3 = sif.embeddings(sentences).unwrap();
        assert_relative_eq!(embeddings_1, embeddings_3);
    }

    #[test]
    fn test_separator() {
        let word_embeddings = SimpleWordEmbeddings::new();
        let unigram_lm = SimpleUnigramLanguageModel::new();

        let sentences_1 = &["A BB CCC DDDD", "BB CCC", "A B C", "Z", ""];
        let sentences_2 = &["A,BB,CCC,DDDD", "BB,CCC", "A,B,C", "Z", ""];

        let sif = USif::new(&word_embeddings, &unigram_lm);
        let sif = sif.fit(sentences_1).unwrap();
        let embeddings_1 = sif.embeddings(sentences_1).unwrap();

        let sif = sif.separator(',');
        let embeddings_2 = sif.embeddings(sentences_2).unwrap();

        assert_relative_eq!(embeddings_1, embeddings_2);
    }

    #[test]
    fn test_param_a() {
        let word_embeddings = SimpleWordEmbeddings::new();
        let unigram_lm = SimpleUnigramLanguageModel::new();

        let sif = USif::new(&word_embeddings, &unigram_lm);
        let sif = sif.fit(&[""]).unwrap();

        let e = sif.param_a(1.);
        assert!(e.is_err());
    }

    #[test]
    fn test_is_fitted() {
        let word_embeddings = SimpleWordEmbeddings::new();
        let unigram_lm = SimpleUnigramLanguageModel::new();

        let sif = USif::new(&word_embeddings, &unigram_lm);
        let sif = sif.fit(&[""]).unwrap();

        assert!(sif.is_fitted());
    }

    #[test]
    fn test_no_fitted() {
        let word_embeddings = SimpleWordEmbeddings::new();
        let unigram_lm = SimpleUnigramLanguageModel::new();

        let sif = USif::new(&word_embeddings, &unigram_lm);
        let embeddings = sif.embeddings([""]);

        assert!(embeddings.is_err());
    }

    #[test]
    fn test_empty_fit() {
        let word_embeddings = SimpleWordEmbeddings::new();
        let unigram_lm = SimpleUnigramLanguageModel::new();

        let sif = USif::new(&word_embeddings, &unigram_lm);
        let sif = sif.fit(&Vec::<&str>::new());

        assert!(sif.is_err());
    }

    #[test]
    fn test_empty_fit_embeddings() {
        let word_embeddings = SimpleWordEmbeddings::new();
        let unigram_lm = SimpleUnigramLanguageModel::new();

        let mut sif = USif::new(&word_embeddings, &unigram_lm);
        let embeddings = sif.fit_embeddings(&Vec::<&str>::new());

        assert!(embeddings.is_err());
    }
}
