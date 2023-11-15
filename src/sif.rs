//! SIF: Smooth Inverse Frequency + Common Component Removal.
use anyhow::{anyhow, Result};
use ndarray::Array1;
use ndarray::Array2;

use crate::util;
use crate::Float;
use crate::SentenceEmbedder;
use crate::WordEmbeddings;
use crate::WordProbabilities;
use crate::DEFAULT_N_SAMPLES_TO_FIT;
use crate::DEFAULT_SEPARATOR;

/// Default value of the SIF-weighting parameter `a`,
/// following the original setting.
pub const DEFAULT_PARAM_A: Float = 1e-3;

/// Default value of the number of principal components to remove,
/// following the original setting.
pub const DEFAULT_N_COMPONENTS: usize = 1;

/// An implementation of *Smooth Inverse Frequency* and *Common Component Removal*,
/// simple but pewerful techniques for sentence embeddings described in the paper:
/// Sanjeev Arora, Yingyu Liang, and Tengyu Ma,
/// [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/forum?id=SyK00v5xx),
/// ICLR 2017.
///
/// # Brief description of API
///
/// The algorithm consists of two steps:
///
/// 1. Compute sentence embeddings with the SIF weighting.
/// 2. Remove the common components from the sentence embeddings.
///
/// The common components are computed from input sentences.
///
/// Our API is designed to allow reuse of common components once computed
/// because it is not always possible to obtain a sufficient number of sentences as queries to compute.
///
/// [`Sif::fit`] computes the common components from input sentences and returns a fitted instance of [`Sif`].
/// [`Sif::embeddings`] computes sentence embeddings with the fitted components.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use std::io::BufReader;
///
/// use finalfusion::compat::text::ReadText;
/// use finalfusion::embeddings::Embeddings;
/// use wordfreq::WordFreq;
///
/// use sif_embedding::{Sif, SentenceEmbedder};
///
/// // Loads word embeddings from a pretrained model.
/// let word_embeddings_text = "las 0.0 1.0 2.0\nvegas -3.0 -4.0 -5.0\n";
/// let mut reader = BufReader::new(word_embeddings_text.as_bytes());
/// let word_embeddings = Embeddings::read_text(&mut reader)?;
///
/// // Loads word probabilities from a pretrained model.
/// let word_probs = WordFreq::new([("las", 0.4), ("vegas", 0.6)]);
///
/// // Prepares input sentences.
/// let sentences = ["las vegas", "mega vegas"];
///
/// // Fits the model with input sentences.
/// let model = Sif::new(&word_embeddings, &word_probs);
/// let model = model.fit(&sentences)?;
///
/// // Computes sentence embeddings in shape (n, m),
/// // where n is the number of sentences and m is the number of dimensions.
/// let sent_embeddings = model.embeddings(sentences)?;
/// assert_eq!(sent_embeddings.shape(), &[2, 3]);
/// # Ok(())
/// # }
/// ```
///
/// ## Only SIF weighting
///
/// If you want to apply only the SIF weighting to avoid the computation of common components,
/// use [`Sif::with_parameters`] and set `n_components` to `0`.
/// In this case, you can skip [`Sif::fit`] and directly perform [`Sif::embeddings`]
/// because there is no parameter to fit
/// (although the quality of the embeddings may be worse).
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use std::io::BufReader;
///
/// use finalfusion::compat::text::ReadText;
/// use finalfusion::embeddings::Embeddings;
/// use wordfreq::WordFreq;
///
/// use sif_embedding::{Sif, SentenceEmbedder};
///
/// // Loads word embeddings from a pretrained model.
/// let word_embeddings_text = "las 0.0 1.0 2.0\nvegas -3.0 -4.0 -5.0\n";
/// let mut reader = BufReader::new(word_embeddings_text.as_bytes());
/// let word_embeddings = Embeddings::read_text(&mut reader)?;
///
/// // Loads word probabilities from a pretrained model.
/// let word_probs = WordFreq::new([("las", 0.4), ("vegas", 0.6)]);
///
/// // When setting `n_components` to `0`, no common components are removed, and
/// // the sentence embeddings can be computed without `fit`.
/// let model = Sif::with_parameters(&word_embeddings, &word_probs, 1e-3, 0)?;
/// let sent_embeddings = model.embeddings(["las vegas", "mega vegas"])?;
/// assert_eq!(sent_embeddings.shape(), &[2, 3]);
/// # Ok(())
/// # }
/// ```
///
/// ## Serialization of fitted parameters
///
/// If you want to serialize and deserialize the fitted parameters,
/// use [`Sif::serialize`] and [`Sif::deserialize`].
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use std::io::BufReader;
///
/// use approx::assert_relative_eq;
/// use finalfusion::compat::text::ReadText;
/// use finalfusion::embeddings::Embeddings;
/// use wordfreq::WordFreq;
///
/// use sif_embedding::{Sif, SentenceEmbedder};
///
/// // Loads word embeddings from a pretrained model.
/// let word_embeddings_text = "las 0.0 1.0 2.0\nvegas -3.0 -4.0 -5.0\n";
/// let mut reader = BufReader::new(word_embeddings_text.as_bytes());
/// let word_embeddings = Embeddings::read_text(&mut reader)?;
///
/// // Loads word probabilities from a pretrained model.
/// let word_probs = WordFreq::new([("las", 0.4), ("vegas", 0.6)]);
///
/// // Prepares input sentences.
/// let sentences = ["las vegas", "mega vegas"];
///
/// // Fits the model and computes sentence embeddings.
/// let model = Sif::new(&word_embeddings, &word_probs);
/// let model = model.fit(&sentences)?;
/// let sent_embeddings = model.embeddings(&sentences)?;
///
/// // Serializes and deserializes the fitted parameters.
/// let bytes = model.serialize()?;
/// let other = Sif::deserialize(&bytes, &word_embeddings, &word_probs)?;
/// let other_embeddings = other.embeddings(&sentences)?;
/// assert_relative_eq!(sent_embeddings, other_embeddings);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Sif<'w, 'p, W, P> {
    word_embeddings: &'w W,
    word_probs: &'p P,
    param_a: Float,
    n_components: usize,
    common_components: Option<Array2<Float>>,
    separator: char,
    n_samples_to_fit: usize,
}

impl<'w, 'p, W, P> Sif<'w, 'p, W, P>
where
    W: WordEmbeddings,
    P: WordProbabilities,
{
    /// Creates a new instance with default parameters defined by
    /// [`DEFAULT_PARAM_A`] and [`DEFAULT_N_COMPONENTS`].
    ///
    /// # Arguments
    ///
    /// * `word_embeddings` - Word embeddings.
    /// * `word_probs` - Word probabilities.
    pub const fn new(word_embeddings: &'w W, word_probs: &'p P) -> Self {
        Self {
            word_embeddings,
            word_probs,
            param_a: DEFAULT_PARAM_A,
            n_components: DEFAULT_N_COMPONENTS,
            common_components: None,
            separator: DEFAULT_SEPARATOR,
            n_samples_to_fit: DEFAULT_N_SAMPLES_TO_FIT,
        }
    }

    /// Creates a new instance with manually specified parameters.
    ///
    /// # Arguments
    ///
    /// * `word_embeddings` - Word embeddings.
    /// * `word_probs` - Word probabilities.
    /// * `param_a` - A parameter `a` for SIF-weighting that should be positive.
    /// * `n_components` - The number of principal components to remove.
    ///
    /// When setting `n_components` to `0`, no principal components are removed.
    ///
    /// # Errors
    ///
    /// Returns an error if `param_a` is not positive.
    pub fn with_parameters(
        word_embeddings: &'w W,
        word_probs: &'p P,
        param_a: Float,
        n_components: usize,
    ) -> Result<Self> {
        if param_a <= 0. {
            return Err(anyhow!("param_a must be positive."));
        }
        Ok(Self {
            word_embeddings,
            word_probs,
            param_a,
            n_components,
            common_components: None,
            separator: DEFAULT_SEPARATOR,
            n_samples_to_fit: DEFAULT_N_SAMPLES_TO_FIT,
        })
    }

    /// Sets a separator for sentence segmentation (default: [`DEFAULT_SEPARATOR`]).
    pub const fn separator(mut self, separator: char) -> Self {
        self.separator = separator;
        self
    }

    /// Sets the number of samples to fit the model (default: [`DEFAULT_N_SAMPLES_TO_FIT`]).
    ///
    /// # Errors
    ///
    /// Returns an error if `n_samples_to_fit` is 0.
    pub fn n_samples_to_fit(mut self, n_samples_to_fit: usize) -> Result<Self> {
        if n_samples_to_fit == 0 {
            return Err(anyhow!("n_samples_to_fit must not be 0."));
        }
        self.n_samples_to_fit = n_samples_to_fit;
        Ok(self)
    }

    /// Applies SIF-weighting.
    /// (Lines 1--3 in Algorithm 1)
    ///
    /// # Complexities
    ///
    /// * Time complexity: `O(avg_num_words * embedding_size * num_sentences)`
    /// * Space complexity: `O(embedding_size * num_sentences)`
    fn weighted_embeddings<I, S>(&self, sentences: I) -> Array2<Float>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut sent_embeddings = vec![];
        let mut n_sentences = 0;
        // O(num_words * embedding_size * num_sentences)
        for sent in sentences {
            let sent = sent.as_ref();
            let mut n_words = 0;
            let mut sent_embedding = Array1::zeros(self.embedding_size());
            // O(avg_num_words * embedding_size)
            for word in sent.split(self.separator) {
                if let Some(word_embedding) = self.word_embeddings.embedding(word) {
                    let weight = self.param_a / (self.param_a + self.word_probs.probability(word));
                    sent_embedding += &(word_embedding.to_owned() * weight);
                    n_words += 1;
                }
            }
            if n_words != 0 {
                sent_embedding /= n_words as Float;
            } else {
                // If no parseable tokens, return a vector of a's
                sent_embedding += self.param_a;
            }
            sent_embeddings.extend(sent_embedding.iter());
            n_sentences += 1;
        }
        Array2::from_shape_vec((n_sentences, self.embedding_size()), sent_embeddings).unwrap()
    }

    /// Serializes the model.
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bincode::serialize_into(&mut bytes, &self.param_a)?;
        bincode::serialize_into(&mut bytes, &self.n_components)?;
        bincode::serialize_into(&mut bytes, &self.common_components)?;
        bincode::serialize_into(&mut bytes, &self.separator)?;
        bincode::serialize_into(&mut bytes, &self.n_samples_to_fit)?;
        Ok(bytes)
    }

    /// Deserializes the model.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Byte sequence exported by [`Self::serialize`].
    /// * `word_embeddings` - Word embeddings.
    /// * `word_probs` - Word probabilities.
    ///
    /// `word_embeddings` and `word_probs` must be the same as those used in serialization.
    pub fn deserialize(bytes: &[u8], word_embeddings: &'w W, word_probs: &'p P) -> Result<Self> {
        let mut bytes = bytes;
        let param_a = bincode::deserialize_from(&mut bytes)?;
        let n_components = bincode::deserialize_from(&mut bytes)?;
        let common_components = bincode::deserialize_from(&mut bytes)?;
        let separator = bincode::deserialize_from(&mut bytes)?;
        let n_samples_to_fit = bincode::deserialize_from(&mut bytes)?;
        Ok(Self {
            word_embeddings,
            word_probs,
            param_a,
            n_components,
            common_components,
            separator,
            n_samples_to_fit,
        })
    }
}

impl<'w, 'p, W, P> SentenceEmbedder for Sif<'w, 'p, W, P>
where
    W: WordEmbeddings,
    P: WordProbabilities,
{
    /// Returns the number of dimensions for sentence embeddings,
    /// which is the same as the number of dimensions for word embeddings.
    fn embedding_size(&self) -> usize {
        self.word_embeddings.embedding_size()
    }

    /// Fits the model with input sentences.
    ///
    /// Sentences to fit are randomly sampled from `sentences` with [`Self::n_samples_to_fit`].
    ///
    /// If `n_components` is 0, does nothing and returns `self`.
    ///
    /// # Errors
    ///
    /// Returns an error if `sentences` is empty.
    ///
    /// # Complexities
    ///
    /// * Time complexity: `O(L*D*S + max(D,S)^3)`
    /// * Space complexity: `O(D*S + max(D,S)^2)`
    ///
    /// where
    ///
    /// * `L` is the average number of words in a sentence.
    /// * `D` is the number of dimensions for word embeddings (`embedding_size`).
    /// * `S` is the number of sentences used to fit (`n_samples_to_fit`).
    fn fit<S>(mut self, sentences: &[S]) -> Result<Self>
    where
        S: AsRef<str>,
    {
        if sentences.is_empty() {
            return Err(anyhow!("Input sentences must not be empty."));
        }
        if self.n_components == 0 {
            eprintln!("Warning: Nothing to fit since n_components is 0.");
            return Ok(self);
        }

        // Time: O(n_samples_to_fit)
        let sentences = util::sample_sentences(sentences, self.n_samples_to_fit);

        // SIF-weighting.
        //
        // Time: O(avg_num_words * embedding_size * n_samples_to_fit)
        // Space: O(embedding_size * n_samples_to_fit)
        let sent_embeddings = self.weighted_embeddings(sentences);

        // Common component removal.
        //
        // Time: O(max(embedding_size, n_samples_to_fit)^3)
        // Space: O(max(embedding_size, n_samples_to_fit)^2)
        let (_, common_components) =
            util::principal_components(&sent_embeddings, self.n_components);
        self.common_components = Some(common_components);

        Ok(self)
    }

    /// Computes embeddings for input sentences using the fitted model.
    ///
    /// If `n_components` is 0, the fitting is not required.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not fitted.
    ///
    /// # Complexities
    ///
    /// * Time complexity: `O(L*D*N + C*D*N)`
    /// * Space complexity: `O(D*N)`
    ///
    /// where
    ///
    /// * `L` is the average number of words in a sentence.
    /// * `D` is the number of dimensions for word embeddings (`embedding_size`).
    /// * `N` is the number of sentences (`sentences.len()`).
    /// * `C` is the number of components to remove (`n_components`).
    fn embeddings<I, S>(&self, sentences: I) -> Result<Array2<Float>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        if self.n_components != 0 && self.common_components.is_none() {
            return Err(anyhow!("The model is not fitted."));
        }

        // SIF-weighting.
        //
        // Time: O(avg_num_words * embedding_size * n_sentences)
        // Space: O(embedding_size * n_sentences)
        let sent_embeddings = self.weighted_embeddings(sentences);
        if sent_embeddings.is_empty() {
            return Ok(sent_embeddings);
        }
        if self.n_components == 0 {
            return Ok(sent_embeddings);
        }

        // Common component removal.
        //
        // Time: O(embedding_size * n_sentences * n_components)
        // Space: O(embedding_size * n_sentences)
        let common_components = self.common_components.as_ref().unwrap();
        let sent_embeddings =
            util::remove_principal_components(&sent_embeddings, common_components, None);
        Ok(sent_embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
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

    struct SimpleWordProbabilities {}

    impl WordProbabilities for SimpleWordProbabilities {
        fn probability(&self, word: &str) -> Float {
            match word {
                "A" => 0.6,
                "BB" => 0.2,
                "CCC" => 0.1,
                "DDDD" => 0.1,
                _ => 0.,
            }
        }

        fn n_words(&self) -> usize {
            4
        }

        fn entries(&self) -> Box<dyn Iterator<Item = (String, Float)> + '_> {
            Box::new(
                [("A", 0.6), ("BB", 0.2), ("CCC", 0.1), ("DDDD", 0.1)]
                    .iter()
                    .map(|&(word, prob)| (word.to_string(), prob)),
            )
        }
    }

    #[test]
    fn test_basic() {
        let word_embeddings = SimpleWordEmbeddings {};
        let word_probs = SimpleWordProbabilities {};

        let sif = Sif::new(&word_embeddings, &word_probs)
            .fit(&["A BB CCC DDDD", "BB CCC", "A B C", "Z", ""])
            .unwrap();

        let sent_embeddings = sif
            .embeddings(["A BB CCC DDDD", "BB CCC", "A B C", "Z", ""])
            .unwrap();
        assert_ne!(sent_embeddings, Array2::zeros((5, 3)));

        let sent_embeddings = sif.embeddings(Vec::<&str>::new()).unwrap();
        assert_eq!(sent_embeddings.shape(), &[0, 3]);

        let sent_embeddings = sif.embeddings([""]).unwrap();
        assert_ne!(sent_embeddings, Array2::zeros((1, 3)));
    }

    #[test]
    fn test_separator() {
        let word_embeddings = SimpleWordEmbeddings {};
        let word_probs = SimpleWordProbabilities {};

        let sentences_1 = &["A BB CCC DDDD", "BB CCC", "A B C", "Z", ""];
        let sentences_2 = &["A,BB,CCC,DDDD", "BB,CCC", "A,B,C", "Z", ""];

        let sif = Sif::new(&word_embeddings, &word_probs);

        let sif = sif.fit(sentences_1).unwrap();
        let embeddings_1 = sif.embeddings(sentences_1).unwrap();

        let sif = sif.separator(',');
        let embeddings_2 = sif.embeddings(sentences_2).unwrap();

        assert_relative_eq!(embeddings_1, embeddings_2);
    }

    #[test]
    fn test_invalid_param_a() {
        let word_embeddings = SimpleWordEmbeddings {};
        let word_probs = SimpleWordProbabilities {};

        let sif = Sif::with_parameters(&word_embeddings, &word_probs, 0., DEFAULT_N_COMPONENTS);
        assert!(sif.is_err());
    }

    #[test]
    fn test_no_fitted() {
        let word_embeddings = SimpleWordEmbeddings {};
        let word_probs = SimpleWordProbabilities {};

        let sentences = &["A BB CCC DDDD", "BB CCC", "A B C", "Z", ""];

        let sif = Sif::new(&word_embeddings, &word_probs);
        let embeddings = sif.embeddings(sentences);
        assert!(embeddings.is_err());
    }

    #[test]
    fn test_empty_fit() {
        let word_embeddings = SimpleWordEmbeddings {};
        let word_probs = SimpleWordProbabilities {};

        let sif = Sif::new(&word_embeddings, &word_probs);
        let sif = sif.fit(&Vec::<&str>::new());
        assert!(sif.is_err());
    }

    #[test]
    fn test_io() {
        let word_embeddings = SimpleWordEmbeddings {};
        let word_probs = SimpleWordProbabilities {};

        let sentences = ["A BB CCC DDDD", "BB CCC", "A B C", "Z", ""];
        let model_a = Sif::new(&word_embeddings, &word_probs)
            .fit(&sentences)
            .unwrap();
        let bytes = model_a.serialize().unwrap();
        let model_b = Sif::deserialize(&bytes, &word_embeddings, &word_probs).unwrap();

        let embeddings_a = model_a.embeddings(sentences).unwrap();
        let embeddings_b = model_b.embeddings(sentences).unwrap();

        assert_relative_eq!(embeddings_a, embeddings_b);
    }
}
