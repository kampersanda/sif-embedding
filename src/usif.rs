//! uSIF: Unsupervised Smooth Inverse Frequency + Piecewise Common Component Removal.
use anyhow::{anyhow, Result};
use ndarray::Array1;
use ndarray::Array2;

use crate::util;
use crate::Float;
use crate::SentenceEmbedder;
use crate::WordEmbeddings;
use crate::WordProbabilities;
use crate::DEFAULT_SEPARATOR;

/// Default value of the number of principal components,
/// following the original setting.
pub const DEFAULT_N_COMPONENTS: usize = 5;

const FLOAT_0_5: Float = 0.5;

/// An implementation of *Unsupervised Smooth Inverse Frequency* and *Piecewise Common Component Removal*,
/// simple but pewerful techniques for sentence embeddings described in the paper:
/// Kawin Ethayarajh,
/// [Unsupervised Random Walk Sentence Embeddings: A Strong but Simple Baseline](https://aclanthology.org/W18-3012/),
/// RepL4NLP 2018.
///
/// # Brief description of API
///
/// The algorithm consists of two steps:
///
/// 1. Compute sentence embeddings with the uSIF weighting.
/// 2. Remove the common components from the sentence embeddings.
///
/// The weighting parameter and common components are computed from input sentences.
///
/// Our API is designed to allow reuse of these values once computed
/// because it is not always possible to obtain a sufficient number of sentences as queries to compute.
///
/// [`USif::fit`] computes these values from input sentences and returns a fitted instance of [`USif`].
/// [`USif::embeddings`] computes sentence embeddings with the fitted values.
///
/// If you find these two steps annoying, you can use [`USif::fit_embeddings`].
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
/// use sif_embedding::{USif, SentenceEmbedder};
///
/// // Loads word embeddings from a pretrained model.
/// let word_embeddings_text = "las 0.0 1.0 2.0\nvegas -3.0 -4.0 -5.0\n";
/// let mut reader = BufReader::new(word_embeddings_text.as_bytes());
/// let word_embeddings = Embeddings::read_text(&mut reader)?;
///
/// // Loads word probabilities from a pretrained model.
/// let word_probs = WordFreq::new([("las", 0.4), ("vegas", 0.6)]);
///
/// // Computes sentence embeddings in shape (n, m),
/// // where n is the number of sentences and m is the number of dimensions.
/// let model = USif::new(&word_embeddings, &word_probs);
/// let (sent_embeddings, model) = model.fit_embeddings(&["las vegas", "mega vegas"])?;
/// assert_eq!(sent_embeddings.shape(), &[2, 3]);
///
/// // Once fitted, the parameters can be used to compute sentence embeddings.
/// let sent_embeddings = model.embeddings(["vegas pro"])?;
/// assert_eq!(sent_embeddings.shape(), &[1, 3]);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct USif<'w, 'p, W, P> {
    word_embeddings: &'w W,
    word_probs: &'p P,
    n_components: usize,
    param_a: Option<Float>,
    weights: Option<Array1<Float>>,
    common_components: Option<Array2<Float>>,
    separator: char,
}

impl<'w, 'p, W, P> USif<'w, 'p, W, P>
where
    W: WordEmbeddings,
    P: WordProbabilities,
{
    /// Creates a new instance with default parameters defined by
    /// [`DEFAULT_N_COMPONENTS`].
    ///
    /// # Arguments
    ///
    /// * `word_embeddings` - Word embeddings.
    /// * `word_probs` - Word probabilities.
    pub const fn new(word_embeddings: &'w W, word_probs: &'p P) -> Self {
        Self {
            word_embeddings,
            word_probs,
            n_components: DEFAULT_N_COMPONENTS,
            param_a: None,
            weights: None,
            common_components: None,
            separator: DEFAULT_SEPARATOR,
        }
    }

    /// Creates a new instance with manually specified parameters.
    ///
    /// # Arguments
    ///
    /// * `word_embeddings` - Word embeddings.
    /// * `word_probs` - Word probabilities.
    /// * `n_components` - The number of principal components to remove.
    pub const fn with_parameters(
        word_embeddings: &'w W,
        word_probs: &'p P,
        n_components: usize,
    ) -> Self {
        Self {
            word_embeddings,
            word_probs,
            n_components,
            param_a: None,
            weights: None,
            common_components: None,
            separator: DEFAULT_SEPARATOR,
        }
    }

    /// Sets a separator for sentence segmentation (default: [`DEFAULT_SEPARATOR`]).
    pub const fn separator(mut self, separator: char) -> Self {
        self.separator = separator;
        self
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
    /// The returned value is always a positive number.
    /// (Lines 5--7 in Algorithm 1)
    fn estimate_param_a(&self, sent_len: Float) -> Float {
        debug_assert!(sent_len > 0.);
        let vocab_size = self.word_probs.n_words() as Float;
        let threshold = 1. - (1. - (1. / vocab_size)).powf(sent_len);
        let n_greater = self
            .word_probs
            .entries()
            .filter(|(_, prob)| *prob > threshold)
            .count() as Float;
        let alpha = n_greater / vocab_size;
        let partiion = 0.5 * vocab_size;
        let param_a = (1. - alpha) / alpha.mul_add(partiion, Float::EPSILON); // avoid division by zero.
        param_a.max(Float::EPSILON) // avoid returning zero.
    }

    /// Applies SIF-weighting for sentences.
    /// (Line 8 in Algorithm 1)
    fn weighted_embeddings<I, S>(&self, sentences: I, param_a: Float) -> Array2<Float>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        debug_assert!(param_a > 0.);
        let mut sent_embeddings = vec![];
        let mut n_sentences = 0;
        for sent in sentences {
            let sent_embedding = self.weighted_embedding(sent.as_ref(), param_a);
            sent_embeddings.extend(sent_embedding.iter());
            n_sentences += 1;
        }
        Array2::from_shape_vec((n_sentences, self.embedding_size()), sent_embeddings).unwrap()
    }

    /// Applies SIF-weighting for a sentence.
    /// (Line 8 in Algorithm 1)
    fn weighted_embedding(&self, sent: &str, param_a: Float) -> Array1<Float> {
        debug_assert!(param_a > 0.);

        // 1. Extract word embeddings and weights.
        let mut n_words = 0;
        let mut word_embeddings: Vec<Float> = vec![];
        let mut word_weights: Vec<Float> = vec![];
        for word in sent.split(self.separator) {
            if let Some(word_embedding) = self.word_embeddings.embedding(word) {
                word_embeddings.extend(word_embedding.iter());
                word_weights
                    .push(param_a / FLOAT_0_5.mul_add(param_a, self.word_probs.probability(word)));
                n_words += 1;
            }
        }

        // If no parseable tokens, return a vector of a's
        if n_words == 0 {
            return Array1::zeros(self.embedding_size()) + param_a;
        }

        // 2. Convert to nd-arrays.
        let word_embeddings =
            Array2::from_shape_vec((n_words, self.embedding_size()), word_embeddings).unwrap();
        let word_weights = Array2::from_shape_vec((n_words, 1), word_weights).unwrap();

        // 3. Normalize word embeddings.
        let axis = ndarray_linalg::norm::NormalizeAxis::Column; // equivalent to Axis(0)
        let (word_embeddings, _) = ndarray_linalg::norm::normalize(word_embeddings, axis);

        // 4. Weight word embeddings.
        let word_embeddings = word_embeddings * &word_weights;

        // 5. Average word embeddings.
        word_embeddings.mean_axis(ndarray::Axis(0)).unwrap()
    }

    /// Estimates the principal components of sentence embeddings.
    /// (Lines 11--17 in Algorithm 1)
    ///
    /// NOTE: Principal components can be empty iff sentence embeddings are all zeros.
    fn estimate_principal_components(
        &self,
        sent_embeddings: &Array2<Float>,
    ) -> (Array1<Float>, Array2<Float>) {
        let (singular_values, singular_vectors) =
            util::principal_components(sent_embeddings, self.n_components);
        let singular_weights = singular_values.mapv(|v| v.powi(2));
        let singular_weights = singular_weights.to_owned() / singular_weights.sum();
        (singular_weights, singular_vectors)
    }

    /// Serializes the model.
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bincode::serialize_into(&mut bytes, &self.n_components)?;
        bincode::serialize_into(&mut bytes, &self.param_a)?;
        bincode::serialize_into(&mut bytes, &self.weights)?;
        bincode::serialize_into(&mut bytes, &self.common_components)?;
        bincode::serialize_into(&mut bytes, &self.separator)?;
        Ok(bytes)
    }

    /// Deserializes the model.
    pub fn deserialize(bytes: &[u8], word_embeddings: &'w W, word_probs: &'p P) -> Result<Self> {
        let mut bytes = bytes;
        let n_components = bincode::deserialize_from(&mut bytes)?;
        let param_a = bincode::deserialize_from(&mut bytes)?;
        let weights = bincode::deserialize_from(&mut bytes)?;
        let common_components = bincode::deserialize_from(&mut bytes)?;
        let separator = bincode::deserialize_from(&mut bytes)?;
        Ok(Self {
            word_embeddings,
            word_probs,
            n_components,
            param_a,
            weights,
            common_components,
            separator,
        })
    }
}

impl<'w, 'p, W, P> SentenceEmbedder for USif<'w, 'p, W, P>
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
    /// # Errors
    ///
    /// Returns an error if `sentences` is empty.
    fn fit<S>(mut self, sentences: &[S]) -> Result<Self>
    where
        S: AsRef<str>,
    {
        if sentences.is_empty() {
            return Err(anyhow!("Input sentences must not be empty."));
        }
        // SIF-weighting.
        let sent_len = self.average_sentence_length(sentences);
        if sent_len == 0. {
            return Err(anyhow!("Input sentences must not be empty."));
        }
        let param_a = self.estimate_param_a(sent_len);
        let sent_embeddings = self.weighted_embeddings(sentences, param_a);
        self.param_a = Some(param_a);
        // Common component removal.
        if self.n_components != 0 {
            let (weights, common_components) = self.estimate_principal_components(&sent_embeddings);
            self.weights = Some(weights);
            self.common_components = Some(common_components);
        }
        Ok(self)
    }

    /// Computes embeddings for input sentences using the fitted model.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not fitted.
    fn embeddings<I, S>(&self, sentences: I) -> Result<Array2<Float>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        if self.param_a.is_none() {
            return Err(anyhow!("The model is not fitted."));
        }
        // SIF-weighting.
        let sent_embeddings = self.weighted_embeddings(sentences, self.param_a.unwrap());
        if sent_embeddings.is_empty() {
            return Ok(sent_embeddings);
        }
        if self.n_components == 0 {
            return Ok(sent_embeddings);
        }
        // Common component removal.
        let weights = self.weights.as_ref().unwrap();
        let common_components = self.common_components.as_ref().unwrap();
        let sent_embeddings =
            util::remove_principal_components(&sent_embeddings, common_components, Some(weights));
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

        let sif = USif::new(&word_embeddings, &word_probs)
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

        let sif = USif::new(&word_embeddings, &word_probs);

        let sif = sif.fit(sentences_1).unwrap();
        let embeddings_1 = sif.embeddings(sentences_1).unwrap();

        let sif = sif.separator(',');
        let embeddings_2 = sif.embeddings(sentences_2).unwrap();

        assert_relative_eq!(embeddings_1, embeddings_2);
    }

    #[test]
    fn test_no_fitted() {
        let word_embeddings = SimpleWordEmbeddings {};
        let word_probs = SimpleWordProbabilities {};

        let sentences = &["A BB CCC DDDD", "BB CCC", "A B C", "Z", ""];

        let sif = USif::new(&word_embeddings, &word_probs);
        let embeddings = sif.embeddings(sentences);

        assert!(embeddings.is_err());
    }

    #[test]
    fn test_empty_fit() {
        let word_embeddings = SimpleWordEmbeddings {};
        let word_probs = SimpleWordProbabilities {};

        let sif = USif::new(&word_embeddings, &word_probs);
        let sif = sif.fit(&Vec::<&str>::new());

        assert!(sif.is_err());
    }

    #[test]
    fn test_io() {
        let word_embeddings = SimpleWordEmbeddings {};
        let word_probs = SimpleWordProbabilities {};

        let sentences = ["A BB CCC DDDD", "BB CCC", "A B C", "Z", ""];
        let model_a = USif::new(&word_embeddings, &word_probs)
            .fit(&sentences)
            .unwrap();
        let bytes = model_a.serialize().unwrap();
        let model_b = USif::deserialize(&bytes, &word_embeddings, &word_probs).unwrap();

        let embeddings_a = model_a.embeddings(sentences).unwrap();
        let embeddings_b = model_b.embeddings(sentences).unwrap();

        assert_relative_eq!(embeddings_a, embeddings_b);
    }
}
