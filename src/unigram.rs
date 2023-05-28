//! Unigram language models.
use hashbrown::HashMap;

use crate::Float;

/// Unigram language model.
///
/// # Examples
///
/// ```
/// use approx::relative_eq;
/// use sif_embedding::UnigramLM;
///
/// let word_weights = [("las", 10.), ("vegas", 30.)];
/// let unigram_lm = UnigramLM::new(word_weights);
///
/// relative_eq!(unigram_lm.probability("las"), 0.25);
/// relative_eq!(unigram_lm.probability("vegas"), 0.75);
/// relative_eq!(unigram_lm.probability("Las"), 0.00);
/// ```
#[derive(Debug, Clone)]
pub struct UnigramLM {
    word2probs: HashMap<String, Float>,
}

impl UnigramLM {
    /// Creates the language model.
    ///
    /// # Arguments
    ///
    /// - `word_weights`: Pairs of words and their frequencies (or probabilities) from a corpus.
    pub fn new<I, W>(word_weights: I) -> Self
    where
        I: IntoIterator<Item = (W, Float)>,
        W: AsRef<str>,
    {
        let mut word2probs: HashMap<_, _> = word_weights
            .into_iter()
            .map(|(word, weight)| (word.as_ref().to_string(), weight))
            .collect();
        let sum_weight = word2probs.values().fold(0., |acc, w| acc + w);
        word2probs.values_mut().for_each(|w| *w /= sum_weight);
        Self { word2probs }
    }

    /// Returns the probability for an input word.
    pub fn probability<W>(&self, word: W) -> Float
    where
        W: AsRef<str>,
    {
        self.word2probs.get(word.as_ref()).cloned().unwrap_or(0.)
    }
}
