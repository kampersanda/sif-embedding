//!
use hashbrown::HashMap;

use crate::Float;

///
pub struct UnigramLM {
    word2probs: HashMap<String, Float>,
}

impl UnigramLM {
    /// Creates an instance from word embeddings and weights.
    ///
    /// `word_weights` is used to estimate unigram probabilities of words.
    /// It should be pairs of a word and its frequency (or probability) obtained from a curpus.
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

    /// Returns the unigram probability for the input word.
    pub fn probability<W>(&self, word: W) -> Float
    where
        W: AsRef<str>,
    {
        self.word2probs.get(word.as_ref()).cloned().unwrap_or(0.)
    }
}
