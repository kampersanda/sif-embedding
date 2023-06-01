//! Unigram language models.
use std::collections::BTreeMap;

use crawdad::Trie;

use crate::Float;

/// Unigram language model.
///
/// # Examples
///
/// ```
/// use approx::assert_relative_eq;
/// use sif_embedding::UnigramLM;
///
/// let word_weights = [("las", 10.), ("vegas", 30.)];
/// let unigram_lm = UnigramLM::new(word_weights);
///
/// assert_relative_eq!(unigram_lm.probability("las"), 0.25);
/// assert_relative_eq!(unigram_lm.probability("vegas"), 0.75);
/// assert_relative_eq!(unigram_lm.probability("Las"), 0.00);
/// ```
pub struct UnigramLM {
    trie: Option<Trie>,
}

impl UnigramLM {
    /// Creates the language model.
    ///
    /// # Arguments
    ///
    /// - `word_weights`: Pairs of words and their frequencies (or probabilities) from a corpus.
    ///
    /// # Notes
    ///
    /// If the input contains duplicate words, the last occurrence is used.
    pub fn new<I, W>(word_weights: I) -> Self
    where
        I: IntoIterator<Item = (W, Float)>,
        W: AsRef<str>,
    {
        let mut word2probs: BTreeMap<_, _> = word_weights
            .into_iter()
            .map(|(word, weight)| (word.as_ref().to_string(), weight))
            .collect();

        if word2probs.is_empty() {
            return Self { trie: None };
        }

        let sum_weight = word2probs.values().fold(0., |acc, w| acc + w);
        word2probs.values_mut().for_each(|w| *w /= sum_weight);

        let trie =
            Trie::from_records(word2probs.into_iter().map(|(w, p)| (w, p.to_bits()))).unwrap();
        Self { trie: Some(trie) }
    }

    /// Returns the probability for an input word.
    pub fn probability<W>(&self, word: W) -> Float
    where
        W: AsRef<str>,
    {
        if let Some(trie) = self.trie.as_ref() {
            trie.exact_match(word.as_ref().chars())
                .map(Float::from_bits)
                .unwrap_or(0.)
        } else {
            0.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;

    #[test]
    fn test_empty() {
        let word_weights = Vec::<(&str, Float)>::new();
        let unigram_lm = UnigramLM::new(word_weights);

        assert_relative_eq!(unigram_lm.probability("las"), 0.00);
        assert_relative_eq!(unigram_lm.probability("vegas"), 0.00);
        assert_relative_eq!(unigram_lm.probability("Las"), 0.00);
    }

    #[test]
    fn test_duplicate() {
        let word_weights = [("las", 10.), ("vegas", 30.), ("las", 20.)];
        let unigram_lm = UnigramLM::new(word_weights);

        assert_relative_eq!(unigram_lm.probability("las"), 0.40);
        assert_relative_eq!(unigram_lm.probability("vegas"), 0.60);
        assert_relative_eq!(unigram_lm.probability("Las"), 0.00);
    }
}
