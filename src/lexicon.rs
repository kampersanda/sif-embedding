//! Handlers for vocabulary.
use hashbrown::HashMap;
use ndarray::{CowArray, Ix1};

use crate::{Float, WordEmbeddings};

/// Lexicon that handles embeddings and unigram probabilities of words.
#[derive(Debug, Clone)]
pub struct Lexicon {
    embeddings: WordEmbeddings,
    word2probs: HashMap<String, Float>,
}

impl Lexicon {
    /// Creates an instance from word embeddings and weights.
    ///
    /// `word_weights` is used to estimate unigram probabilities of words.
    /// It should be pairs of a word and its frequency (or probability) obtained from a curpus.
    pub fn new<I, W>(embeddings: WordEmbeddings, word_weights: I) -> Self
    where
        I: IntoIterator<Item = (W, Float)>,
        W: AsRef<str>,
    {
        let mut word2weight: HashMap<_, _> = word_weights
            .into_iter()
            .map(|(word, weight)| (word.as_ref().to_string(), weight))
            .collect();

        // To probability
        let sum_weight = word2weight.values().fold(0., |acc, w| acc + w);
        word2weight.values_mut().for_each(|w| *w /= sum_weight);

        Self {
            embeddings,
            word2probs: word2weight,
        }
    }

    /// Returns the embedding for the input word.
    pub fn embedding<W>(&self, word: W) -> Option<CowArray<'_, Float, Ix1>>
    where
        W: AsRef<str>,
    {
        if let Some(embedding) = self.embeddings.lookup(word.as_ref()) {
            return Some(embedding);
        }
        None
    }

    /// Returns the unigram probability for the input word.
    pub fn probability<W>(&self, word: W) -> Float
    where
        W: AsRef<str>,
    {
        self.word2probs.get(word.as_ref()).cloned().unwrap_or(0.)
    }

    /// Returns the number of dimensions for word embeddings.
    pub fn embedding_size(&self) -> usize {
        self.embeddings.embedding_size()
    }
}
