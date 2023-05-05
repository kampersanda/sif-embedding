//! Handlers for vocabulary.
use hashbrown::HashMap;
use ndarray::{CowArray, Ix1};

use crate::{Float, WordEmbeddings};

/// Lexicon that handles embeddings and weights of words.
#[derive(Debug, Clone)]
pub struct Lexicon {
    embeddings: WordEmbeddings,
    word2weight: HashMap<String, Float>,
}

impl Lexicon {
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
            word2weight,
        }
    }

    pub fn embedding<W>(&self, word: W) -> Option<CowArray<'_, Float, Ix1>>
    where
        W: AsRef<str>,
    {
        if let Some(embedding) = self.embeddings.lookup(word.as_ref()) {
            return Some(embedding);
        }
        None
    }

    pub fn probability<W>(&self, word: W) -> Float
    where
        W: AsRef<str>,
    {
        self.word2weight.get(word.as_ref()).cloned().unwrap_or(0.)
    }

    pub fn embedding_size(&self) -> usize {
        self.embeddings.embedding_size()
    }
}
