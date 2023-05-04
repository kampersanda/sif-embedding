use std::collections::HashMap;

use ndarray::Array2;

use crate::util;
use crate::{Float, WordEmbeddings};

pub struct Sif {
    word_embeddings: WordEmbeddings,
    word2weight: HashMap<String, Float>,
    separator: char,
    param_a: Float,
    principal_component: Option<Float>,
}

impl Sif {
    ///
    pub fn new<W>(word_embeddings: WordEmbeddings, word_weights: &[(W, Float)]) -> Self
    where
        W: AsRef<str>,
    {
        let word2weight = word_weights
            .iter()
            .map(|(word, weight)| (word.as_ref().to_string(), *weight))
            .collect();
        Self {
            word_embeddings,
            word2weight,
            separator: ' ',
            param_a: 1e-3,
            principal_component: None,
        }
    }

    pub fn embedding_size(&self) -> usize {
        self.word_embeddings.embedding_size()
    }

    /// Computes embeddings for the input sentences,
    /// returning a 2D-array of shape `(sentences.len(), embedding_size())`.
    ///
    /// # Arguments
    ///
    /// - `sentences`: Sentences to be embedded.
    pub fn embeddings<S>(mut self, sentences: &[S]) -> (Array2<Float>, Self)
    where
        S: AsRef<str>,
    {
        self.update_word_weigths();
        let sent_embeddings = self.weighted_average_embeddings(sentences);
        let principal_components = util::principal_components(&sent_embeddings, 1);
        self.principal_component = Some(principal_components[[0, 0]]);
        println!("principal_component = {:?}", self.principal_component);
        if self.principal_component.unwrap() == 1. {
            panic!("principal_component == 1.");
        }
        let sent_embeddings = self.subtract_principal_components(sent_embeddings);
        (sent_embeddings, self)
    }

    /// a: Hyperparameter in Eq (3)
    fn update_word_weigths(&mut self) {
        let sum_weight = self.word2weight.values().fold(0., |acc, w| acc + w);
        self.word2weight
            .values_mut()
            .for_each(|w| *w = self.param_a / (self.param_a + *w / sum_weight));
    }

    /// Lines 1--3
    fn weighted_average_embeddings<S>(&self, sentences: &[S]) -> Array2<Float>
    where
        S: AsRef<str>,
    {
        let mut sent_embeddings =
            Array2::<Float>::zeros((sentences.len(), self.word_embeddings.embedding_size()));
        for (sent, mut sent_embedding) in sentences.iter().zip(sent_embeddings.rows_mut()) {
            let sent = sent.as_ref();
            let mut n_words = 0.;
            let mut word_weight = 0.;
            for word in sent.split(self.separator) {
                n_words += 1.;
                if let Some(&weight) = self.word2weight.get(word) {
                    word_weight += weight;
                    // } else {
                    //     word_weight += 1.;
                }
                if let Some(word_embedding) = self.word_embeddings.lookup(word) {
                    sent_embedding += &word_embedding;
                }
            }
            sent_embedding *= word_weight;
            sent_embedding /= n_words;
        }
        sent_embeddings
    }

    /// Lines 5--7
    fn subtract_principal_components(&mut self, sent_embeddings: Array2<Float>) -> Array2<Float> {
        let project = self.principal_component.unwrap();
        sent_embeddings.to_owned() - &(sent_embeddings * project)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sif_basic() {
        let we_text = "A 0.0 1.0 2.0\nBB -3.0 -4.0 -5.0\nCCC 6.0 -7.0 8.0\nDDDD -9.0 10.0 -11.0\n";
        let we = WordEmbeddings::from_text(we_text.as_bytes()).unwrap();

        let (se, _) = Sif::new(we, &[("A", 1.), ("BB", 2.), ("CCC", 3.), ("DDDD", 4.)])
            .embeddings(&["A BB CCC DDDD", "BB CCC", "A B C", "Z", ""]);
        assert_eq!(se.shape(), &[5, 3]);
    }
}
