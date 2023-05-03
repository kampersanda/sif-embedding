use std::collections::HashMap;

use ndarray::Array2;

use crate::util;
use crate::{Float, WordEmbeddings};

pub struct Sif {
    word_embeddings: WordEmbeddings,
    word2weight: HashMap<String, Float>,
    separator: char,
    param_a: Float,
    param_k: usize,
    principal_components: Option<Array2<Float>>,
}

impl Sif {
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
            param_k: 1,
            principal_components: None,
        }
    }

    pub fn embeddings<S>(&mut self, sentences: &[S]) -> Array2<Float>
    where
        S: AsRef<str>,
    {
        self.update_word_weigths();
        let mut sent_embeddings = self.weighted_average_embeddings(sentences);
        self.principal_components =
            Some(util::principal_components(&sent_embeddings, self.param_k));
        self.subtract_principal_components(&mut sent_embeddings);
        sent_embeddings
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
                } else {
                    word_weight += 1.;
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

    fn subtract_principal_components(&mut self, sent_embeddings: &mut Array2<Float>) {
        let proj = self.principal_components.as_ref().unwrap();
        for mut sent_embedding in sent_embeddings.rows_mut() {
            let sub = proj.dot(&sent_embedding);
            sent_embedding -= &sub;
        }
    }
}
