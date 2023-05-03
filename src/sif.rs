use std::collections::HashMap;

use ndarray::Array2;

use crate::WordEmbeddings;

pub struct Sif<'a, W, S> {
    word_embeddings: WordEmbeddings,
    word_weights: &'a [(W, f32)],
    sentences: &'a [S],
    separator: char,
    param_a: f32,
}

impl<'a, W, S> Sif<'a, W, S>
where
    W: AsRef<str>,
    S: AsRef<str>,
{
    pub fn new(
        word_embeddings: WordEmbeddings,
        word_weights: &'a [(W, f32)],
        sentences: &'a [S],
    ) -> Self {
        Self {
            word_embeddings,
            word_weights,
            sentences,
            separator: ' ',
            param_a: 1e-3,
        }
    }

    /// a: Hyperparameter in Eq (3)
    fn word_to_weight(&self) -> HashMap<String, f32> {
        let mut sum_weight = 0.;
        let mut word2weight = HashMap::new();
        for (word, weight) in self.word_weights {
            sum_weight += weight;
            *word2weight.entry(word.as_ref().to_string()).or_insert(0.) += weight;
        }
        for (_, weight) in word2weight.iter_mut() {
            *weight = self.param_a / (self.param_a + *weight / sum_weight);
        }
        word2weight
    }

    /// Lines 1--3
    fn averaged_embeddings(&self) -> Array2<f32> {
        let mut sent_embeddings =
            Array2::<f32>::zeros((self.sentences.len(), self.word_embeddings.embedding_size()));
        for (sent, mut sent_embedding) in self.sentences.iter().zip(sent_embeddings.rows_mut()) {
            let sent = sent.as_ref();
            let mut num_words = 0;
            for word in sent.split(self.separator) {
                num_words += 1;
                if let Some(word_embedding) = self.word_embeddings.lookup(word) {
                    sent_embedding += &word_embedding;
                }
            }
        }
        sent_embeddings
    }
}
