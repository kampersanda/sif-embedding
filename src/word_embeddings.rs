pub mod io;

use hashbrown::HashMap;
use ndarray::{self, Array2};

use crate::Embedding;

#[derive(Debug)]
pub struct WordEmbeddings {
    embeddings: Array2<f32>,
    word2idx: HashMap<String, usize>,
}

impl WordEmbeddings {
    pub fn lookup(&self, word: &str) -> Option<Embedding> {
        if let Some(&idx) = self.word2idx.get(word) {
            let row = self.embeddings.slice(ndarray::s![idx, ..]);
            Some(row.into())
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.embeddings.shape()[0]
    }

    pub fn embedding_size(&self) -> usize {
        self.embeddings.shape()[1]
    }
}
