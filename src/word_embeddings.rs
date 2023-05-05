pub mod io;

use hashbrown::HashMap;
use ndarray::{self, Array2, CowArray, Ix1};

use crate::Float;

#[derive(Debug, Clone)]
pub struct WordEmbeddings {
    word2idx: HashMap<String, usize>,
    embeddings: Array2<Float>,
}

impl WordEmbeddings {
    pub fn lookup(&self, word: &str) -> Option<CowArray<'_, Float, Ix1>> {
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
