//! WordEmbeddings implementations for [`finalfusion::embeddings::Embeddings`].
use crate::Float;
use crate::WordEmbeddings;

use finalfusion::embeddings::Embeddings;
use finalfusion::storage::Storage;
use finalfusion::vocab::Vocab;
use ndarray::{CowArray, Ix1};

impl<V, S> WordEmbeddings for Embeddings<V, S>
where
    V: Vocab,
    S: Storage,
{
    fn embedding(&self, word: &str) -> Option<CowArray<Float, Ix1>> {
        self.embedding(word)
    }

    fn embedding_size(&self) -> usize {
        self.dims()
    }

    fn n_words(&self) -> usize {
        self.vocab().words_len()
    }

    fn words(&self) -> Box<dyn Iterator<Item = String> + '_> {
        Box::new(self.vocab().words().iter().cloned())
    }
}
