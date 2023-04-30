pub mod sif;

use ndarray::{CowArray, Ix1};

pub type Embedding<'a> = CowArray<'a, f32, Ix1>;

pub trait WordEmbeddings {
    fn lookup(&self, word: &str) -> Option<Embedding>;
}
