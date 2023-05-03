pub mod sif;
pub mod word_embeddings;

pub use word_embeddings::WordEmbeddings;

use ndarray::{CowArray, Ix1};

pub type Embedding<'a> = CowArray<'a, f32, Ix1>;

#[cfg(test)]
pub mod tool_test;
