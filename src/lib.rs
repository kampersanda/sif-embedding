pub mod sif;
pub mod util;
pub mod word_embeddings;

#[cfg(test)]
pub mod tool_test;

pub type Float = f64;

pub use sif::Sif;
pub use word_embeddings::WordEmbeddings;
