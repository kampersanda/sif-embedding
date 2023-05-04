pub mod sif;
pub mod util;
pub mod word_embeddings;

pub use word_embeddings::WordEmbeddings;

pub type Float = f32;

#[cfg(test)]
pub mod tool_test;

pub use sif::Sif;
