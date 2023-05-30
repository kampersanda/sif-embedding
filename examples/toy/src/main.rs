// https://github.com/rust-ndarray/ndarray-linalg/blob/ndarray-linalg-v0.16.0/lax/src/lib.rs
#[cfg(any(feature = "intel-mkl-static", feature = "intel-mkl-system"))]
extern crate intel_mkl_src;
#[cfg(any(feature = "netlib-static", feature = "netlib-system"))]
extern crate netlib_src;
#[cfg(any(feature = "openblas-static", feature = "openblas-system"))]
extern crate openblas_src;

use std::io::BufReader;

use finalfusion::compat::text::ReadText;
use finalfusion::embeddings::Embeddings;

use sif_embedding::{Sif, UnigramLM};

fn main() {
    // Load word embeddings from a pretrained model.
    let word_model = "las 0.0 1.0 2.0\nvegas -3.0 -4.0 -5.0\n";
    let mut reader = BufReader::new(word_model.as_bytes());
    let word_embeddings = Embeddings::read_text(&mut reader).unwrap();

    // Create a unigram language model.
    let word_weights = [("las", 10.), ("vegas", 20.)];
    let unigram_lm = UnigramLM::new(word_weights);

    // Compute sentence embeddings.
    let sif = Sif::new(&word_embeddings, &unigram_lm);
    let sent_embeddings = sif.embeddings(["go to las vegas", "mega vegas"]);

    println!("{sent_embeddings}");
}
