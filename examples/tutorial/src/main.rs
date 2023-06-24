// These declarations are required to recognize the backend.
// https://github.com/rust-ndarray/ndarray-linalg/blob/ndarray-linalg-v0.16.0/lax/src/lib.rs
#[cfg(any(feature = "intel-mkl-static", feature = "intel-mkl-system"))]
extern crate intel_mkl_src as _src;
#[cfg(any(feature = "netlib-static", feature = "netlib-system"))]
extern crate netlib_src as _src;
#[cfg(any(feature = "openblas-static", feature = "openblas-system"))]
extern crate openblas_src as _src;

use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use clap::Parser;
use finalfusion::prelude::*;
use wordfreq_model::{self, ModelKind};

use sif_embedding::SentenceEmbedder;
use sif_embedding::Sif;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'f', long)]
    input_fifu: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let word_embeddings = {
        let mut reader = BufReader::new(File::open(&args.input_fifu)?);
        Embeddings::<VocabWrap, StorageWrap>::mmap_embeddings(&mut reader)?
    };

    let unigram_lm = wordfreq_model::load_wordfreq(ModelKind::LargeEn)?;

    let sentences = vec![
        "This is a sentence.",
        "This is another sentence.",
        "This is a third sentence.",
    ];

    let mut sif = Sif::new(&word_embeddings, &unigram_lm);
    let sent_embeddings = sif.fit_embeddings(&sentences)?;

    println!("Sentence embeddings: {:?}", sent_embeddings);

    Ok(())
}
