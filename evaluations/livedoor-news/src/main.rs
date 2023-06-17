#[cfg(any(feature = "intel-mkl-static", feature = "intel-mkl-system"))]
extern crate intel_mkl_src as _src;
#[cfg(any(feature = "netlib-static", feature = "netlib-system"))]
extern crate netlib_src as _src;
#[cfg(any(feature = "openblas-static", feature = "openblas-system"))]
extern crate openblas_src as _src;

mod dataset;

use std::cell::RefCell;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use clap::Parser;
use finalfusion::prelude::*;
use ndarray::Array2;
use ndarray_stats::CorrelationExt;
use sif_embedding::util;
use sif_embedding::{Float, Sif, UnigramLanguageModel, WordEmbeddings};
use tantivy::tokenizer::*;
use unicode_normalization::UnicodeNormalization;
use vibrato::dictionary::Dictionary;
use vibrato::tokenizer::worker::Worker as VibratoWorker;
use vibrato::Tokenizer;
use wordfreq_model::{self, ModelKind};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'd', long)]
    data_dir: String,

    #[arg(short = 'f', long)]
    input_fifu: String,

    #[arg(short = 'v', long)]
    input_vibrato_zst: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let word_embeddings = {
        let mut reader = BufReader::new(File::open(&args.input_fifu)?);
        Embeddings::<VocabWrap, StorageWrap>::mmap_embeddings(&mut reader)?
    };
    eprintln!("word_embeddings.len() = {}", word_embeddings.len());
    eprintln!("word_embeddings.dims() = {}", word_embeddings.dims());

    let unigram_lm = wordfreq_model::load_wordfreq(ModelKind::LargeJa)?;
    let sif = Sif::new(&word_embeddings, &unigram_lm);

    let tokenizer = {
        let reader = zstd::Decoder::new(File::open(args.input_vibrato_zst)?)?;
        let dict = Dictionary::read(reader)?;
        Tokenizer::new(dict)
            .ignore_space(true)?
            .max_grouping_len(24)
    };

    let (sentences, labels) = dataset::load_livedoor_data(&args.data_dir)?;
    println!("{} sentences", sentences.len());
    println!("{} labels", labels.len());

    // Tokenize sentences.
    let worker = RefCell::new(tokenizer.new_worker());
    let tokenized: Vec<String> = sentences.iter().map(|s| tokenize(s, &worker)).collect();
    println!("0: {}", tokenized[0]);
    println!("1: {}", tokenized[1]);

    // Compute sentence embeddings.
    let sent_embeddings = sif.embeddings(&tokenized);
    println!("shape {:?}", sent_embeddings.shape());

    let e1 = &sent_embeddings.row(0);
    let e2 = &sent_embeddings.row(1);
    let score = util::cosine_similarity(e1, e2).unwrap_or(0.); // ok?
    println!("score = {}", score);

    Ok(())
}

fn tokenize(sentence: &str, worker: &RefCell<VibratoWorker>) -> String {
    let mut surfaces = vec![];
    for line in sentence.split('\n') {
        let line = line.nfkc().collect::<String>();
        let mut worker = worker.borrow_mut();
        worker.reset_sentence(line);
        worker.tokenize();
        surfaces.extend(worker.token_iter().map(|t| t.surface().to_string()));
    }
    surfaces.join(" ")
}
