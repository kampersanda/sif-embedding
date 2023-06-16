#[cfg(any(feature = "intel-mkl-static", feature = "intel-mkl-system"))]
extern crate intel_mkl_src as _src;
#[cfg(any(feature = "netlib-static", feature = "netlib-system"))]
extern crate netlib_src as _src;
#[cfg(any(feature = "openblas-static", feature = "openblas-system"))]
extern crate openblas_src as _src;

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
use vibrato::dictionary::Dictionary;
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
    input_vibrato: String,
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

    let dict = {
        let reader = zstd::Decoder::new(File::open(args.input_vibrato)?)?;
        Dictionary::read(reader)?
    };
    let tokenizer = Tokenizer::new(dict)
        .ignore_space(true)?
        .max_grouping_len(24);
    let mut worker = tokenizer.new_worker();

    let categories = vec![
        "dokujo-tsushin",
        "it-life-hack",
        "kaden-channel",
        "livedoor-homme",
        "movie-enter",
        "peachy",
        "smax",
        "sports-watch",
        "topic-news",
    ];

    let mut sentences = vec![];
    let mut labels = vec![];

    let data_dir = args.data_dir;
    for (label, &cate) in categories.iter().enumerate() {
        for filepath in glob::glob(&format!("{data_dir}/{cate}/{cate}*.txt"))? {
            let filepath = filepath?;
            let sentence = read_sentence(&filepath)?;
            sentences.push(sentence);
            labels.push(label);
        }
    }

    println!("{} sentences", sentences.len());
    println!("{}", sentences[0]);

    Ok(())
}

fn read_sentence<P: AsRef<Path>>(filepath: P) -> Result<String, Box<dyn Error>> {
    let file = File::open(filepath)?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();
    Ok(lines[3..].join("\n"))
}
