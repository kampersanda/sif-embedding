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
use std::io::BufReader;
use std::str::FromStr;

use clap::Parser;
use finalfusion::prelude::*;
use linfa::Dataset;
use ndarray::Array1;
use sif_embedding::SentenceEmbedder;
use sif_embedding::Sif;
use sif_embedding::USif;
use unicode_normalization::UnicodeNormalization;
use vibrato::dictionary::Dictionary;
use vibrato::tokenizer::worker::Worker as VibratoWorker;
use vibrato::Tokenizer;
use wordfreq_model::{self, ModelKind};

#[derive(Clone, Debug)]
enum MethodKind {
    Sif,
    USif,
}

impl FromStr for MethodKind {
    type Err = &'static str;
    fn from_str(mode: &str) -> Result<Self, Self::Err> {
        match mode {
            "sif" => Ok(Self::Sif),
            "usif" => Ok(Self::USif),
            _ => Err("Could not parse a mode"),
        }
    }
}

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

    let word_probs = wordfreq_model::load_wordfreq(ModelKind::LargeJa)?;

    let tokenizer = {
        let reader = zstd::Decoder::new(File::open(args.input_vibrato)?)?;
        let dict = Dictionary::read(reader)?;
        Tokenizer::new(dict)
            .ignore_space(true)?
            .max_grouping_len(24)
    };
    let vibrato_worker = RefCell::new(tokenizer.new_worker());

    let records = {
        let mut records = dataset::load_livedoor_data(&args.data_dir, true)?;
        for record in &mut records {
            record.sentence = tokenize(&record.sentence, &vibrato_worker);
        }
        records
    };
    eprintln!("records.len() = {}", records.len());

    let train_size = (records.len() as f64 * 0.75) as usize;
    let test_size = records.len() - train_size;
    eprintln!("train_size = {}", train_size);
    eprintln!("test_size = {}", test_size);

    let sentences = records
        .iter()
        .map(|r| r.sentence.as_str())
        .collect::<Vec<_>>();

    let targets = Array1::from_iter(records.iter().map(|r| r.label));
    let train_targets = targets.slice(ndarray::s![..train_size]);
    let test_targets = targets.slice(ndarray::s![train_size..]);
    eprintln!("train_targets.len() = {}", train_targets.len());
    eprintln!("test_targets.len() = {}", test_targets.len());

    let (train_embeddings, sent_embedder) =
        USif::new(&word_embeddings, &word_probs).fit_embeddings(&sentences[..train_size])?;
    let train_dataset = Dataset::new(train_embeddings, train_targets.to_owned());
    eprintln!("train_dataset.ntargets() = {}", train_dataset.ntargets());

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
