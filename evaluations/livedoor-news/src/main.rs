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
use linfa::prelude::*;
use linfa_logistic::MultiLogisticRegression;
// use linfa_svm::Svm;
use ndarray::Array1;
use polars::prelude::*;
use sif_embedding::SentenceEmbedder;
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

    // Load dataset
    let mut df = dataset::load_livedoor_data(&args.data_dir)?;

    // Tokenize sentences
    {
        let tokenizer = {
            let reader = zstd::Decoder::new(File::open(args.input_vibrato)?)?;
            let dict = Dictionary::read(reader)?;
            Tokenizer::new(dict)
                .ignore_space(true)?
                .max_grouping_len(24)
        };
        let vibrato_worker = RefCell::new(tokenizer.new_worker());
        let tokenized = df
            .column("sentence")?
            .utf8()?
            .into_iter()
            .map(|s| tokenize(s.unwrap(), &vibrato_worker))
            .collect::<Vec<_>>();
        let tokenized = Series::new("tokenized", tokenized);
        df.hstack_mut(&[tokenized])?;
    }

    eprintln!("{:?}", df);

    let train_size = (df.height() as f64 * 0.75) as usize;
    let test_size = df.height() - train_size;
    let train_df = df.slice(0, train_size);
    let test_df = df.slice(train_size as i64, test_size);

    let sent_embedder = USif::new(&word_embeddings, &word_probs);

    // Train
    let train_tokenized = train_df
        .column("tokenized")?
        .utf8()?
        .into_iter()
        .map(|s| s.unwrap())
        .collect::<Vec<_>>();
    let (train_embeddings, sent_embedder) = sent_embedder.fit_embeddings(&train_tokenized)?;
    let train_targets = Array1::from_iter(
        train_df
            .column("label")?
            .u32()?
            .into_iter()
            .map(|t| t.unwrap()),
    );
    let train_dataset = Dataset::new(train_embeddings, train_targets);

    // Test
    let test_tokenized = test_df
        .column("tokenized")?
        .utf8()?
        .into_iter()
        .map(|s| s.unwrap())
        .collect::<Vec<_>>();
    let test_embeddings = sent_embedder.embeddings(&test_tokenized)?;
    let test_targets = Array1::from_iter(
        test_df
            .column("label")?
            .u32()?
            .into_iter()
            .map(|t| t.unwrap()),
    );
    let test_dataset = Dataset::new(test_embeddings, test_targets);

    // Training
    eprintln!("Training...");
    let classifier = MultiLogisticRegression::default()
        .max_iterations(150)
        .fit(&train_dataset)?;

    // Prediction
    eprintln!("Predicting...");
    let predicted = classifier.predict(test_dataset);
    println!("predicted = {:?}", predicted);

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
