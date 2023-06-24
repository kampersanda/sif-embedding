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
use sif_embedding::util;
use sif_embedding::Model;
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
    input_vibrato_zst: String,

    #[arg(short = 'm', long, default_value = "sif")]
    method: MethodKind,
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

    let tokenizer = {
        let reader = zstd::Decoder::new(File::open(args.input_vibrato_zst)?)?;
        let dict = Dictionary::read(reader)?;
        Tokenizer::new(dict)
            .ignore_space(true)?
            .max_grouping_len(24)
    };

    let (sentences, labels) = dataset::load_livedoor_data(&args.data_dir)?;
    eprintln!("{} sentences", sentences.len());
    eprintln!("{} labels", labels.len());

    // Tokenize sentences.
    let worker = RefCell::new(tokenizer.new_worker());
    let tokenized: Vec<String> = sentences.iter().map(|s| tokenize(s, &worker)).collect();

    let sent_embeddings = match args.method {
        MethodKind::Sif => {
            let mut sif = Sif::new(&word_embeddings, &unigram_lm);
            sif.fit_embeddings(&tokenized)?
        }
        MethodKind::USif => {
            let mut sif = USif::new(&word_embeddings, &unigram_lm);
            sif.fit_embeddings(&tokenized)?
        }
    };
    eprintln!("sent_embeddings.shape() = {:?}", sent_embeddings.shape());

    // NN scores.
    let mut num_corrects = 0;
    for (i, e1) in sent_embeddings.outer_iter().enumerate() {
        let mut top_index = 0;
        let mut top_score = -1.;
        for (j, e2) in sent_embeddings.outer_iter().enumerate() {
            if i == j {
                continue;
            }
            let score = util::cosine_similarity(&e1, &e2).unwrap_or(-1.);
            if score > top_score {
                top_index = j;
                top_score = score;
            }
        }
        if labels[i] == labels[top_index] {
            num_corrects += 1;
        }
    }
    println!(
        "NN accuracy: {:.3}",
        num_corrects as f64 / labels.len() as f64
    );

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
