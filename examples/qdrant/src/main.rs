#[cfg(any(feature = "intel-mkl-static", feature = "intel-mkl-system"))]
extern crate intel_mkl_src as _src;
#[cfg(any(feature = "netlib-static", feature = "netlib-system"))]
extern crate netlib_src as _src;
#[cfg(any(feature = "openblas-static", feature = "openblas-system"))]
extern crate openblas_src as _src;

use std::cell::RefCell;
use std::error::Error;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;

use clap::Parser;
use finalfusion::prelude::*;
use sif_embedding::SentenceEmbedder;
use sif_embedding::USif;
use sif_embedding::WordProbabilities;
use unicode_normalization::UnicodeNormalization;
use vibrato::dictionary::Dictionary;
use vibrato::tokenizer::worker::Worker as VibratoWorker;
use vibrato::Tokenizer;
use wordfreq_model::ModelKind;

const CATEGORIES: &[&str] = &[
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

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'i', long)]
    input_dir: PathBuf,

    #[arg(short = 'f', long)]
    fifu_model: PathBuf,

    #[arg(short = 'v', long)]
    vibrato_model: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let word_embeddings = {
        let mut reader = BufReader::new(File::open(&args.fifu_model)?);
        Embeddings::<VocabWrap, StorageWrap>::mmap_embeddings(&mut reader)?
    };
    eprintln!("word_embeddings.len() = {}", word_embeddings.len());
    eprintln!("word_embeddings.dims() = {}", word_embeddings.dims());

    let unigram_lm = wordfreq_model::load_wordfreq(ModelKind::LargeJa)?;
    eprintln!("unigram_lm.n_words() = {}", unigram_lm.n_words());

    let tokenizer = {
        let reader = zstd::Decoder::new(File::open(args.vibrato_model)?)?;
        let dict = Dictionary::read(reader)?;
        Tokenizer::new(dict)
            .ignore_space(true)?
            .max_grouping_len(24)
    };

    let (sentences, labels) = load_livedoor_data(&args.input_dir)?;
    eprintln!("{} sentences", sentences.len());
    eprintln!("{} labels", labels.len());

    // Tokenize sentences.
    let worker = RefCell::new(tokenizer.new_worker());
    let tokenized: Vec<String> = sentences.iter().map(|s| tokenize(s, &worker)).collect();

    let sif = USif::new(&word_embeddings, &unigram_lm);
    let sent_embeddings = sif.embeddings(&tokenized)?;
    eprintln!("sent_embeddings.shape() = {:?}", sent_embeddings.shape());

    Ok(())
}

pub fn load_livedoor_data<P: AsRef<Path>>(
    data_dir: P,
) -> Result<(Vec<String>, Vec<usize>), Box<dyn Error>> {
    let data_dir = data_dir.as_ref().to_str().unwrap();

    let mut sentences = vec![];
    let mut labels = vec![];
    for (label, &cate) in CATEGORIES.iter().enumerate() {
        for filepath in glob::glob(&format!("{data_dir}/{cate}/{cate}-*.txt"))? {
            let filepath = filepath?;
            let sentence = read_sentence(&filepath)?;
            sentences.push(sentence);
            labels.push(label);
        }
    }
    Ok((sentences, labels))
}

fn read_sentence<P: AsRef<Path>>(filepath: P) -> Result<String, Box<dyn Error>> {
    let file = File::open(filepath)?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();
    // Skip the first three lines for the header.
    Ok(lines[3..].join("\n"))
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
