#[cfg(any(feature = "intel-mkl-static", feature = "intel-mkl-system"))]
extern crate intel_mkl_src as _src;
#[cfg(any(feature = "netlib-static", feature = "netlib-system"))]
extern crate netlib_src as _src;
#[cfg(any(feature = "openblas-static", feature = "openblas-system"))]
extern crate openblas_src as _src;

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use finalfusion::prelude::*;
use sif_embedding::SentenceEmbedder;
use sif_embedding::Sif;
use sif_embedding::USif;
use sif_embedding::WordProbabilities;
use vtext::tokenize::Tokenizer;
use vtext::tokenize::VTextTokenizerParams;
use wordfreq_model::ModelKind;

const BATCH_SIZE: usize = 1 << 16;
const RUNS: usize = 5;

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
    dataset_file: PathBuf,

    #[arg(short = 'f', long)]
    fifu_model: PathBuf,

    #[arg(short = 'm', long, default_value = "sif")]
    method: MethodKind,

    #[arg(short = 'n', long)]
    n_components: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let sentences = load_wiki1m_dataset(&args.dataset_file)?;
    let avg_sent_len = average_sentence_length(&sentences);
    eprintln!("Number of loaded sentences: {}", sentences.len());
    eprintln!("Average sentence length: {:.2}", avg_sent_len);

    let mut reader = BufReader::new(File::open(&args.fifu_model)?);
    let word_embeddings = Embeddings::<VocabWrap, StorageWrap>::mmap_embeddings(&mut reader)?;
    eprintln!("word_embeddings.len() = {}", word_embeddings.len());
    eprintln!("word_embeddings.dims() = {}", word_embeddings.dims());

    let unigram_lm = wordfreq_model::load_wordfreq(ModelKind::LargeEn)?;
    eprintln!("unigram_lm.n_words() = {}", unigram_lm.n_words());

    match args.method {
        MethodKind::Sif => {
            let param_a = sif_embedding::sif::DEFAULT_PARAM_A;
            let n_components = args
                .n_components
                .unwrap_or(sif_embedding::sif::DEFAULT_N_COMPONENTS);
            let model = Sif::with_parameters(&word_embeddings, &unigram_lm, param_a, n_components)?;
            let model = model.fit(&sentences)?;
            benchmark(model, &sentences)?;
        }
        MethodKind::USif => {
            let n_components = args
                .n_components
                .unwrap_or(sif_embedding::usif::DEFAULT_N_COMPONENTS);
            let model = USif::with_parameters(&word_embeddings, &unigram_lm, n_components);
            let model = model.fit(&sentences)?;
            benchmark(model, &sentences)?;
        }
    }

    Ok(())
}

/// https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse
fn load_wiki1m_dataset<P: AsRef<Path>>(dataset_file: P) -> Result<Vec<String>> {
    let reader = BufReader::new(File::open(dataset_file)?);
    let tokenizer = VTextTokenizerParams::default().lang("en").build()?;

    let mut sentences = vec![];
    let separator = sif_embedding::DEFAULT_SEPARATOR.to_string();

    for line in reader.lines() {
        let line = line?;
        let sentence = tokenizer
            .tokenize(&line)
            .collect::<Vec<_>>()
            .join(&separator)
            .to_lowercase();
        sentences.push(sentence);
    }

    Ok(sentences)
}

fn average_sentence_length<S>(sentences: &[S]) -> f32
where
    S: AsRef<str>,
{
    let mut n_words = 0;
    for sent in sentences {
        let sent = sent.as_ref();
        n_words += sent.split(sif_embedding::DEFAULT_SEPARATOR).count();
    }
    n_words as f32 / sentences.len() as f32
}

fn benchmark<M: SentenceEmbedder>(model: M, sentences: &[String]) -> Result<()> {
    let mut sentences_per_sec_results = Vec::with_capacity(RUNS);

    for r in 0..=RUNS {
        if r == 0 {
            eprintln!("Warming up...");
        } else {
            eprintln!("Benchmarking...");
            eprintln!("Run {}/{}", r, RUNS);
        }

        let start = Instant::now();
        for batch in sentences.chunks(BATCH_SIZE) {
            let sent_embeddings = model.embeddings(batch)?;
            eprintln!("sent_embeddings.shape() = {:?}", sent_embeddings.shape());
        }
        let elapsed = start.elapsed();
        let ms_per_sentence = elapsed.as_secs_f64() / sentences.len() as f64 * 1000.0;
        let sentences_per_sec = sentences.len() as f64 / elapsed.as_secs_f64();

        eprintln!("Total elapsed time: {:?}", elapsed);
        eprintln!("Milli seconds per sentence: {:.4}", ms_per_sentence);
        eprintln!("Sentences per second: {:.1}", sentences_per_sec);

        if r != 0 {
            sentences_per_sec_results.push(sentences_per_sec);
        }
    }

    let n_runs = sentences_per_sec_results.len() as f64;
    let sentences_per_sec_mean = sentences_per_sec_results.iter().sum::<f64>() / n_runs;
    let sentences_per_sec_stddev = (sentences_per_sec_results
        .iter()
        .map(|x| (x - sentences_per_sec_mean).powi(2))
        .sum::<f64>()
        / n_runs)
        .sqrt();

    eprintln!(
        "Sentences per second: {:.1} ± {:.1}",
        sentences_per_sec_mean, sentences_per_sec_stddev
    );

    Ok(())
}
