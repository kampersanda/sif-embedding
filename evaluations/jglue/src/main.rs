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
use std::str::FromStr;

use clap::Parser;
use finalfusion::prelude::*;
use serde::Deserialize;
use sif_embedding::util;
use sif_embedding::SentenceEmbedder;
use sif_embedding::Sif;
use sif_embedding::USif;
use unicode_normalization::UnicodeNormalization;
use vibrato::dictionary::Dictionary;
use vibrato::tokenizer::worker::Worker as VibratoWorker;
use vibrato::Tokenizer;
use wordfreq_model::ModelKind;

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
    data_file: String,

    #[arg(short = 'f', long)]
    input_fifu: String,

    #[arg(short = 'v', long)]
    input_vibrato: String,

    #[arg(short = 'm', long, default_value = "sif")]
    method: MethodKind,

    #[arg(short = 'n', long)]
    n_components: Option<usize>,
}

#[derive(Deserialize)]
struct JstsExample {
    _sentence_pair_id: String,
    _yjcaptions_id: String,
    sentence1: String,
    sentence2: String,
    label: f64,
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
        let reader = zstd::Decoder::new(File::open(args.input_vibrato)?)?;
        let dict = Dictionary::read(reader)?;
        Tokenizer::new(dict)
            .ignore_space(true)?
            .max_grouping_len(24)
    };

    let (gold_scores, sentences) = load_jsts_data(&args.data_file)?;

    let worker = RefCell::new(tokenizer.new_worker());
    let sentences: Vec<String> = sentences.iter().map(|s| tokenize(s, &worker)).collect();

    let (pearson, spearman) = match args.method {
        MethodKind::Sif => {
            let param_a = sif_embedding::sif::DEFAULT_PARAM_A;
            let n_components = args
                .n_components
                .unwrap_or(sif_embedding::sif::DEFAULT_N_COMPONENTS);
            let model = Sif::with_parameters(&word_embeddings, &unigram_lm, param_a, n_components)?;
            evaluate(model, &gold_scores, &sentences)?
        }
        MethodKind::USif => {
            let n_components = args
                .n_components
                .unwrap_or(sif_embedding::usif::DEFAULT_N_COMPONENTS);
            let model = USif::with_parameters(&word_embeddings, &unigram_lm, n_components);
            evaluate(model, &gold_scores, &sentences)?
        }
    };
    println!("{pearson}\t{spearman}");

    Ok(())
}

fn load_jsts_data(input_path: &str) -> Result<(Vec<f64>, Vec<String>), Box<dyn Error>> {
    let input_lines: Vec<String> = BufReader::new(File::open(input_path)?)
        .lines()
        .map(|l| l.unwrap())
        .collect();
    let mut gold_scores = vec![];
    let mut sentences = vec![];
    for line in input_lines {
        let example: JstsExample = serde_json::from_str(&line)?;
        gold_scores.push(example.label);
        sentences.push(example.sentence1);
        sentences.push(example.sentence2);
    }
    Ok((gold_scores, sentences))
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

fn evaluate<M>(
    model: M,
    gold_scores: &[f64],
    sentences: &[String],
) -> Result<(f64, f64), Box<dyn Error>>
where
    M: SentenceEmbedder,
{
    let (sent_embeddings, _) = model.fit_embeddings(sentences)?;
    let n_examples = gold_scores.len();
    let mut pred_scores = Vec::with_capacity(n_examples);
    for i in 0..n_examples {
        let e1 = &sent_embeddings.row(i * 2);
        let e2 = &sent_embeddings.row(i * 2 + 1);
        let score = util::cosine_similarity(e1, e2).unwrap_or(-1.) as f64; // ok?
        pred_scores.push(score);
    }
    let pearson = pearson_correlation(&pred_scores, gold_scores);
    let spearman = spearman_correlation(&pred_scores, gold_scores);
    Ok((pearson, spearman))
}

fn pearson_correlation(s1: &[f64], s2: &[f64]) -> f64 {
    assert_eq!(s1.len(), s2.len());
    rgsl::statistics::correlation(s1, 1, s2, 1, s1.len())
}

fn spearman_correlation(s1: &[f64], s2: &[f64]) -> f64 {
    assert_eq!(s1.len(), s2.len());
    let mut work = Vec::with_capacity(2 * s1.len());
    rgsl::statistics::spearman(s1, 1, s2, 1, s1.len(), &mut work)
}
