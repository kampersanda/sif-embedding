#[cfg(any(feature = "intel-mkl-static", feature = "intel-mkl-system"))]
extern crate intel_mkl_src as _src;
#[cfg(any(feature = "netlib-static", feature = "netlib-system"))]
extern crate netlib_src as _src;
#[cfg(any(feature = "openblas-static", feature = "openblas-system"))]
extern crate openblas_src as _src;

use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

use clap::Parser;
use finalfusion::prelude::*;
use ndarray::Array2;
use ndarray_stats::CorrelationExt;
use wordfreq::{self, WordFreq};

use sif_embedding::util;
use sif_embedding::{Float, Sif, UnigramLanguageModel, WordEmbeddings};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'f', long)]
    input_fifu: String,

    #[arg(short = 'w', long)]
    input_weights: String,

    #[arg(short = 'c', long)]
    corpora_dir: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let word_embeddings = {
        let mut reader = BufReader::new(File::open(&args.input_fifu)?);
        Embeddings::<VocabWrap, StorageWrap>::mmap_embeddings(&mut reader)?
    };
    let word_freq = {
        let reader = BufReader::new(File::open(&args.input_weights)?);
        let word_weights = wordfreq::word_weights_from_text(reader)?;
        WordFreq::new(word_weights)
    };

    eprintln!("word_embeddings.len() = {}", word_embeddings.len());
    eprintln!(
        "word_embeddings.embedding_size() = {}",
        word_embeddings.dims()
    );

    let sif = Sif::new(&word_embeddings, &word_freq);

    let corpora_dir = args.corpora_dir;
    let corpora = vec![
        (
            "STS12-en-test",
            vec![
                "MSRpar",
                "MSRvid",
                "SMTeuroparl",
                "surprise.OnWN",
                "surprise.SMTnews",
            ],
        ),
        ("STS13-en-test", vec!["FNWN", "headlines", "OnWN"]),
        (
            "STS14-en-test",
            vec![
                "deft-forum",
                "deft-news",
                "headlines",
                "images",
                "OnWN",
                "tweet-news",
            ],
        ),
        (
            "STS15-en-test",
            vec![
                "answers-forums",
                "answers-students",
                "belief",
                "headlines",
                "images",
            ],
        ),
        (
            "STS16-en-test",
            vec![
                "answer-answer",
                "headlines",
                "plagiarism",
                "postediting",
                "question-question",
            ],
        ),
    ];

    for (year, files) in corpora {
        println!("{year}");
        for &file in &files {
            let gs_file = format!("{corpora_dir}/{year}/STS.gs.{file}.txt");
            let input_file = format!("{corpora_dir}/{year}/STS.input.{file}.txt");
            println!("{gs_file}\t{input_file}");

            let (gold_scores, sentences) = load_sts_data(&gs_file, &input_file)?;
            let corr = evaluate(&sif, &gold_scores, &sentences)?;
            println!("{file}\t{corr}");
        }
        println!();
    }

    Ok(())
}

fn load_sts_data(
    gs_path: &str,
    input_path: &str,
) -> Result<(Vec<Float>, Vec<String>), Box<dyn Error>> {
    let gs_lines: Vec<String> = BufReader::new(File::open(gs_path)?)
        .lines()
        .map(|l| l.unwrap())
        .collect();
    let input_lines: Vec<String> = BufReader::new(File::open(input_path)?)
        .lines()
        .map(|l| l.unwrap())
        .collect();
    assert_eq!(gs_lines.len(), input_lines.len());

    let mut gold_scores = vec![];
    let mut sentences = vec![];
    for (gs_line, input_line) in gs_lines.iter().zip(input_lines.iter()) {
        if gs_line.is_empty() {
            continue;
        }
        gold_scores.push(gs_line.parse::<Float>()?);
        let cols: Vec<_> = input_line.split('\t').collect();
        assert_eq!(cols.len(), 2);
        sentences.push(cols[0].to_string());
        sentences.push(cols[1].to_string());
    }

    Ok((gold_scores, sentences))
}

fn evaluate<W, U>(
    sif: &Sif<W, U>,
    gold_scores: &[Float],
    sentences: &[String],
) -> Result<Float, Box<dyn Error>>
where
    W: WordEmbeddings,
    U: UnigramLanguageModel,
{
    let n_examples = gold_scores.len();
    eprintln!("n_examples = {}", n_examples);

    let sent_embeddings = sif.embeddings(sentences);
    let mut pred_scores = Vec::with_capacity(n_examples);

    for i in 0..n_examples {
        let e1 = &sent_embeddings.row(i * 2);
        let e2 = &sent_embeddings.row(i * 2 + 1);
        let score = util::cosine_similarity(e1, e2).unwrap_or(0.); // ok?
        pred_scores.push(score);
    }

    Ok(pearson_correlation(&pred_scores, &gold_scores))
}

fn pearson_correlation(s1: &[Float], s2: &[Float]) -> Float {
    assert_eq!(s1.len(), s2.len());
    let concat = [s1, s2].concat();
    let scores = Array2::from_shape_vec((2, s1.len()), concat).unwrap();
    let corr = scores.pearson_correlation().unwrap();
    corr[[0, 1]]
}
