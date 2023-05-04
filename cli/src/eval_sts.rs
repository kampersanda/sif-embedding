use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use clap::Parser;
use ndarray::Array2;
use ndarray_stats::CorrelationExt;

use sif_embedding::util;
use sif_embedding::{Float, Sif, WordEmbeddings};

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'e', long)]
    word_embedding: PathBuf,

    #[arg(short = 'w', long)]
    word_weights: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let word_embeddings = {
        let reader = BufReader::new(File::open(&args.word_embedding)?);
        WordEmbeddings::from_text(reader)?
    };
    let word_weights = {
        let reader = BufReader::new(File::open(&args.word_weights)?);
        util::word_weights_from_text(reader)?
    };

    println!("word_embeddings.len() = {}", word_embeddings.len());
    println!(
        "word_embeddings.embedding_size() = {}",
        word_embeddings.embedding_size()
    );
    println!("word_weights.len() = {}", word_weights.len());

    let mut sentences = vec![];
    let mut gold_scores = vec![];

    let lines = std::io::stdin().lock().lines();
    for line in lines {
        let line = line?;
        let cols: Vec<_> = line.split('\t').collect();
        sentences.push(cols[0].to_string());
        sentences.push(cols[1].to_string());
        gold_scores.push(cols[2].parse::<Float>()?);
    }
    let n_examples = gold_scores.len();

    println!("n_examples = {}", n_examples);
    println!("sentences.len() = {}", sentences.len());

    let (sent_embeddings, _) = Sif::new(word_embeddings, &word_weights).embeddings(&sentences);
    let mut pred_scores = Vec::with_capacity(sentences.len());

    for i in 0..n_examples {
        let e1 = &sent_embeddings.row(i * 2);
        let e2 = &sent_embeddings.row(i * 2 + 1);
        let score = util::cosine_similarity(e1, e2).unwrap();
        pred_scores.push(score);
    }

    let scores = array2_from_two_slices(&pred_scores, &gold_scores).unwrap();
    let corr = scores.pearson_correlation().unwrap();

    println!("{:?}", corr[[0, 1]]);

    Ok(())
}

fn array2_from_two_slices<T: Copy>(s1: &[T], s2: &[T]) -> Option<Array2<T>> {
    if s1.len() == s2.len() {
        let concat = [s1, s2].concat();
        Array2::from_shape_vec((2, s1.len()), concat).ok()
    } else {
        None
    }
}
