extern crate openblas_src;

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use clap::Parser;
use instant_segment::Search;
use instant_segment::Segmenter;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    // #[arg(short = 'f', long)]
    // input_fifu: PathBuf,

    // #[arg(short = 'w', long)]
    // input_weights: PathBuf,

    // #[arg(short = 'c', long)]
    // corpora_dir: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    // let args = Args::parse();

    let unigram_reader = BufReader::new(File::open("data/en-unigrams.txt")?);
    let bigram_reader = BufReader::new(File::open("data/en-bigrams.txt")?);

    let unigram_iter = unigram_reader.lines().map(|line| {
        let line = line.unwrap();
        let mut split = line.split('\t');
        let word = split.next().unwrap();
        let prob = split.next().unwrap();
        (word.into(), prob.parse::<f64>().unwrap())
    });
    let bigram_iter = bigram_reader.lines().map(|line| {
        let line = line.unwrap();
        let mut split = line.split('\t');
        let words = split.next().unwrap();
        let prob = split.next().unwrap();
        let mut word_split = words.split(' ');
        let word1 = word_split.next().unwrap();
        let word2 = word_split.next().unwrap();
        ((word1.into(), word2.into()), prob.parse::<f64>().unwrap())
    });

    let segmenter = Segmenter::from_iters(unigram_iter, bigram_iter);
    let mut search = Search::default();

    let words = segmenter.segment("haven't", &mut search)?;
    println!("{:?}", words.collect::<Vec<&str>>());

    Ok(())
}
