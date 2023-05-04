use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use clap::Parser;

use sif_embedding::util;
use sif_embedding::{Sif, WordEmbeddings};

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
    let mut scores = vec![];

    let lines = std::io::stdin().lock().lines();
    for line in lines {
        let line = line?;
        let cols: Vec<_> = line.split('\t').collect();
        sentences.push(cols[0].to_string());
        sentences.push(cols[1].to_string());
        scores.push(cols[2].parse::<f32>()?);
    }
    println!("sentences.len() = {}", sentences.len());

    let (sent_embeddings, _) = Sif::new(word_embeddings, &word_weights).embeddings(&sentences);
    for i in 0..10 {
        let sim =
            util::cosine_similarity(&sent_embeddings.row(i * 2), &sent_embeddings.row(i * 2 + 1))
                .unwrap_or(0.);
        println!("{}: {:.3} vs {}", i, sim, scores[i]);
    }

    Ok(())
}
