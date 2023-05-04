use std::error::Error;
use std::fs::File;
use std::io::BufReader;
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

    println!("we.len() = {}", word_embeddings.len());
    println!("we.embedding_size() = {}", word_embeddings.embedding_size());
    println!("ww.len() = {}", word_weights.len());

    let (sent_embeddings, _) = Sif::new(word_embeddings, &word_weights)
        .embeddings(&["it 's a charming and often affecting journey ."]);

    println!("sent_embeddings = {:?}", sent_embeddings);

    Ok(())
}
