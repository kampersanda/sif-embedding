use std::fs::File;
use std::io::BufReader;

use finalfusion::prelude::Embeddings;
use finalfusion::prelude::ReadText;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut reader = BufReader::new(File::open(&args[1]).unwrap());
    let embeddings = Embeddings::read_text(&mut reader).unwrap();

    // Look up an embedding.
    let embedding = embeddings.embedding("apple");

    // Print the embedding.
    println!("{:?}", embedding);
}
