use std::fs::File;
use std::io::BufReader;

use finalfusion::prelude::Embeddings;
use finalfusion::prelude::ReadFastText;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut reader = BufReader::new(File::open(&args[1]).unwrap());

    // Read a file in .bin.
    let embeddings = Embeddings::read_fasttext(&mut reader).unwrap();

    // Look up an embedding.
    let embedding = embeddings.embedding("zwei");

    // Print the embedding.
    println!("{:?}", embedding);
}
