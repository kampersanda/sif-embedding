use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

use clap::Parser;
use finalfusion::io::WriteEmbeddings;
use finalfusion::prelude::*;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'i', long)]
    input_fasttext: PathBuf,

    #[arg(short = 'o', long)]
    output_fifu: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let mut reader = BufReader::new(File::open(args.input_fasttext)?);
    let embeddings = Embeddings::read_fasttext(&mut reader)?;

    let mut writer = BufWriter::new(File::create(args.output_fifu)?);
    embeddings.write_embeddings(&mut writer)?;

    Ok(())
}
