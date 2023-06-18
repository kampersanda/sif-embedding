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
    input_text: PathBuf,

    #[arg(short = 'o', long)]
    output_fifu: PathBuf,

    #[arg(short = 'd', long)]
    with_dims: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let mut reader = BufReader::new(File::open(args.input_text)?);
    let embeddings = if args.with_dims {
        Embeddings::read_text_dims(&mut reader)?
    } else {
        Embeddings::read_text(&mut reader)?
    };

    let mut writer = BufWriter::new(File::create(args.output_fifu)?);
    embeddings.write_embeddings(&mut writer)?;

    Ok(())
}
