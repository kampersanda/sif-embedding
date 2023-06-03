extern crate openblas_src;

use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

use clap::Parser;

use sif_embedding::util;
use sif_embedding::UnigramLM;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'i', long)]
    input_weights: PathBuf,

    #[arg(short = 'o', long)]
    output_model: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let unigram_lm = {
        let reader = BufReader::new(File::open(&args.input_weights)?);
        let word_weights = util::word_weights_from_text(reader)?;
        UnigramLM::new(word_weights)
    };

    {
        let mut writter = BufWriter::new(File::create(&args.output_model)?);
        unigram_lm.write(&mut writter)?;
    }

    Ok(())
}
