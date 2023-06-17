use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::BufRead;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'i', long)]
    input_text: PathBuf,

    #[arg(short = 'o', long)]
    output_text: PathBuf,

    #[arg(short = 'd', long)]
    with_dims: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let mut word2idx = HashMap::new();
    let reader = BufReader::new(File::open(args.input_text)?);
    let mut writer = BufWriter::new(File::create(args.output_text)?);

    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        if args.with_dims && idx == 0 {
            writer.write_all(line.as_bytes())?;
            writer.write_all(b"\n")?;
            continue;
        }
        let cols: Vec<_> = line.split_ascii_whitespace().collect();
        let word = cols[0].to_owned();
        match word2idx.entry(word) {
            Entry::Occupied(e) => {
                eprintln!(
                    "Line {}: word {} is already registered at line {}. Skipped.",
                    idx + 1,
                    e.key(),
                    e.get()
                );
            }
            Entry::Vacant(e) => {
                e.insert(idx);
                writer.write_all(line.as_bytes())?;
                writer.write_all(b"\n")?;
            }
        }
    }

    Ok(())
}
