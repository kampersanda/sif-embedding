use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use polars::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

const CATEGORIES: &[&str] = &[
    "dokujo-tsushin",
    "it-life-hack",
    "kaden-channel",
    "livedoor-homme",
    "movie-enter",
    "peachy",
    "smax",
    "sports-watch",
    "topic-news",
];

pub fn load_livedoor_data(data_dir: &str) -> Result<DataFrame, Box<dyn Error>> {
    let mut files = vec![];
    for (label, &cate) in CATEGORIES.iter().enumerate() {
        for filepath in glob::glob(&format!("{data_dir}/{cate}/{cate}-*.txt"))? {
            let filepath = filepath?;
            files.push((label, filepath));
        }
    }

    let mut rng = ChaCha8Rng::seed_from_u64(334);
    files.shuffle(&mut rng);

    let mut labels = vec![];
    let mut sentences = vec![];
    for (label, filepath) in files {
        let sentence = read_sentence(&filepath)?;
        labels.push(label as u32);
        sentences.push(sentence);
    }
    let labels = Series::new("label", labels);
    let sentences = Series::new("sentence", sentences);
    Ok(DataFrame::new(vec![labels, sentences])?)
}

fn read_sentence<P: AsRef<Path>>(filepath: P) -> Result<String, Box<dyn Error>> {
    let file = File::open(filepath)?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();
    // Skip the first three lines for the header.
    Ok(lines[3..].join("\n"))
}
