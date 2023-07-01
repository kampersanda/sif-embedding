use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use rand::seq::SliceRandom;

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

pub struct Record {
    pub sentence: String,
    pub label: usize,
}

pub fn load_livedoor_data(
    data_dir: &str,
    with_shuffle: bool,
) -> Result<Vec<Record>, Box<dyn Error>> {
    let mut records = vec![];
    for (label, &cate) in CATEGORIES.iter().enumerate() {
        for filepath in glob::glob(&format!("{data_dir}/{cate}/{cate}-*.txt"))? {
            let filepath = filepath?;
            let sentence = read_sentence(&filepath)?;
            records.push(Record { sentence, label });
        }
    }
    if with_shuffle {
        records.shuffle(&mut rand::thread_rng());
    }
    Ok(records)
}

fn read_sentence<P: AsRef<Path>>(filepath: P) -> Result<String, Box<dyn Error>> {
    let file = File::open(filepath)?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();
    // Skip the first three lines for the header.
    Ok(lines[3..].join("\n"))
}
