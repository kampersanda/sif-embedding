use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

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

pub fn load_livedoor_data(data_dir: &str) -> Result<(Vec<String>, Vec<usize>), Box<dyn Error>> {
    let mut sentences = vec![];
    let mut labels = vec![];
    for (label, &cate) in CATEGORIES.iter().enumerate() {
        for filepath in glob::glob(&format!("{data_dir}/{cate}/{cate}-*.txt"))? {
            let filepath = filepath?;
            let sentence = read_sentence(&filepath)?;
            sentences.push(sentence);
            labels.push(label);
        }
    }
    Ok((sentences, labels))
}

fn read_sentence<P: AsRef<Path>>(filepath: P) -> Result<String, Box<dyn Error>> {
    let file = File::open(filepath)?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();
    // Skip the first three lines for the header.
    Ok(lines[3..].join("\n"))
}
