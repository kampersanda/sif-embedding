use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let reader = BufReader::new(File::open(&args[1]).unwrap());
    let mut word2idx = HashMap::new();

    for (idx, line) in reader.lines().enumerate() {
        let line = line.unwrap();
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
                println!("{line}");
            }
        }
    }
}
