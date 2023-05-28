use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::error::Error;
use std::io::BufRead;

fn main() -> Result<(), Box<dyn Error>> {
    let mut word2idx = HashMap::new();

    let lines = std::io::stdin().lock().lines();
    for (idx, line) in lines.enumerate() {
        let line = line?;
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

    Ok(())
}
