use std::io::BufRead;

use anyhow::Result;
use hashbrown::hash_map::Entry;
use hashbrown::HashMap;
use ndarray::Array2;

use super::WordEmbeddings;
use crate::Float;

impl WordEmbeddings {
    pub fn from_text<R: BufRead>(rdr: R) -> Result<Self> {
        let mut embeddings = vec![];
        let mut word2idx = HashMap::new();
        let mut embedding_size = None;

        for (i, line) in rdr.lines().enumerate() {
            let line = line?;
            let cols: Vec<_> = line.split_ascii_whitespace().collect();

            let word = cols[0].to_owned();
            let raw_embedding = &cols[1..];

            if let Some(es) = embedding_size {
                if es != raw_embedding.len() {
                    eprintln!(
                        "Line {i}: embedding_size should be {es}, but got {}. Skipped.",
                        raw_embedding.len()
                    );
                    continue;
                }
            } else {
                embedding_size = Some(raw_embedding.len());
            }

            let idx = word2idx.len();
            match word2idx.entry(word) {
                Entry::Occupied(e) => {
                    eprintln!(
                        "Line {i}: word {} is already registered at line {}. Skipped.",
                        e.key(),
                        e.get()
                    );
                    continue;
                }
                Entry::Vacant(e) => {
                    e.insert(idx);
                }
            }
            embeddings.extend(raw_embedding.iter().map(|&v| v.parse::<Float>().unwrap()));
        }

        let embedding_size = embedding_size.unwrap();
        let embeddings =
            Array2::from_shape_vec((word2idx.len(), embedding_size), embeddings).unwrap();

        Ok(Self {
            embeddings,
            word2idx,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn test_from_text() {
        let text = "A 0.0 1.0 2.0\nBB -3.0 -4.0 -5.0\nCCC 6.0 -7.0 8.0\nDDDD -9.0 10.0 -11.0\n";
        let we = WordEmbeddings::from_text(text.as_bytes()).unwrap();

        assert_eq!(we.len(), 4);
        assert_eq!(we.embedding_size(), 3);

        assert_relative_eq!(we.lookup("A").unwrap(), arr1(&[0.0, 1.0, 2.0]));
        assert_relative_eq!(we.lookup("BB").unwrap(), arr1(&[-3.0, -4.0, -5.0]));
        assert_relative_eq!(we.lookup("CCC").unwrap(), arr1(&[6.0, -7.0, 8.0]));
        assert_relative_eq!(we.lookup("DDDD").unwrap(), arr1(&[-9.0, 10.0, -11.0]));
        assert_eq!(we.lookup("EEEEE"), None);
    }
}
