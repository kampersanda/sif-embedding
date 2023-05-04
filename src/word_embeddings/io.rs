use std::io::BufRead;

use anyhow::{anyhow, Ok, Result};
use hashbrown::HashMap;
use ndarray::Array2;

use super::WordEmbeddings;

impl WordEmbeddings {
    pub fn from_text<R: BufRead>(rdr: R) -> Result<Self> {
        let mut embeddings = vec![];
        let mut word2idx = HashMap::new();
        let mut embedding_size = None;

        for line in rdr.lines() {
            let line = line.unwrap();
            let cols: Vec<&str> = line.split_ascii_whitespace().collect();

            let idx = word2idx.len();
            let word = cols[0].to_owned();
            word2idx.try_insert(word, idx).unwrap();

            let raw_embedding = &cols[1..];
            if let Some(es) = embedding_size {
                if es != raw_embedding.len() {
                    return Err(anyhow!(""));
                }
            } else {
                embedding_size = Some(raw_embedding.len());
            }
            embeddings.extend(raw_embedding.iter().map(|&v| v.parse::<f32>().unwrap()));
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
    use ndarray::arr2;

    #[test]
    fn test_from_text() {
        let text = "A 0.0 1.0 2.0\nB -3.0 -4.0 -5.0\nC 6.0 -7.0 8.0\n";
        let WordEmbeddings {
            embeddings,
            word2idx,
        } = WordEmbeddings::from_text(text.as_bytes()).unwrap();

        assert_eq!(
            word2idx,
            [
                ("A".to_string(), 0),
                ("B".to_string(), 1),
                ("C".to_string(), 2)
            ]
            .into_iter()
            .collect::<HashMap<String, usize>>()
        );

        assert_relative_eq!(
            embeddings,
            arr2(&[[0.0, 1.0, 2.0], [-3.0, -4.0, -5.0], [6.0, -7.0, 8.0]])
        );
    }
}
