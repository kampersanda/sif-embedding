//! Handlers for word embeddings from pretrained word2vec/[GloVe](https://nlp.stanford.edu/projects/glove/) models.
use std::io::BufRead;

use anyhow::Result;
use hashbrown::hash_map::Entry;
use hashbrown::HashMap;
use ndarray::{self, Array1, Array2, ArrayView1, CowArray, Ix1};

use crate::Float;

/// Handlers for word embeddings from pretrained word2vec/[GloVe](https://nlp.stanford.edu/projects/glove/) models.
#[derive(Debug, Clone)]
pub struct WordEmbeddings {
    word2idx: HashMap<String, usize>,
    embeddings: Array2<Float>,
    avg_embedding: Option<Array1<Float>>, // for OOV
}

impl WordEmbeddings {
    /// Loads a pretrained model in a text format,
    /// where each line has a word and its embedding sparated by the ASCII whitespace.
    ///
    /// ```text
    /// <word> <value1> <value2> ... <valueD>
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use sif_embedding::WordEmbeddings;
    ///
    /// let word_model = "las 0.0 1.0 2.0\nvegas -3.0 -4.0 -5.0\n";
    /// let word_embeddings = WordEmbeddings::from_text(word_model.as_bytes())?;
    ///
    /// assert_eq!(word_embeddings.len(), 2);
    /// assert_eq!(word_embeddings.embedding_size(), 3);
    ///
    /// word_embeddings.lookup("vegas");
    /// // => Some([-3.0, -4.0, -5.0])
    /// # Ok(())
    /// # }
    /// ```
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
            avg_embedding: None,
        })
    }

    /// Builds the average embedding for OOV words, referencing [this page](https://stackoverflow.com/questions/49239941/what-is-unk-in-the-pretrained-glove-vector-files-e-g-glove-6b-50d-txt).
    ///
    /// If enabled, [`Self::lookup()`] will never return [`None`].
    pub fn build_oov_embedding(mut self) -> Self {
        let mut avg_embedding = Array1::zeros(self.embedding_size());
        for embedding in self.embeddings.rows() {
            avg_embedding += &embedding;
        }
        avg_embedding /= self.len() as Float;
        self.avg_embedding = Some(avg_embedding);
        self
    }

    /// Returns the embedding for the input word.
    pub fn lookup(&self, word: &str) -> Option<CowArray<'_, Float, Ix1>> {
        if let Some(&idx) = self.word2idx.get(word) {
            let row = self.embeddings.slice(ndarray::s![idx, ..]);
            return Some(row.into());
        }
        if let Some(avg) = self.avg_embedding.as_ref() {
            return Some(avg.into());
        }
        None
    }

    /// Returns an iterator to enumerate words and their embeddings.
    pub fn iter(&self) -> impl Iterator<Item = (&str, ArrayView1<Float>)> {
        self.word2idx
            .iter()
            .map(|(w, i)| (w.as_str(), self.embeddings.row(*i)))
    }

    /// Returns the number of words.
    pub fn len(&self) -> usize {
        self.embeddings.shape()[0]
    }

    /// Checks if it is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of dimensions for word embeddings.
    pub fn embedding_size(&self) -> usize {
        self.embeddings.shape()[1]
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

    #[test]
    fn test_build_oov_embedding() {
        let text = "A 0.0 1.0 2.0\nBB -3.0 -4.0 -5.0\nCCC 6.0 -7.0 8.0\nDDDD -9.0 10.0 -11.0\n";
        let we = WordEmbeddings::from_text(text.as_bytes())
            .unwrap()
            .build_oov_embedding();

        assert_relative_eq!(we.lookup("A").unwrap(), arr1(&[0.0, 1.0, 2.0]));
        assert_relative_eq!(we.lookup("BB").unwrap(), arr1(&[-3.0, -4.0, -5.0]));
        assert_relative_eq!(we.lookup("CCC").unwrap(), arr1(&[6.0, -7.0, 8.0]));
        assert_relative_eq!(we.lookup("DDDD").unwrap(), arr1(&[-9.0, 10.0, -11.0]));
        assert_relative_eq!(we.lookup("EEEEE").unwrap(), arr1(&[-1.5, 0.0, -1.5]));
    }
}
