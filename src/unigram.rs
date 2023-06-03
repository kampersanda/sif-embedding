//! Unigram language models.
use std::collections::BTreeMap;
use std::io::{Read, Write};

use anyhow::{anyhow, Result};
use crawdad::Trie;

use crate::Float;

const MODEL_MAGIC: &[u8] = b"sif-embedding::UnigramLM 0.3.1\n";

/// Unigram language model.
///
/// # Examples
///
/// ```
/// use approx::assert_relative_eq;
/// use sif_embedding::UnigramLM;
///
/// let word_weights = [("las", 10.), ("vegas", 30.)];
/// let unigram_lm = UnigramLM::new(word_weights);
///
/// // Querying
/// assert_relative_eq!(unigram_lm.probability("las"), 0.25);
/// assert_relative_eq!(unigram_lm.probability("vegas"), 0.75);
/// assert_relative_eq!(unigram_lm.probability("Las"), 0.00);
///
/// // Serializing the model.
/// let mut model = vec![];
/// let size = unigram_lm.write(&mut model).unwrap();
/// assert_eq!(size, model.len());
///
/// // Deserializing the model.
/// let other = UnigramLM::read(&model[..]).unwrap();
/// assert_relative_eq!(other.probability("las"), 0.25);
/// assert_relative_eq!(other.probability("vegas"), 0.75);
/// assert_relative_eq!(other.probability("Las"), 0.00);
/// ```
pub struct UnigramLM {
    trie: Option<Trie>,
}

impl UnigramLM {
    /// Creates the language model.
    ///
    /// # Arguments
    ///
    /// - `word_weights`: Pairs of words and their frequencies (or probabilities) from a corpus.
    ///
    /// # Notes
    ///
    /// If the input contains duplicate words, the last occurrence is used.
    pub fn new<I, W>(word_weights: I) -> Self
    where
        I: IntoIterator<Item = (W, Float)>,
        W: AsRef<str>,
    {
        let mut word2probs: BTreeMap<_, _> = word_weights
            .into_iter()
            .map(|(word, weight)| (word.as_ref().to_string(), weight))
            .collect();

        if word2probs.is_empty() {
            return Self { trie: None };
        }

        let sum_weight = word2probs.values().fold(0., |acc, w| acc + w);
        word2probs.values_mut().for_each(|w| *w /= sum_weight);

        let trie =
            Trie::from_records(word2probs.into_iter().map(|(w, p)| (w, p.to_bits()))).unwrap();
        Self { trie: Some(trie) }
    }

    /// Returns the probability for an input word.
    pub fn probability<W>(&self, word: W) -> Float
    where
        W: AsRef<str>,
    {
        self.trie.as_ref().map_or(0., |trie| {
            trie.exact_match(word.as_ref().chars())
                .map(Float::from_bits)
                .unwrap_or(0.)
        })
    }

    /// Exports the model data, returning the number of bytes written.
    pub fn write<W>(&self, mut wtr: W) -> Result<usize>
    where
        W: Write,
    {
        wtr.write_all(MODEL_MAGIC)?;
        let bytes = self.trie.as_ref().map_or_else(
            || vec![0],
            |trie| {
                let mut dest = Vec::with_capacity(1 + trie.io_bytes());
                dest.push(1);
                dest.extend(trie.serialize_to_vec());
                dest
            },
        );
        let n_bytes = u32::try_from(bytes.len())?;
        wtr.write_all(&n_bytes.to_le_bytes())?;
        wtr.write_all(&bytes)?;
        Ok(MODEL_MAGIC.len() + std::mem::size_of::<u32>() + bytes.len())
    }

    /// Read the model data.
    pub fn read<R>(mut rdr: R) -> Result<Self>
    where
        R: Read,
    {
        let mut magic = [0; MODEL_MAGIC.len()];
        rdr.read_exact(&mut magic)?;
        if magic != MODEL_MAGIC {
            return Err(anyhow!("The magic number of the input model mismatches."));
        }
        let mut n_bytes = [0; std::mem::size_of::<u32>()];
        rdr.read_exact(&mut n_bytes)?;
        let n_bytes = u32::from_le_bytes(n_bytes) as usize;
        let mut source = vec![0; n_bytes];
        rdr.read_exact(&mut source)?;
        if source[0] == 1 {
            let (trie, _) = Trie::deserialize_from_slice(&source[1..]);
            Ok(Self { trie: Some(trie) })
        } else if source[0] == 0 {
            Ok(Self { trie: None })
        } else {
            Err(anyhow!("Invalid UnigramLM model format."))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;

    #[test]
    fn test_empty() {
        let word_weights = Vec::<(&str, Float)>::new();
        let unigram_lm = UnigramLM::new(word_weights);

        assert_relative_eq!(unigram_lm.probability("las"), 0.00);
        assert_relative_eq!(unigram_lm.probability("vegas"), 0.00);
        assert_relative_eq!(unigram_lm.probability("Las"), 0.00);

        let mut model = vec![];
        let size = unigram_lm.write(&mut model).unwrap();
        assert_eq!(size, model.len());

        let other = UnigramLM::read(&model[..]).unwrap();
        assert_relative_eq!(other.probability("las"), 0.00);
        assert_relative_eq!(other.probability("vegas"), 0.00);
        assert_relative_eq!(other.probability("Las"), 0.00);
    }

    #[test]
    fn test_duplicate() {
        let word_weights = [("las", 10.), ("vegas", 30.), ("las", 20.)];
        let unigram_lm = UnigramLM::new(word_weights);

        assert_relative_eq!(unigram_lm.probability("las"), 0.40);
        assert_relative_eq!(unigram_lm.probability("vegas"), 0.60);
        assert_relative_eq!(unigram_lm.probability("Las"), 0.00);
    }
}
