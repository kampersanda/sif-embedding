//! WordProbabilities implementations for [`wordfreq::WordFreq`].
//! This module is available if the `wordfreq` feature is enabled.
use crate::Float;
use crate::WordProbabilities;

use wordfreq::WordFreq;

impl WordProbabilities for WordFreq {
    fn probability(&self, word: &str) -> Float {
        self.word_frequency(word)
    }

    fn n_words(&self) -> usize {
        self.word_frequency_map().len()
    }

    fn entries(&self) -> Box<dyn Iterator<Item = (String, Float)> + '_> {
        Box::new(
            self.word_frequency_map()
                .iter()
                .map(|(k, v)| (k.to_owned(), *v)),
        )
    }
}
