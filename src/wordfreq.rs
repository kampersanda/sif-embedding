//! UnigramLanguageModel implementations for [`wordfreq::WordFreq`].
use crate::Float;
use crate::UnigramLanguageModel;

use wordfreq::WordFreq;

impl UnigramLanguageModel for WordFreq {
    fn probability(&self, word: &str) -> Float {
        self.word_frequency(word)
    }
}
