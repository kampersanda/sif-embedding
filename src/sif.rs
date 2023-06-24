//! Smooth Inverse Frequency (SIF).
use anyhow::Ok;
use anyhow::{anyhow, Result};
use ndarray::Array1;
use ndarray::Array2;

use crate::util;
use crate::Float;
use crate::UnigramLanguageModel;
use crate::WordEmbeddings;

const N_COMPONENTS: usize = 1;

/// An implementation of *Smooth Inverse Frequency (SIF)* that is a simple but pewerful
/// embedding technique for sentences, described in the paper:
///
/// > Sanjeev Arora, Yingyu Liang, and Tengyu Ma,
/// > [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/forum?id=SyK00v5xx),
/// > ICLR 2017.
///
/// # Examples
///
/// See [the top page](crate).
#[derive(Clone)]
pub struct Sif<'w, 'u, W, U> {
    word_embeddings: &'w W,
    unigram_lm: &'u U,
    separator: char,
    param_a: Float,
    common_components: Option<Array2<Float>>,
}

impl<'w, 'u, W, U> Sif<'w, 'u, W, U>
where
    W: WordEmbeddings,
    U: UnigramLanguageModel,
{
    /// Creates a new instance.
    pub const fn new(word_embeddings: &'w W, unigram_lm: &'u U) -> Self {
        Self {
            word_embeddings,
            unigram_lm,
            separator: ' ',
            param_a: 1e-3,
            common_components: None,
        }
    }

    /// Returns the number of dimensions for sentence embeddings,
    /// which is equivalent to that of the input word embeddings.
    pub fn embedding_size(&self) -> usize {
        self.word_embeddings.embedding_size()
    }

    /// Sets a separator for sentence segmentation (default: ASCII whitespace).
    pub const fn separator(mut self, separator: char) -> Self {
        self.separator = separator;
        self
    }

    /// Sets a SIF-weighting parameter `a` (default: `1e-3`).
    pub fn param_a(mut self, param_a: Float) -> Result<Self> {
        if self.is_fitted() {
            Err(anyhow!("already fitted"))
        } else {
            self.param_a = param_a;
            Ok(self)
        }
    }

    ///
    pub fn is_fitted(&self) -> bool {
        self.common_components.is_some()
    }

    ///
    pub fn fit<S>(mut self, sentences: &[S]) -> Result<Self>
    where
        S: AsRef<str>,
    {
        if sentences.is_empty() {
            return Err(anyhow!("no sentences"));
        }
        let sent_embeddings = self.weighted_embeddings(sentences);
        let (_, common_components) = util::principal_components(&sent_embeddings, N_COMPONENTS);
        self.common_components = Some(common_components);
        Ok(self)
    }

    /// Computes embeddings for input sentences,
    /// returning a 2D-array of shape `(n_sentences, embedding_size)`, where
    ///
    /// - `n_sentences` is the number of input sentences, and
    /// - `embedding_size` is [`Self::embedding_size()`].
    pub fn embeddings<I, S>(&self, sentences: I) -> Result<Array2<Float>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        if !self.is_fitted() {
            return Err(anyhow!("not fitted"));
        }
        let sent_embeddings = self.weighted_embeddings(sentences);
        if sent_embeddings.is_empty() {
            return Ok(sent_embeddings);
        }
        let common_components = self.common_components.as_ref().unwrap();
        let sent_embeddings =
            util::remove_principal_components(&sent_embeddings, common_components, None);
        Ok(sent_embeddings)
    }

    ///
    pub fn fit_embeddings<S>(&mut self, sentences: &[S]) -> Result<Array2<Float>>
    where
        S: AsRef<str>,
    {
        if sentences.is_empty() {
            return Err(anyhow!("no sentences"));
        }
        let sent_embeddings = self.weighted_embeddings(sentences);
        let (_, common_components) = util::principal_components(&sent_embeddings, N_COMPONENTS);
        let sent_embeddings =
            util::remove_principal_components(&sent_embeddings, &common_components, None);
        self.common_components = Some(common_components);
        Ok(sent_embeddings)
    }

    /// Applies SIF-weighting. (Lines 1--3 in Algorithm 1)
    fn weighted_embeddings<I, S>(&self, sentences: I) -> Array2<Float>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut sent_embeddings = vec![];
        let mut n_sentences = 0;
        for sent in sentences {
            let sent = sent.as_ref();
            let mut n_words = 0;
            let mut sent_embedding = Array1::zeros(self.embedding_size());
            for word in sent.split(self.separator) {
                if let Some(word_embedding) = self.word_embeddings.embedding(word) {
                    let weight = self.param_a / (self.param_a + self.unigram_lm.probability(word));
                    sent_embedding += &(word_embedding.to_owned() * weight);
                    n_words += 1;
                }
            }
            if n_words != 0 {
                sent_embedding /= n_words as Float;
            }
            sent_embeddings.extend(sent_embedding.iter());
            n_sentences += 1;
        }
        Array2::from_shape_vec((n_sentences, self.embedding_size()), sent_embeddings).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::{arr1, CowArray, Ix1};

    struct SimpleWordEmbeddings {
        words: Vec<String>,
    }

    impl SimpleWordEmbeddings {
        fn new() -> Self {
            Self {
                words: vec![
                    "A".to_owned(),
                    "BB".to_owned(),
                    "CCC".to_owned(),
                    "DDDD".to_owned(),
                ],
            }
        }
    }

    impl WordEmbeddings for SimpleWordEmbeddings {
        fn embedding(&self, word: &str) -> Option<CowArray<Float, Ix1>> {
            match word {
                "A" => Some(arr1(&[1., 2., 3.]).into()),
                "BB" => Some(arr1(&[4., 5., 6.]).into()),
                "CCC" => Some(arr1(&[7., 8., 9.]).into()),
                "DDDD" => Some(arr1(&[10., 11., 12.]).into()),
                _ => None,
            }
        }

        fn embedding_size(&self) -> usize {
            3
        }

        fn n_words(&self) -> usize {
            4
        }

        fn words(&self) -> Box<dyn Iterator<Item = String> + '_> {
            Box::new(self.words.iter().cloned())
        }
    }

    struct SimpleUnigramLanguageModel {}

    impl UnigramLanguageModel for SimpleUnigramLanguageModel {
        fn probability(&self, word: &str) -> Float {
            match word {
                "A" => 1.,
                "BB" => 2.,
                "CCC" => 3.,
                "DDDD" => 4.,
                _ => 0.,
            }
        }
    }

    #[test]
    fn test_embeddings() {
        let word_embeddings = SimpleWordEmbeddings::new();
        let unigram_lm = SimpleUnigramLanguageModel {};

        let sif = Sif::new(&word_embeddings, &unigram_lm);
        let sif = sif
            .fit(&["A BB CCC DDDD", "BB CCC", "A B C", "Z", ""])
            .unwrap();

        let sent_embeddings = sif
            .embeddings(["A BB CCC DDDD", "BB CCC", "A B C", "Z", ""])
            .unwrap();
        assert_eq!(sent_embeddings.shape(), &[5, 3]);

        let sent_embeddings = sif.embeddings(Vec::<&str>::new()).unwrap();
        assert_eq!(sent_embeddings.shape(), &[0, 3]);

        let sent_embeddings = sif.embeddings(["", ""]).unwrap();
        assert_eq!(sent_embeddings.shape(), &[2, 3]);
    }
}
