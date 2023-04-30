use crate::Embedding;
use crate::WordEmbeddings;

pub struct Sif<'a, WE> {
    word_embeddings: WE,
    sent_embeddings: Vec<Embedding<'a>>,
}

impl<'a, WE> Sif<'a, WE>
where
    WE: WordEmbeddings,
{
    pub fn new(word_embeddings: WE) {}

    pub fn add<I, S>(&mut self, sentence: I)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for word in sentence {
            let word = word.as_ref();
            if let Some(wv) = self.word_embeddings.lookup(word) {}
        }
    }
}
