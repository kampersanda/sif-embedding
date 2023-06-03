# sif-embedding/examples/embeddings

This directory provides an easy tool to perform sentence embedding using sif-embedding.

## Preparation

sif-embedding, or the SIF algorithm, requires (pre-trained) word embeddings and unigram probabilities.
You first need to prepare them.

For word embeddings, [finalfusion-tools](../finalfusion-tools) provides instructions to download and compile well-known pre-trained word embeddings such as GloVe or fastText.
For unigram probabilities, [wordfreq-tools](../wordfreq-tools) provides instructions to download and compile unigram probabilities from data provided in wordfreq.

## Sentence embedding

Sentence embeddings can be performed for input lines using the models prepared in the above steps.

```
$ echo "hello i am\ngood morning" | cargo run --release -- -f path/to/glove.42B.300d.fifu -u path/to/large_en.unigram
0.0037920314 -0.018138476 0.010073537 ... -0.002471894
-0.0029124343 0.013930968 -0.0077368026 ... 0.001898489
```
