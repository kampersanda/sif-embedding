# Tutorial on sif-embedding

This directory provides a tutorial for getting started with sif-embedding.

## Preparation

sif-embedding, or the SIF algorithm, requires pre-trained word embeddings and unigram language models.
You first need to prepare them.

### Word embeddings

sif-embedding employs [finalfusion](https://docs.rs/finalfusion/) to handle different types of word embeddings, such as Glove and fastText.
[`finalfusion-tools`](../../finalfusion-tools) provides instructions to download and compile those pre-trained word embeddings.

### Unigram language models

[`wordfreq-tools`](../../wordfreq-tools) provides instructions to download and compile unigram language models from data provided in [wordfreq](https://github.com/rspeer/wordfreq).

## Sentence embedding

Sentence embeddings can be performed for input lines using the models prepared in the above steps.

```
$ echo "hello i am\ngood morning" | cargo run --release -- -f path/to/glove.42B.300d.fifu -u path/to/large_en.unigram
0.0037920314 -0.018138476 0.010073537 ... -0.002471894
-0.0029124343 0.013930968 -0.0077368026 ... 0.001898489
```
