# Tutorial on sif-embedding

This directory provides a tutorial for getting started with sif-embedding.

## Preparation

sif-embedding, or the SIF algorithm, requires pre-trained word embeddings and unigram language models.
First, you need to prepare them.
This repository provides useful tools for this purpose.

### Word embeddings

sif-embedding employs [finalfusion](https://docs.rs/finalfusion/) to handle different types of word embeddings, such as Glove and fastText.
[`finalfusion-tools`](../../finalfusion-tools) provides instructions to download and compile those pre-trained word embeddings.

### Unigram language models

Pre-compiled unigram language models are available in [the release page](https://github.com/kampersanda/sif-embedding/releases/tag/v0.3.1).
(The models are distributed on the release page of a specific version, but are valid for later versions as well.)
Download an appropriate model for your language, as follows:

```
$ wget https://github.com/kampersanda/sif-embedding/releases/download/v0.3.1/large_en.unigram
```

Or, you can download and compile unigram language models manually, following the instructions in [wordfreq-tools](../../wordfreq-tools).

## Sentence embedding

Assume that you have prepared word embedding model `glove.42B.300d.fifu` and unigram language model `large_en.unigram` through the above steps.
Sentence embeddings can be performed for input lines as follows:

```
$ echo "hello i am\ngood morning" | cargo run --release -- -f path/to/glove.42B.300d.fifu -u path/to/large_en.unigram
0.0037920314 -0.018138476 0.010073537 ... -0.002471894
-0.0029124343 0.013930968 -0.0077368026 ... 0.001898489
```

[The source code](./src/main.rs) will be a good example to understand how to handle the models and compute sentence embeddings.

## Tips

If you are having problems compiling this library due to the backend,
[my tips](https://github.com/kampersanda/sif-embedding/wiki/Trouble-shooting) may help.
