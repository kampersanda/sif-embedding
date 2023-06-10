# Tutorial on sif-embedding

This directory provides a tutorial for getting started with sif-embedding.

This tutorial focuses on how to prepare input models and providing a simple example code of sentence embeddings.
See the [documentation](https://docs.rs/sif-embedding/) for the specifications in `Cargo.toml`.

## Preparation

sif-embedding, or the SIF algorithm, requires the following two components as input:

1. Word embeddings
2. Unigram language models

Here, we describe how to use [finalfusion](https://docs.rs/finalfusion/) and [wordfreq](https://docs.rs/wordfreq/latest/wordfreq/) for these components, respectively.
You can prepare the pre-trained models as follows.

### Word embeddings

[finalfusion](https://docs.rs/finalfusion/) is a library to handle different types of word embeddings such as Glove and fastText.
[finalfusion-tools](../../finalfusion-tools) provides instructions to download and compile those pre-trained word embeddings.

### Unigram language models

[wordfreq](https://docs.rs/wordfreq/latest/wordfreq/) is a library to look up the frequencies of words.
[wordfreq-model](https://docs.rs/wordfreq-model/) allows you to load pre-compiled models in many languages trained from various resources.
See the [documentation](https://docs.rs/wordfreq-model/) for getting started.

## Sentence embedding

Assuming that you have prepared word embedding model `glove.42B.300d.fifu` and specified `features = ["large-en"]`  in wordfreq-model,
sentence embeddings can be performed for input lines as follows:

```
$ echo "hello i am\ngood morning" | cargo run --release -- -f path/to/glove.42B.300d.fifu
0.0037920314 -0.018138476 0.010073537 ... -0.002471894
-0.0029124343 0.013930968 -0.0077368026 ... 0.001898489
```

[src/main.rs](./src/main.rs) will be a good example to understand how to handle the models and compute sentence embeddings.

## Tips

If you are having problems compiling this library due to the backend,
[my tips](https://github.com/kampersanda/sif-embedding/wiki/Trouble-shooting) may help.
