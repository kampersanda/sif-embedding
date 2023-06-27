# Tutorial on sif-embedding

This directory provides an example crate and aims to at least get you started with sif-embedding by looking here.
This tutorial assumes that you will compute embeddings for English sentences using several public resources.

Note that the specifications of this and related libraries are not described here.

## Preparation

sif-embedding requires the following two components as input:

1. Word embeddings
2. Word probabilities (or unigram language models)

We describe how to use [finalfusion](https://docs.rs/finalfusion/) and [wordfreq](https://docs.rs/wordfreq/latest/wordfreq/) for these components, respectively.

### Word embeddings

[finalfusion](https://docs.rs/finalfusion/) is a library to handle different types of word embeddings such as GloVe and fastText.
[finalfusion-tools](../../finalfusion-tools) provides instructions to download and compile those pre-trained word embeddings.
Follow the instructions and prepare a model of pre-trained word embeddings in the finalfusion format.

Here, we assume that you have prepared a GloVe model `glove.42B.300d.fifu` in the current directory.
Specify the dependency in `Cargo.toml` as follows:

```toml
[dependencies]
finalfusion = "0.17.2"
```

Then, the model can be loaded with few lines of code as follows:

```rust
use finalfusion::prelude::*;

let mut reader = BufReader::new(File::open("glove.42B.300d.fifu")?);
let word_embeddings = Embeddings::<VocabWrap, StorageWrap>::mmap_embeddings(&mut reader)?;
```

### Word probabilities

[wordfreq](https://docs.rs/wordfreq/latest/wordfreq/) is a library to look up the frequencies of words.
[wordfreq-model](https://docs.rs/wordfreq-model/) allows you to load pre-compiled models in many languages trained from various resources.

If you want to use the English model, specify the dependency in `Cargo.toml` as follows:

```toml
[dependencies]
wordfreq-model = { version = "0.2.3", features = ["large-en"] }
```

Then, the model can be loaded with few lines of code as follows:

```rust
use wordfreq_model::ModelKind;

let word_probs = wordfreq_model::load_wordfreq(ModelKind::LargeEn)?;
```

## Sentence embedding

Assuming that you have prepared word embedding model `glove.42B.300d.fifu` and specified `features = ["large-en"]`  in wordfreq-model,
sentence embeddings can be performed for input lines as follows:

```shell
$ cargo run --release --features openblas -- -f glove.42B.300d.fifu
[[0.0015605423, -0.009350954, 0.0045850407, -0.0023774623, -0.005104989, ..., -0.0021531796, -0.0049471697, -0.0047046337, 0.00046235695, 0.007420418],
 [0.00095811766, 0.0077166725, -0.0046481024, 0.0014976687, 0.0034232885, ..., -0.0023950897, -0.0035641948, -0.00756339, -0.0003676638, -0.0021527726],
 [-0.0028154051, -0.00013797358, 0.0011601038, 0.0005516759, 0.000922475, ..., 0.0052407943, 0.009591073, 0.014395955, -1.0702759e-5, -0.004908829]], shape=[3, 300], strides=[300, 1], layout=Cc (0x5), const ndim=2
[[0.00035341084, -0.0054921675, 0.0015013912, -0.0015475352, -0.0036347732, ..., -0.006240675, -0.0066263573, -0.004742044, 0.008010669, 0.00335777],
 [-9.8407036e-5, 0.0073085483, -0.0054234676, 0.0013588136, 0.0027614273, ..., -0.0064221052, -0.0055891275, -0.0068861144, 0.007388141, -0.00382212]], shape=[2, 300], strides=[300, 1], layout=Cc (0x5), const ndim=2
```

[src/main.rs](./src/main.rs) will be a good example to understand how to handle the models and compute sentence embeddings.

## Tips

If you are having problems compiling this library due to the backend,
[my tips](https://github.com/kampersanda/sif-embedding/wiki/Trouble-shooting) may help.
