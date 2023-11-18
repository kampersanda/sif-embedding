# Tutorial on sif-embedding

This directory provides an example crate and aims to at least get you started with sif-embedding by looking here.
In this tutorial, you will understand this example crate and be able to use sif-embedding for your own project.
This tutorial assumes that you will compute embeddings for English sentences using several public resources.

Note that the specifications of this and related libraries are not described here.
See the [API documentation](https://docs.rs/sif-embedding/) for them.

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

## Basic usage of sif-embedding

### Plugging in external libraries

By enabling the `finalfusion` and `wordfreq` features, the above external libraries can be plugged into sif-embedding.

```toml
[dependencies.sif-embedding]
version = "0.6"
features = ["finalfusion", "wordfreq"]
default-features = false
```

### Specifying a backend for linear algebra

You also need to specify a backend option for linear algebra from one of [openblas-src](https://github.com/blas-lapack-rs/openblas-src), [netlib-src](https://github.com/blas-lapack-rs/netlib-src), and [intel-mkl-src](https://github.com/rust-math/intel-mkl-src).

If you want to use openblas-src with static linking, specify the dependencies in `Cargo.toml` as follows:

```toml
[dependencies.sif-embedding]
version = "0.6"
features = ["finalfusion", "wordfreq", "openblas-static"]
default-features = false

[dependencies.openblas-src]
version = "0.10.4"
features = ["cblas", "static"]
default-features = false
```

Then, declare it to be recognized at the root of your crate.

```rust
extern crate openblas_src as _src;
```

### Computing sentence embeddings

You can compute sentence embeddings as follows:

```rust
use sif_embedding::SentenceEmbedder;
use sif_embedding::Sif;

let sentences = vec![
    "This is a sentence.",
    "This is another sentence.",
    "This is a third sentence.",
];
let model = Sif::new(&word_embeddings, &word_probs);
let model = model.fit(&sentences)?;
let sent_embeddings = model.embeddings(sentences)?;
println!("{:?}", sent_embeddings);
```

`sent_embeddings` is a 2D-array of shape `(#sentences, #dimensions)`.

The returned `model` maintains the parameters estimated from the input sentences.
You can use it to compute embeddings for new sentences as follows:

```rust
let new_sentences = vec!["This is a new sentence.", "This is another new sentence."];
let sent_embeddings = model.embeddings(new_sentences)?;
println!("{:?}", sent_embeddings);
```

The fitted model will be useful for handling additional fewer sentences.

## Example

This crate is an example of reproducing the above steps.
You can run it with the following command:

```shell
$ cargo run --release --features openblas-static -- -f glove.42B.300d.fifu
```

## Tips

If you are having problems compiling this library due to the backend,
[my tips](https://github.com/kampersanda/sif-embedding/wiki/Trouble-shooting) may help.
