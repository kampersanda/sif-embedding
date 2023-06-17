# sif-embedding/finalfusion-tools

sif-embedding employs [finalfusion](https://docs.rs/finalfusion/) to handle different types of word embeddings, such as GloVe and fastText.
This directory provides tools to convert word embeddings to finalfusion format.

We show a tutorial to convert GloVe and fastText models to finalfusion format below.

## GloVe models

First download a pre-trained GloVe model from [the project page](https://nlp.stanford.edu/projects/glove/).

```shell
$ wget https://nlp.stanford.edu/data/glove.42B.300d.zip
$ unzip glove.42B.300d.zip
```

GloVe models are in text format *without* header.

Then run `compile_text` to convert the model to finalfusion format.

```shell
$ cargo run --release --bin compile_text -- -i glove.42B.300d.txt -o glove.42B.300d.fifu
```

### Removing duplicate words

`compile_text` requires the input model to have no duplicate words.
If a model has duplicate words, you can remove them by `unique_text`.
(e.g., `glove.840B.300d.txt` seems to have duplicate words.)

```shell
$ cargo run --release --bin unique_text -- -i glove.840B.300d.txt -o glove.840B.300d.unique.txt
```

## fastText models (text)

First download a pre-trained fastText model in `.vec` format from [the project page](https://fasttext.cc/).

```shell
$ wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
$ gunzip cc.en.300.vec.gz
```

Those `.vec` models are in text format *with* header.

Then run `compile_text` to convert the model to finalfusion format (with `-d` option).

```shell
$ cargo run --release --bin compile_text -- -i cc.en.300.vec -o cc.en.300.vec.fifu -d
```

## fastText models (binary)

First download a pre-trained fastText model in `.bin` format from [the project page](https://fasttext.cc/).

```shell
$ wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
$ gunzip cc.en.300.bin.gz
```

Then run `compile_fasttext` to convert the model to finalfusion format.

```shell
$ cargo run --release --bin compile_fasttext -- -i cc.en.300.bin -o cc.en.300.bin.fifu
```

