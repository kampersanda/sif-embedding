# sif-embedding/finalfusion-tools

sif-embedding employs [finalfusion](https://docs.rs/finalfusion/) to handle different types of word embeddings, such as Glove and fastText.
This directory provides tools to convert word embeddings to finalfusion format.

## Glove

First download a pre-trained Glove model from [the project page](https://nlp.stanford.edu/projects/glove/).

```shell
$ wget https://nlp.stanford.edu/data/glove.42B.300d.zip
$ unzip glove.42B.300d.zip
```

Then run `compile_glove` to convert the model to finalfusion format.

```shell
$ cargo run --release --bin compile_glove -- -i glove.42B.300d.txt -o glove.42B.300d.fifu
```

### Removing duplicate words

`compile_glove` requires the input model to have no duplicate words.
If a model has duplicate words, you can remove them by `unique_glove`.
(e.g., `glove.840B.300d.txt` seems to have duplicate words.)

```shell
$ cargo run --release --bin unique_glove < glove.840B.300d.txt > glove.840B.300d.unique.txt
```

## fastText

First download a pre-trained fastText model (`.bin`) from [the project page](https://fasttext.cc/).

```shell
$ wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
$ gunzip cc.en.300.bin.gz
```

Then run `compile_fasttext` to convert the model to finalfusion format.

```shell
$ cargo run --release --bin compile_fasttext -- -i cc.en.300.bin -o cc.en.300.fifu
```

