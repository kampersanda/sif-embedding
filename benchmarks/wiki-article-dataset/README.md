# Benchmark for wiki-article-dataset

This directory provides benchmark for [wiki-article-dataset](https://github.com/Hironsan/wiki-article-dataset).

## Evaluation steps

### 1. Download dataset

Run the following commands:

```shell
$ wget https://s3-ap-northeast-1.amazonaws.com/dev.tech-sketch.jp/chakki/public/ja.wikipedia_250k.zip
$ unzip ja.wikipedia_250k.zip
$ rm ja.wikipedia_250k.zip
```

### 2. Prepare pretrained word embeddings

You need to prepare pretrained word embeddings in [finalfusion](https://docs.rs/finalfusion/) format.
Prepare a model following [finalfusion-tools](../../finalfusion-tools).

Here, we assume that you have `cc.ja.300.vec.fifu` in the current directory.

### 3. Evaluate

`src/main.rs` provides evaluation for SIF (with `-m sif`) and uSIF (with `-m usif`).

```shell
$ cargo run --release --features openblas -- \
    -d ja.wikipedia_250k.txt \
    -f cc.ja.300.vec.fifu \
    -m sif
```
