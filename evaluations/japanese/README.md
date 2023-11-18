# Similarity evaluation tasks for Japanese

Here, we provide a tool to evaluate this library on
[JGLUE JSTS](https://github.com/yahoojapan/JGLUE) and [JSICK](https://github.com/verypluming/JSICK) tasks.

## Requirements

This tool employs [rust-GSL](https://github.com/GuillaumeGomez/rust-GSL)
to compute Pearson's and Spearman's correlation coefficients.
You need to install the GSL library, following https://crates.io/crates/GSL/6.0.0.

## Evaluation steps

We show steps to run the evaluation, assuming you are at this directory.

### 1. Download datasets

Run the following command:

```shell
$ ./download.sh
```

We will use the data under the `data/jsick` and `data/jsts` directories.

### 2. Prepare pretrained word embeddings

You need to prepare pretrained word embeddings in [finalfusion](https://docs.rs/finalfusion/) format.
Prepare a model following [finalfusion-tools](../../finalfusion-tools).

Here, we assume that you have `cc.ja.300.vec.fifu` in the current directory.

### 3. Prepare Vibrato models

The model `cc.ja.300.vec` is trained by word segmentation using MeCab as a preprocessor.
To reproduce the word segmentation, we will preprocess sentences using [Vibrato](https://github.com/daac-tools/vibrato), a MeCab-compatible Rust library.

Although not explicitly stated on the fastText website, we assume that IPADIC was used and will use the IPADIC model distributed on the Vibrato website.

```shell
$ wget https://github.com/daac-tools/vibrato/releases/download/v0.5.0/ipadic-mecab-2_7_0.tar.xz
$ tar xf ipadic-mecab-2_7_0.tar.xz
```

### 4. Run evaluation

`src/main.rs` provides evaluation for SIF (with `-m sif`) and uSIF (with `-m usif`).
This tool will report the Pearson's and Spearman's correlation coefficients
between the cosine similarity of the sentence embeddings and the gold scores.

#### 4.1 JSTS

An example command is as follows:

```shell
$ cargo run --release --features openblas -- \
    -d data/jsts/valid-v1.1.json \
    -e jsts \
    -f cc.ja.300.vec.fifu \
    -v ipadic-mecab-2_7_0/system.dic.zst \
    -m sif
```

#### 4.1 JSICK

An example command is as follows:

```shell
$ cargo run --release --features openblas -- \
    -d data/jsick/test.tsv \
    -e jsick_test \
    -f cc.ja.300.vec.fifu \
    -v ipadic-mecab-2_7_0/system.dic.zst \
    -m sif
```

## Experimental results

We show the results obtained by the above commands.
The values are Spearman's rank correlation coefficient (×100).

As a comparison, results using Japanese-SimCSE models published by [cl-nagoya](https://huggingface.co/cl-nagoya) are also shown.
The results are obtained from the [technical report](https://arxiv.org/abs/2310.19349).

| Model                           | JSICK (test) | JSTS (train) | JSTS (val) | Avg.  |
| ------------------------------- | :----------: | :----------: | :--------: | :---: |
| sif_embedding::Sif              |     79.7     |     67.6     |    74.6    | 74.0  |
| sif_embedding::USif             |     79.7     |     69.3     |    76.0    | 75.0  |
| cl-nagoya/unsup-simcse-ja-base  |     79.0     |     74.5     |    79.0    | 77.5  |
| cl-nagoya/unsup-simcse-ja-large |     79.6     |     77.8     |    81.4    | 79.6  |
| cl-nagoya/sup-simcse-ja-base    |     82.8     |     77.9     |    80.9    | 80.5  |
| cl-nagoya/sup-simcse-ja-large   |     83.1     |     79.6     |    83.1    | 81.9  |
