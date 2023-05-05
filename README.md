# sif-embedding

**This is currently a prototype version.**

This is a Rust implementation of *Smooth inverse frequency (SIF)* that is a simple but powerful embedding technique for sentences, described in the paper:

> Sanjeev Arora, Yingyu Liang, and Tengyu Ma,
> [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/forum?id=SyK00v5xx),
> ICLR 2017.

## Documentation

https://docs.rs/sif-embedding/

## TODO

- Add the unsupervised SIF described in [the paper](https://aclanthology.org/W18-3012/).
- Support [fastText](https://fasttext.cc/) models for word embeddings.
- Conduct more evaluations.

## Evaluation

This repository provides an easy tool to evaluate this library using [SemEval STS Task](https://aclanthology.org/S16-1081/).
We show an procedure example run the evaluation, assuming you are in the root of this repository.

### 1. Download SemEval STS datasets

The [dataset-sts](https://github.com/brmson/dataset-sts) repository provides SemEval STS datasets from 2012 to 2016.
Download these and create a symbolic link to the target directory.

```shell
$ git clone https://github.com/brmson/dataset-sts.git
$ ln -s dataset-sts/data/sts/semeval-sts
```

### 2. Proprocess datasets

`scripts/clean_semeval_sts.py` is a script to preprocess sentences (such as tokenization, or lowercasing), referencing [the authors' code](https://github.com/PrincetonML/SIF).

```shell
$ python -m venv venv
$ . ./venv/bin/activate
$ pip install -r scripts/requirements.txt
$ python scripts/clean_semeval_sts.py
```

`semeval-sts-clean` will be generated.

```shell
$ ls -1 semeval-sts-clean/all
2012.MSRpar.test.tsv
2012.MSRpar.train.tsv
2012.OnWN.test.tsv
...
```

### 3. Download pretrained word embeddings

Download a pretrained model of word embeddings, such as [GloVe](https://nlp.stanford.edu/projects/glove/).

```shell
$ wget https://nlp.stanford.edu/data/glove.840B.300d.zip
$ unzip glove.840B.300d.zip
```

### 4. Run evaluation

`cli/semeval_sts` is a command line tool that evaluates `sif_embedding::Sif` using the SemEval STS datasets.
The SIF algorithm requires unigram probabilities.
You can use `auxiliary_data/enwiki_vocab_min200.txt` that has word frequencies (copied from [the authors' repository](https://github.com/PrincetonML/SIF)).

```shell
$ cargo run --release -p cli --bin semeval_sts -- -e glove.840B.300d.txt -w auxiliary_data/enwiki_vocab_min200.txt -c semeval-sts-clean/all
```

This will report Pearson’s $r$ between estimated similarities (i.e., cosine similarity between sentence embeddings) and gold scores, following the evaluation metric of the task.

### Experimental results

The following table shows the actual results obtained from the above procedure.
The original results by the authors are also shown as a baseline, from Table 5 (GloVe+WR) in [ICLR 2017](https://openreview.net/forum?id=SyK00v5xx).

| Dataset                    | ICLR 2017 | `sif_embedding::Sif` |
| -------------------------- | --------- | -------------------- |
| 2012                       |           |                      |
| MSRpar.test.tsv            | 35.6%     | 21.9%                |
| OnWN.test.tsv              | 66.2%     | 66.2%                |
| SMTeuroparl.test.tsv       | 49.9%     | 50.3%                |
| SMTnews.test.tsv           | 45.6%     | 48.7%                |
| 2013                       |           |                      |
| FNWN.test.tsv              | 39.4%     | 40.5%                |
| headlines.test.tsv         | 69.2%     | 70.4%                |
| OnWN.test.tsv              | 82.8%     | 80.1%                |
| 2014                       |           |                      |
| deft-forum.test.tsv        | 41.2%     | 41.1%                |
| deft-news.test.tsv         | 69.4%     | 69.3%                |
| headlines.test.tsv         | 64.7%     | 65.5%                |
| images.test.tsv            | 82.6%     | 82.9%                |
| OnWN.test.tsv              | 82.8%     | 83.1%                |
| 2015                       |           |                      |
| answers-forums.test.tsv    | 63.9%     | 63.9%                |
| answers-students.test.tsv  | 70.4%     | 70.7%                |
| belief.test.tsv            | 71.8%     | 72.5%                |
| headlines.test.tsv         | 70.7%     | 73.5%                |
| images.test.tsv            | 81.5%     | 81.5%                |
| 2016                       |           |                      |
| answer-answer.test.tsv     | NA        | 51.9%                |
| headlines.test.tsv         | NA        | 69.7%                |
| plagiarism.test.tsv        | NA        | 79.4%                |
| postediting.test.tsv       | NA        | 79.4%                |
| question-question.test.tsv | NA        | 69.6%                |

This library is not an exact port of the original code, and the experimental results do not exactly match.
However, similar results were obtained (except for `2012.MSRpar.test.tsv`).

## Trouble shooting

I provide tips on how to resolve errors I faced in my environment (WSL2+Ubuntu22.04 that installs `build-essential` at least).

### Compile error from ndarray-linalg

Maybe due to FFI for LAPACK.

#### openssl-sys

```shell
$ cargo build
...
error: failed to run custom build command for `openssl-sys vX.Y.Z`
```

However, OpenSSL is installed.

```shell
$ openssl version
OpenSSL 3.0.2 15 Mar 2022 (Library: OpenSSL 3.0.2 15 Mar 2022)
```

It is necessary to set up the link correctly.
First, find the locations of openssl.

```shell
$ dpkg -L libssl-dev | grep lib
$ dpkg -L libssl-dev | grep include
```

Next, set up links to the appropriate paths.

```shell
$ export OPENSSL_LIB_DIR=/usr/lib/x86_64-linux-gnu
$ export OPENSSL_INCLUDE_DIR=/usr/include/openssl
```

Solve it!

#### openblas-src

```shell
$ cargo build
...
error: failed to run custom build command for `openblas-src v0.10.8`
```

Maybe, Fortran is not installed.

```shell
$ sudo apt install gfortran
```

Solve it!

## Licensing

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
