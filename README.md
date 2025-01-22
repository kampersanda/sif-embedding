# sif-embedding

<p align="left">
  <a href="https://github.com/kampersanda/sif-embedding/actions/workflows/rust.yml?query=branch%3Amain"><img src="https://img.shields.io/github/actions/workflow/status/kampersanda/sif-embedding/rust.yml?branch=main&style=flat-square" alt="actions status" /></a>
  &nbsp;
  <a href="https://crates.io/crates/sif-embedding"><img src="https://img.shields.io/crates/v/sif-embedding.svg?style=flat-square" alt="Crates.io version" /></a>
  &nbsp;
  <a href="https://docs.rs/sif-embedding"><img src="https://img.shields.io/badge/docs-latest-blue.svg?style=flat-square" alt="docs.rs docs" /></a>
</p>

This is a Rust implementation of simple but powerful sentence embedding algorithms based on
SIF and uSIF described in the following papers:

 - Sanjeev Arora, Yingyu Liang, and Tengyu Ma,
   [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/forum?id=SyK00v5xx),
   ICLR 2017
 - Kawin Ethayarajh,
   [Unsupervised Random Walk Sentence Embeddings: A Strong but Simple Baseline](https://aclanthology.org/W18-3012/),
   RepL4NLP 2018

My blog post in Japanese is available [here](https://kampersanda.hatenablog.jp/entry/2023/12/09/124846).

## Features

 - **No GPU required**: This library runs on CPU only.
 - **Fast embeddings**: This library provides fast sentence embeddings thanks to the simple algorithms of SIF and uSIF. We observed that our SIF implementation could process ~80K sentences per second on M2 MacBook Air. (See [benchmarks](./benchmarks/).)
 - **Reasonable evaluation scores**: The performances of SIF and uSIF on similarity evaluation tasks do not outperform those of SOTA models such as SimCSE. However, they are not so worse. (See [evaluations](./evaluations/).)

This library will help you if

 - DNN-based sentence embeddings are too slow for your application,
 - you do not have an option using GPUs, or
 - you want baseline sentence embeddings for your development.

## Documentation

https://docs.rs/sif-embedding/

## Getting started

See [tutorial](./tutorial).

## Benchmarks

[benchmarks](./benchmarks/) provides speed benchmarks.

We observed that, with an English Wikipedia dataset,
our SIF implementation could process ~80K sentences per second
on MacBook Air (one core of Apple M2, 24 GB RAM).

## Evaluations

[evaluations](./evaluations/) provides tools to evaluate sif-embedding on several similarity evaluation tasks.

### STS/SICK

[evaluations/senteval](./evaluations/senteval/) provides evaluation tools and results
for [SentEval STS/SICK Tasks](https://github.com/princeton-nlp/SimCSE/tree/main/SentEval).

As one example, the following table shows the evaluation results with the Spearman's rank correlation coefficient
for the STS-Benchmark.

| Model                                        | train |  dev  | test  | Avg.  |
| -------------------------------------------- | :---: | :---: | :---: | :---: |
| sif_embedding::Sif                           | 65.2  | 75.3  | 63.6  | 68.0  |
| sif_embedding::USif                          | 68.0  | 78.2  | 66.3  | 70.8  |
| princeton-nlp/unsup-simcse-bert-base-uncased | 76.9  | 81.7  | 76.5  | 78.4  |
| princeton-nlp/sup-simcse-bert-base-uncased   | 83.3  | 86.2  | 84.3  | 84.6  |

### JSTS/JSICK

[eveluations/japanese](./evaluations/japanese/) provides evaluation tools and results
for [JGLUE JSTS](https://github.com/yahoojapan/JGLUE) and [JSICK](https://github.com/verypluming/JSICK) tasks.

As one example, the following table shows the evaluation results with the Spearman's rank correlation coefficient.

| Model                           | JSICK (test) | JSTS (train) | JSTS (val) | Avg.  |
| ------------------------------- | :----------: | :----------: | :--------: | :---: |
| sif_embedding::Sif              |     79.7     |     67.6     |    74.6    | 74.0  |
| sif_embedding::USif             |     79.7     |     69.3     |    76.0    | 75.0  |
| cl-nagoya/unsup-simcse-ja-base  |     79.0     |     74.5     |    79.0    | 77.5  |
| cl-nagoya/unsup-simcse-ja-large |     79.6     |     77.8     |    81.4    | 79.6  |
| cl-nagoya/sup-simcse-ja-base    |     82.8     |     77.9     |    80.9    | 80.5  |
| cl-nagoya/sup-simcse-ja-large   |     83.1     |     79.6     |    83.1    | 81.9  |


## Similarity search

[qdrant-examples](./qdrant-examples/) provides an example of using sif-embedding with [qdrant/rust-client](https://github.com/qdrant/rust-client).

## Wiki

[Trouble shooting](https://github.com/kampersanda/sif-embedding/wiki/Trouble-shooting): Tips on how to resolve errors I faced in my environment.

## Licensing

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
