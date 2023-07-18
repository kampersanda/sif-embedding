# sif-embedding

![](https://github.com/kampersanda/sif-embedding/actions/workflows/rust.yml/badge.svg)
[![Documentation](https://docs.rs/sif-embedding/badge.svg)](https://docs.rs/sif-embedding)
[![Crates.io](https://img.shields.io/crates/v/sif-embedding.svg)](https://crates.io/crates/sif-embedding)

This is a Rust implementation of simple but powerful sentence embedding algorithms based on
*Smooth Inverse Frequency* and *Common Component Removal* described in the following papers:

 - Sanjeev Arora, Yingyu Liang, and Tengyu Ma,
   [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/forum?id=SyK00v5xx),
   ICLR 2017
 - Kawin Ethayarajh,
   [Unsupervised Random Walk Sentence Embeddings: A Strong but Simple Baseline](https://aclanthology.org/W18-3012/),
   RepL4NLP 2018

This library will help you if

 - DNN-based sentence embeddings are too slow for your application, or
 - you do not have an option using GPUs.

## Documentation

https://docs.rs/sif-embedding/

## Getting started

See [tutorial](./tutorial).

## Evaluations

[evaluations](./evaluations/) provides tools to evaluate sif-embedding on several tasks.

## Wiki

[Trouble shooting](https://github.com/kampersanda/sif-embedding/wiki/Trouble-shooting): Tips on how to resolve errors I faced in my environment.

## Licensing

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
