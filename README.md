# sif-embedding

![](https://github.com/kampersanda/sif-embedding/actions/workflows/rust.yml/badge.svg)
[![Documentation](https://docs.rs/sif-embedding/badge.svg)](https://docs.rs/sif-embedding)
[![Crates.io](https://img.shields.io/crates/v/sif-embedding.svg)](https://crates.io/crates/sif-embedding)

**This is currently a prototype version.**

This is a Rust implementation of *smooth inverse frequency (SIF)* that is a simple but powerful embedding technique for sentences, described in the paper:

> Sanjeev Arora, Yingyu Liang, and Tengyu Ma,
> [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/forum?id=SyK00v5xx),
> ICLR 2017.

## Documentation

https://docs.rs/sif-embedding/

## Specifications

This library depends on [ndarray-linalg](https://github.com/rust-ndarray/ndarray-linalg).
You must *always* specify which backend will be used with `features`, following the specifications of ndarray-linalg.
See [README of ndarray-linalg v0.16.0](https://github.com/rust-ndarray/ndarray-linalg/tree/ndarray-linalg-v0.16.0) since the feature names of sif-embedding are the same.

For example, you can specify the [OpenBLAS](https://www.openblas.net/) backend as follows:

```toml
# Cargo.toml

[dependencies]
sif-embedding = { version = "0.1", features = ["openblas"] }
```

If you are having problems compiling this library due to the backend, [my tips](https://github.com/kampersanda/sif-embedding/wiki/Trouble-shooting) may help.

## TODO

- [ ] Add the unsupervised SIF described in [the paper](https://aclanthology.org/W18-3012/).
- [ ] Support [fastText](https://fasttext.cc/) models for word embeddings.
- [ ] Support serialization/deserialization of models.
- [ ] Provide Python binding
- [ ] Conduct more evaluations.

## Evaluations

[`evaluations/semeval`](./evaluations/semeval) provides tools to evaluate sif-embedding on SemEval STS Task.

## Wiki

- [Trouble shooting](https://github.com/kampersanda/sif-embedding/wiki/Trouble-shooting): Tips on how to resolve errors I faced in my environment.

## Licensing

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
