# sif-embedding/wordfreq-tools

sif-embedding, or the SIF algorithm, requires unigram probabilities.
This directory provides tools to create unigram language models\
from data provided in [wordfreq](https://github.com/rspeer/wordfreq).

## 1. Download wordfreq data

Download the wordfreq repository and checkout the version 3.0.2 (if you want to reproduce our environment).

```shell
$ git clone https://github.com/rspeer/wordfreq.git
$ cd wordfreq
$ git checkout v3.0.2
$ cd ..
```

## 2. Preprocess datasets

Parse the wordfreq data and extract unigram probabilities.
(Here, we assume to parse the English data.)

```shell
$ python -m venv venv
$ source ./venv/bin/activate
$ pip install -r scripts/requirements.txt
$ python scripts/parse_msgpack.py wordfreq/wordfreq/data/large_en.msgpack.gz > en.txt
```

## 3. Compile unigram language models

Compile the unigram language model from the parsed data.

```shell
$ cargo run --release -- -i en.txt -o en.unigram
```
