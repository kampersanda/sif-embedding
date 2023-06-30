# SentEval STS Task

Here, we provide a tool to evaluate this library on [SentEval STS Task](https://github.com/princeton-nlp/SimCSE/tree/main/SentEval) provided by the [SimCSE](https://github.com/princeton-nlp/SimCSE) repository.

## Requirements

This tool employs [rust-GSL](https://github.com/GuillaumeGomez/rust-GSL)
to compute Pearson's and Spearman's correlation coefficients.
You need to install the GSL library, following https://crates.io/crates/GSL/6.0.0.

## Evaluation steps

We show steps to run the evaluation, assuming you are at this directory.

### 1. Download SentEval dataset

Run the following commands:

```shell
$ wget https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/senteval.tar
$ mkdir data
$ tar xvf senteval.tar -C data
```

We will use the data under the `data/STS` directory (STS12-16).

### 2. Prepare pretrained word embeddings

You need to prepare pretrained word embeddings in [finalfusion](https://docs.rs/finalfusion/) format.
Prepare a model following [finalfusion-tools](../../finalfusion-tools).

Here, we assume that you have `glove.42B.300d.fifu` in the current directory.

### 3. Evaluate

`src/main.rs` provides evaluation for the SIF and uSIF algorithms.
Run the following commands:

```shell
$ cargo run --release --features openblas -- -d data/STS -f glove.42B.300d.fifu -m sif > sif-score.tsv
$ cargo run --release --features openblas -- -d data/STS -f glove.42B.300d.fifu -m usif > usif-score.tsv
```

This command will report the Pearson's and Spearman's correlation coefficients between the cosine similarity of the sentence embeddings and the gold scores.

## Experimental results

We show the results obtained by the above procedure using `glove.42B.300d.fifu` (GloVe) or `cc.en.300.bin.fifu` (fastText).

As baseline methods, we also show the results of SimCSE obtained by the evaluation tool in [original repository](https://github.com/princeton-nlp/SimCSE).
We evaluated two models: `unsup-simcse-bert-base-uncased` and `sup-simcse-bert-base-uncased`.

### Pearson's correlation coefficient ($\times 100$)

| 2012              | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| ----------------- | -------------: | ----------------: | --------------: | -----------------: | ----------------: | --------------: |
| MSRpar            |           40.7 |              36.9 |            42.2 |               43.0 |              63.1 |            62.0 |
| MSRvid            |           84.0 |              84.4 |            84.1 |               85.2 |              85.7 |            92.6 |
| SMTeuroparl       |           52.7 |              49.7 |            54.2 |               54.7 |              52.6 |            49.9 |
| surprise.OnWN     |           71.8 |              72.8 |            70.2 |               71.8 |              73.7 |            76.6 |
| surprise.SMTnews  |           53.5 |              54.6 |            57.0 |               60.7 |              65.5 |            72.9 |
| Avg.              |           60.5 |              59.7 |            61.5 |               63.1 |              68.1 |            70.8 |
|                   |                |                   |                 |                    |                   |                 |
| 2013              | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| FNWN              |           49.0 |              54.6 |            53.9 |               55.6 |              62.2 |            62.9 |
| headlines         |           73.3 |              72.4 |            74.0 |               74.7 |              78.5 |            80.1 |
| OnWN              |           83.4 |              84.2 |            84.0 |               85.5 |              86.5 |            87.7 |
| Avg.              |           68.5 |              70.4 |            70.6 |               71.9 |              75.7 |            76.9 |
|                   |                |                   |                 |                    |                   |                 |
| 2014              | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| deft-forum        |           47.9 |              47.1 |            48.1 |               48.9 |              59.2 |            64.8 |
| deft-news         |           70.6 |              69.4 |            75.3 |               75.6 |              78.8 |            82.4 |
| headlines         |           69.7 |              69.0 |            69.3 |               70.4 |              76.9 |            79.3 |
| images            |           83.0 |              83.8 |            82.3 |               83.7 |              81.5 |            89.4 |
| OnWN              |           85.1 |              85.2 |            85.5 |               87.0 |              87.9 |            89.5 |
| tweet-news        |           77.6 |              78.3 |            77.1 |               78.2 |              79.6 |            83.6 |
| Avg.              |           72.3 |              72.2 |            72.9 |               74.0 |              77.3 |            81.5 |
|                   |                |                   |                 |                    |                   |                 |
| 2015              | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| answers-forums    |           69.3 |              69.8 |            71.2 |               73.4 |              77.2 |            74.5 |
| answers-students  |           73.1 |              74.8 |            71.0 |               71.0 |              73.2 |            74.4 |
| belief            |           75.6 |              75.6 |            75.3 |               76.9 |              81.5 |            85.2 |
| headlines         |           75.3 |              74.0 |            76.2 |               75.7 |              81.4 |            82.1 |
| images            |           82.6 |              82.2 |            83.3 |               83.6 |              84.7 |            92.7 |
| Avg.              |           75.2 |              75.3 |            75.4 |               76.1 |              79.6 |            81.8 |
|                   |                |                   |                 |                    |                   |                 |
| 2016              | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| answer-answer     |           50.6 |              56.9 |            49.5 |               52.8 |              68.3 |            76.3 |
| headlines         |           72.6 |              72.1 |            73.0 |               73.5 |              80.1 |            79.5 |
| plagiarism        |           81.8 |              83.0 |            82.6 |               84.9 |              84.8 |            84.3 |
| postediting       |           80.4 |              82.6 |            81.3 |               83.9 |              84.9 |            84.5 |
| question-question |           70.1 |              71.5 |            71.5 |               73.8 |              70.4 |            72.9 |
| Avg.              |           71.1 |              73.2 |            71.6 |               73.8 |              77.7 |            79.5 |

### Spearman's correlation coefficient ($\times 100$)

| 2012              | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| ----------------- | -------------: | ----------------: | --------------: | -----------------: | ----------------: | --------------: |
| MSRpar            |           45.8 |              41.7 |            46.7 |               46.6 |              64.3 |            62.5 |
| MSRvid            |           81.8 |              82.4 |            81.3 |               82.6 |              85.1 |            92.9 |
| SMTeuroparl       |           59.3 |              57.5 |            61.8 |               61.9 |              61.5 |            58.8 |
| surprise.OnWN     |           67.0 |              68.2 |            66.5 |               68.2 |              69.8 |            69.9 |
| surprise.SMTnews  |           45.2 |              47.7 |            48.3 |               51.4 |              58.1 |            61.0 |
| Avg.              |           59.8 |              59.5 |            60.9 |               62.1 |              67.8 |            69.0 |
|                   |                |                   |                 |                    |                   |                 |
| 2013              | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| FNWN              |           50.0 |              55.3 |            54.4 |               56.2 |              64.1 |            63.8 |
| headlines         |           72.9 |              71.6 |            72.7 |               73.5 |              79.0 |            82.3 |
| OnWN              |           81.8 |              82.9 |            81.8 |               83.5 |              83.8 |            86.4 |
| Avg.              |           68.2 |              70.0 |            69.6 |               71.1 |              75.6 |            77.5 |
|                   |                |                   |                 |                    |                   |                 |
| 2014              | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| deft-forum        |           47.8 |              45.9 |            48.1 |               48.2 |              57.6 |            64.4 |
| deft-news         |           64.7 |              64.2 |            69.2 |               69.3 |              75.2 |            81.0 |
| headlines         |           65.7 |              64.9 |            64.7 |               66.0 |              75.9 |            79.1 |
| images            |           78.1 |              79.5 |            77.6 |               79.2 |              77.8 |            86.6 |
| OnWN              |           84.4 |              85.0 |            84.4 |               85.4 |              85.1 |            87.5 |
| tweet-news        |           71.7 |              72.9 |            71.1 |               72.9 |              72.3 |            76.8 |
| Avg.              |           68.7 |              68.7 |            69.2 |               70.2 |              74.0 |            79.2 |
|                   |                |                   |                 |                    |                   |                 |
| 2015              | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| answers-forums    |           70.3 |              70.5 |            72.5 |               74.4 |              77.7 |            74.6 |
| answers-students  |           73.2 |              75.0 |            70.9 |               71.3 |              73.5 |            75.1 |
| belief            |           77.2 |              77.2 |            76.6 |               78.6 |              83.3 |            87.2 |
| headlines         |           74.6 |              72.6 |            75.2 |               74.2 |              82.0 |            85.4 |
| images            |           82.9 |              82.5 |            83.2 |               83.6 |              86.2 |            93.7 |
| Avg.              |           75.7 |              75.6 |            75.7 |               76.4 |              80.5 |            83.2 |
|                   |                |                   |                 |                    |                   |                 |
| 2016              | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| answer-answer     |           49.5 |              56.6 |            49.0 |               52.5 |              68.1 |            76.5 |
| headlines         |           72.8 |              72.3 |            72.9 |               73.8 |              81.8 |            83.5 |
| plagiarism        |           82.0 |              82.8 |            83.2 |               85.5 |              85.8 |            86.5 |
| postediting       |           82.7 |              84.1 |            82.9 |               85.0 |              86.2 |            88.6 |
| question-question |           71.5 |              73.1 |            71.9 |               74.6 |              70.1 |            73.5 |
| Avg.              |           71.7 |              73.8 |            72.0 |               74.3 |              78.4 |            81.7 |
