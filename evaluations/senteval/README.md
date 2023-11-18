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

We will use the data under the `data/STS` and `data/SICK` directories.

### 2. Prepare pretrained word embeddings

You need to prepare pretrained word embeddings in [finalfusion](https://docs.rs/finalfusion/) format.
Prepare a model following [finalfusion-tools](../../finalfusion-tools).

Here, we assume that you have `glove.42B.300d.fifu` in the current directory.

### 3. Evaluate

`src/main.rs` provides evaluation for the SIF and uSIF algorithms.
Run the following commands:

```shell
$ cargo run --release --features openblas -- -d data -f glove.42B.300d.fifu -m sif > sif-score.tsv
$ cargo run --release --features openblas -- -d data -f glove.42B.300d.fifu -m usif > usif-score.tsv
```

This command will report the Pearson's and Spearman's correlation coefficients between the cosine similarity of the sentence embeddings and the gold scores.

## Experimental results

We show the results obtained by the above procedure using `glove.42B.300d.fifu` (GloVe) or `cc.en.300.bin.fifu` (fastText).

As baseline methods, we also show the results of SimCSE obtained by the evaluation tool in [original repository](https://github.com/princeton-nlp/SimCSE).
We evaluated two models: `unsup-simcse-bert-base-uncased` and `sup-simcse-bert-base-uncased`.

### Pearson's correlation coefficient (×100)

| STS/STS12-en-test   | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| ------------------- | :------------: | :---------------: | :-------------: | :----------------: | :---------------: | :-------------: |
| MSRpar              |      37.9      |       36.5        |      38.4       |        41.3        |       63.1        |      62.0       |
| MSRvid              |      83.9      |       84.3        |      83.5       |        85.2        |       85.7        |      92.6       |
| SMTeuroparl         |      49.1      |       50.6        |      49.1       |        52.7        |       52.6        |      49.9       |
| surprise.OnWN       |      66.9      |       66.4        |      63.9       |        65.8        |       73.7        |      76.6       |
| surprise.SMTnews    |      48.4      |       47.3        |      52.0       |        53.9        |       65.5        |      72.9       |
| Avg.                |      57.2      |       57.0        |      57.4       |        59.8        |       68.1        |      70.8       |
|                     |                |                   |                 |                    |                   |                 |
| STS/STS13-en-test   | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| FNWN                |      43.6      |       47.1        |      49.4       |        48.8        |       62.2        |      62.9       |
| headlines           |      71.3      |       68.9        |      71.7       |        71.9        |       78.5        |      80.1       |
| OnWN                |      80.0      |       80.1        |      78.7       |        81.0        |       86.5        |      87.7       |
| Avg.                |      65.0      |       65.4        |      66.6       |        67.2        |       75.7        |      76.9       |
|                     |                |                   |                 |                    |                   |                 |
| STS/STS14-en-test   | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| deft-forum          |      36.2      |       36.8        |      35.8       |        39.2        |       59.2        |      64.8       |
| deft-news           |      71.6      |       69.0        |      75.4       |        75.1        |       78.8        |      82.4       |
| headlines           |      66.2      |       64.4        |      65.4       |        66.2        |       76.9        |      79.3       |
| images              |      82.2      |       82.9        |      81.6       |        83.2        |       81.5        |      89.4       |
| OnWN                |      82.7      |       82.3        |      81.5       |        83.3        |       87.9        |      89.5       |
| tweet-news          |      70.6      |       68.3        |      69.8       |        70.0        |       79.6        |      83.6       |
| Avg.                |      68.2      |       67.3        |      68.3       |        69.5        |       77.3        |      81.5       |
|                     |                |                   |                 |                    |                   |                 |
| STS/STS15-en-test   | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| answers-forums      |      66.9      |       65.9        |      68.0       |        70.3        |       77.2        |      74.5       |
| answers-students    |      72.6      |       73.7        |      71.2       |        70.9        |       73.2        |      74.4       |
| belief              |      70.4      |       68.9        |      69.9       |        71.8        |       81.5        |      85.2       |
| headlines           |      72.7      |       70.9        |      72.6       |        73.0        |       81.4        |      82.1       |
| images              |      82.0      |       81.1        |      82.4       |        82.9        |       84.7        |      92.7       |
| Avg.                |      72.9      |       72.1        |      72.8       |        73.8        |       79.6        |      81.8       |
|                     |                |                   |                 |                    |                   |                 |
| STS/STS16-en-test   | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| answer-answer       |      40.1      |       49.3        |      39.6       |        45.8        |       68.3        |      76.3       |
| headlines           |      71.2      |       70.1        |      71.4       |        71.7        |       80.1        |      79.5       |
| plagiarism          |      79.5      |       78.8        |      80.4       |        81.9        |       84.8        |      84.3       |
| postediting         |      81.3      |       82.4        |      79.1       |        82.1        |       84.9        |      84.5       |
| question-question   |      66.9      |       66.9        |      67.9       |        70.6        |       70.4        |      72.9       |
| Avg.                |      67.8      |       69.5        |      67.7       |        70.4        |       77.7        |      79.5       |
|                     |                |                   |                 |                    |                   |                 |
| STS/STSBenchmark    | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| sts-train           |      71.5      |       70.4        |      71.9       |        73.1        |       79.0        |      83.7       |
| sts-dev             |      76.3      |       75.4        |      77.6       |        78.6        |       81.4        |      85.7       |
| sts-test            |      67.7      |       67.6        |      67.3       |        69.9        |       77.3        |      83.3       |
| Avg.                |      71.8      |       71.1        |      72.3       |        73.8        |       79.2        |      84.2       |
|                     |                |                   |                 |                    |                   |                 |
| SICK                | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| SICK_train          |      72.0      |       72.5        |      71.9       |        73.2        |       79.9        |      85.8       |
| SICK_trial          |      69.1      |       70.4        |      69.6       |        71.3        |       78.7        |      85.2       |
| SICK_test_annotated |      71.0      |       71.9        |      70.4       |        71.9        |       79.2        |      85.1       |
| Avg.                |      70.7      |       71.6        |      70.6       |        72.1        |       79.3        |      85.4       |

### Spearman's correlation coefficient (×100)

| STS/STS12-en-test   | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| ------------------- | :------------: | :---------------: | :-------------: | :----------------: | :---------------: | :-------------: |
| MSRpar              |      41.5      |       39.9        |      42.2       |        44.0        |       64.3        |      62.5       |
| MSRvid              |      81.9      |       82.5        |      81.2       |        82.9        |       85.1        |      92.9       |
| SMTeuroparl         |      56.4      |       58.4        |      57.9       |        59.9        |       61.5        |      58.8       |
| surprise.OnWN       |      61.0      |       60.5        |      58.7       |        60.8        |       69.8        |      69.9       |
| surprise.SMTnews    |      40.2      |       40.7        |      41.8       |        43.5        |       58.1        |      61.0       |
| Avg.                |      56.2      |       56.4        |      56.4       |        58.2        |       67.8        |      69.0       |
|                     |                |                   |                 |                    |                   |                 |
| STS/STS13-en-test   | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| FNWN                |      45.4      |       48.2        |      51.5       |        48.6        |       64.1        |      63.8       |
| headlines           |      70.7      |       67.5        |      70.1       |        70.3        |       79.0        |      82.3       |
| OnWN                |      78.1      |       78.9        |      77.0       |        79.3        |       83.8        |      86.4       |
| Avg.                |      64.8      |       64.9        |      66.2       |        66.1        |       75.6        |      77.5       |
|                     |                |                   |                 |                    |                   |                 |
| STS/STS14-en-test   | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| deft-forum          |      37.5      |       37.0        |      36.4       |        39.0        |       57.6        |      64.4       |
| deft-news           |      65.8      |       63.8        |      68.3       |        68.7        |       75.2        |      81.0       |
| headlines           |      61.6      |       60.2        |      60.4       |        61.6        |       75.9        |      79.1       |
| images              |      77.0      |       78.4        |      76.7       |        78.6        |       77.8        |      86.6       |
| OnWN                |      81.9      |       82.0        |      80.7       |        82.0        |       85.1        |      87.5       |
| tweet-news          |      63.9      |       62.0        |      64.2       |        64.4        |       72.3        |      76.8       |
| Avg.                |      64.6      |       63.9        |      64.5       |        65.7        |       74.0        |      79.2       |
|                     |                |                   |                 |                    |                   |                 |
| STS/STS15-en-test   | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| answers-forums      |      68.1      |       66.3        |      69.3       |        71.5        |       77.7        |      74.6       |
| answers-students    |      72.5      |       73.7        |      71.0       |        71.1        |       73.5        |      75.1       |
| belief              |      71.4      |       70.6        |      70.8       |        73.3        |       83.3        |      87.2       |
| headlines           |      71.8      |       69.6        |      71.5       |        71.8        |       82.0        |      85.4       |
| images              |      82.3      |       81.4        |      82.4       |        83.0        |       86.2        |      93.7       |
| Avg.                |      73.2      |       72.3        |      73.0       |        74.1        |       80.5        |      83.2       |
|                     |                |                   |                 |                    |                   |                 |
| STS/STS16-en-test   | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| answer-answer       |      40.7      |       49.4        |      38.7       |        45.9        |       68.1        |      76.5       |
| headlines           |      71.0      |       70.3        |      70.9       |        71.8        |       81.8        |      83.5       |
| plagiarism          |      79.9      |       79.3        |      80.8       |        82.9        |       85.8        |      86.5       |
| postediting         |      82.6      |       83.5        |      81.2       |        83.5        |       86.2        |      88.6       |
| question-question   |      68.3      |       68.1        |      68.3       |        70.9        |       70.1        |      73.5       |
| Avg.                |      68.5      |       70.1        |      68.0       |        71.0        |       78.4        |      81.7       |
|                     |                |                   |                 |                    |                   |                 |
| STS/STSBenchmark    | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| sts-train           |      66.3      |       65.2        |      66.4       |        68.0        |       76.9        |      83.3       |
| sts-dev             |      76.3      |       75.3        |      77.3       |        78.2        |       81.7        |      86.2       |
| sts-test            |      63.6      |       63.6        |      63.2       |        66.3        |       76.5        |      84.3       |
| Avg.                |      68.8      |       68.0        |      69.0       |        70.8        |       78.4        |      84.6       |
|                     |                |                   |                 |                    |                   |                 |
| SICK                | Sif<br>(GloVe) | Sif<br>(fastText) | USif<br>(GloVe) | USif<br>(fastText) | SimCSE<br>(unsup) | SimCSE<br>(sup) |
| SICK_train          |      59.4      |       60.3        |      58.9       |        60.2        |       72.5        |      81.0       |
| SICK_trial          |      57.9      |       59.3        |      57.7       |        60.3        |       74.2        |      82.6       |
| SICK_test_annotated |      58.7      |       60.1        |      57.7       |        59.2        |       71.9        |      80.4       |
| Avg.                |      58.7      |       59.9        |      58.1       |        59.9        |       72.8        |      81.3       |
