# SentEval STS Task

Here, we provide a tool to evaluate this library on [SentEval STS Task](https://github.com/princeton-nlp/SimCSE/tree/main/SentEval) provided by the [SimCSE](https://github.com/princeton-nlp/SimCSE) repository.

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
Prepare a model following [finalfusion-tools/README.md](../../finalfusion-tools/README.md).

Here, we assume that you have `glove.42B.300d.fifu` in the current directory.

### 3. Evaluate

Run the following command:

```shell
$ cargo run --release --features openblas -- -d data/STS -f glove.42B.300d.fifu > score.tsv
```

This command will report the Pearson correlation coefficient between the cosine similarity of the sentence embeddings and the gold scores.

```
$ sudo apt install libgsl0-dev
```

## Experimental results

We show the actual results obtained by the above procedure using `glove.42B.300d.fifu` (GloVe+WR) or `cc.en.300.bin.fifu` (fastText+WR).

As baseline methods, we also show the following results:
- SIF-ICLR17: Obtained from the original SIF paper (Table 5 in [ICLR 2017](https://openreview.net/forum?id=SyK00v5xx)).
- SimCSE: Obtained by the evaluation tool in the [SimCSE](https://github.com/princeton-nlp/SimCSE) repository.

We note that the mean result in 2013 does not contain that of SMT because the SMT dataset is not available.

### Pearson correlation coefficient ($\times 100$)

| 2012              | Sif<br>(GloVe) | USif<br>(GloVe) | Sif<br>(fastText) | USif<br>(fastText) | SimCSE<br>(unsup-simcse-<br>bert-base-uncased) | SimCSE<br>(sup-simcse-<br>bert-base-uncased) |
| ----------------- | -------------: | --------------: | ----------------: | -----------------: | ---------------------------------------------: | -------------------------------------------: |
| MSRpar            |           40.7 |            42.2 |              36.9 |               43.0 |                                           63.1 |                                         62.0 |
| MSRvid            |           84.0 |            84.1 |              84.4 |               85.2 |                                           85.7 |                                         92.6 |
| SMTeuroparl       |           52.7 |            54.2 |              49.7 |               54.7 |                                           52.6 |                                         49.9 |
| surprise.OnWN     |           71.8 |            70.2 |              72.8 |               71.8 |                                           73.7 |                                         76.6 |
| surprise.SMTnews  |           53.5 |            57.0 |              54.6 |               60.7 |                                           65.5 |                                         72.9 |
| Avg.              |           60.5 |            61.5 |              59.7 |               63.1 |                                           68.1 |                                         70.8 |
|                   |                |                 |                   |                    |                                                |                                              |
| 2013              | Sif<br>(GloVe) | USif<br>(GloVe) | Sif<br>(fastText) | USif<br>(fastText) | SimCSE<br>(unsup-simcse-<br>bert-base-uncased) | SimCSE<br>(sup-simcse-<br>bert-base-uncased) |
| FNWN              |           49.0 |            53.9 |              54.6 |               55.6 |                                           62.2 |                                         62.9 |
| headlines         |           73.3 |            74.0 |              72.4 |               74.7 |                                           78.5 |                                         80.1 |
| OnWN              |           83.4 |            84.0 |              84.2 |               85.5 |                                           86.5 |                                         87.7 |
| Avg.              |           68.5 |            70.6 |              70.4 |               71.9 |                                           75.7 |                                         76.9 |
|                   |                |                 |                   |                    |                                                |                                              |
| 2014              | Sif<br>(GloVe) | USif<br>(GloVe) | Sif<br>(fastText) | USif<br>(fastText) | SimCSE<br>(unsup-simcse-<br>bert-base-uncased) | SimCSE<br>(sup-simcse-<br>bert-base-uncased) |
| deft-forum        |           47.9 |            48.1 |              47.1 |               48.9 |                                           59.2 |                                         64.8 |
| deft-news         |           70.6 |            75.3 |              69.4 |               75.6 |                                           78.8 |                                         82.4 |
| headlines         |           69.7 |            69.3 |              69.0 |               70.4 |                                           76.9 |                                         79.3 |
| images            |           83.0 |            82.3 |              83.8 |               83.7 |                                           81.5 |                                         89.4 |
| OnWN              |           85.1 |            85.5 |              85.2 |               87.0 |                                           87.9 |                                         89.5 |
| tweet-news        |           77.6 |            77.1 |              78.3 |               78.2 |                                           79.6 |                                         83.6 |
| Avg.              |           72.3 |            72.9 |              72.2 |               74.0 |                                           77.3 |                                         81.5 |
|                   |                |                 |                   |                    |                                                |                                              |
| 2015              | Sif<br>(GloVe) | USif<br>(GloVe) | Sif<br>(fastText) | USif<br>(fastText) | SimCSE<br>(unsup-simcse-<br>bert-base-uncased) | SimCSE<br>(sup-simcse-<br>bert-base-uncased) |
| answers-forums    |           69.3 |            71.2 |              69.8 |               73.4 |                                           77.2 |                                         74.5 |
| answers-students  |           73.1 |            71.0 |              74.8 |               71.0 |                                           73.2 |                                         74.4 |
| belief            |           75.6 |            75.3 |              75.6 |               76.9 |                                           81.5 |                                         85.2 |
| headlines         |           75.3 |            76.2 |              74.0 |               75.7 |                                           81.4 |                                         82.1 |
| images            |           82.6 |            83.3 |              82.2 |               83.6 |                                           84.7 |                                         92.7 |
| Avg.              |           75.2 |            75.4 |              75.3 |               76.1 |                                           79.6 |                                         81.8 |
|                   |                |                 |                   |                    |                                                |                                              |
| 2016              | Sif<br>(GloVe) | USif<br>(GloVe) | Sif<br>(fastText) | USif<br>(fastText) | SimCSE<br>(unsup-simcse-<br>bert-base-uncased) | SimCSE<br>(sup-simcse-<br>bert-base-uncased) |
| answer-answer     |           50.6 |            49.5 |              56.9 |               52.8 |                                           68.3 |                                         76.3 |
| headlines         |           72.6 |            73.0 |              72.1 |               73.5 |                                           80.1 |                                         79.5 |
| plagiarism        |           81.8 |            82.6 |              83.0 |               84.9 |                                           84.8 |                                         84.3 |
| postediting       |           80.4 |            81.3 |              82.6 |               83.9 |                                           84.9 |                                         84.5 |
| question-question |           70.1 |            71.5 |              71.5 |               73.8 |                                           70.4 |                                         72.9 |
| Avg.              |           71.1 |            71.6 |              73.2 |               73.8 |                                           77.7 |                                         79.5 |
