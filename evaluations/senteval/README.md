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

## Experimental results

We show the actual results obtained by the above procedure using `glove.42B.300d.fifu` (GloVe+WR) or `cc.en.300.fifu` (fastText+WR).

As baseline methods, we also show the following results:
- SIF-ICLR17: Obtained from the original SIF paper (Table 5 in [ICLR 2017](https://openreview.net/forum?id=SyK00v5xx)).
- SimCSE: Obtained by the evaluation tool in the [SimCSE](https://github.com/princeton-nlp/SimCSE) repository.

We note that the mean result in 2013 does not contain that of SMT because the SMT dataset is not available.

### Pearson correlation coefficient

| 2012              | sif-embedding<br>(fastText+WR) | sif-embedding<br>(GloVe+WR) | SIF-ICLR17<br>(GloVe+WR) | SimCSE<br>(unsup-simcse-<br>bert-base-uncased) | SimCSE<br>(sup-simcse-<br>bert-base-uncased) |
| ----------------- | -----------------------------: | --------------------------: | -----------------------: | ---------------------------------------------: | -------------------------------------------: |
| MSRpar            |                          35.4% |                       39.5% |                    35.6% |                                          63.1% |                                        62.0% |
| MSRvid            |                          84.6% |                       84.1% |                    83.8% |                                          85.7% |                                        92.6% |
| SMTeuroparl       |                          49.7% |                       51.2% |                    49.9% |                                          52.6% |                                        49.9% |
| surprise.OnWN     |                          72.7% |                       71.6% |                    66.2% |                                          73.7% |                                        76.6% |
| surprise.SMTnews  |                          54.7% |                       53.4% |                    45.6% |                                          65.5% |                                        72.9% |
| Mean              |                          59.4% |                       60.0% |                    56.2% |                                          68.1% |                                        70.8% |
|                   |                                |                             |                          |                                                |                                              |
| 2013              | sif-embedding<br>(fastText+WR) | sif-embedding<br>(GloVe+WR) | SIF-ICLR17<br>(GloVe+WR) | SimCSE<br>(unsup-simcse-<br>bert-base-uncased) | SimCSE<br>(sup-simcse-<br>bert-base-uncased) |
| FNWN              |                          54.3% |                       48.9% |                    39.4% |                                          62.2% |                                        62.9% |
| headlines         |                          72.4% |                       73.3% |                    69.2% |                                          78.5% |                                        80.1% |
| OnWN              |                          84.7% |                       83.6% |                    82.8% |                                          86.5% |                                        87.7% |
| Mean              |                          70.5% |                       68.6% |                    63.8% |                                          75.7% |                                        76.9% |
|                   |                                |                             |                          |                                                |                                              |
| 2014              | sif-embedding<br>(fastText+WR) | sif-embedding<br>(GloVe+WR) | SIF-ICLR17<br>(GloVe+WR) | SimCSE<br>(unsup-simcse-<br>bert-base-uncased) | SimCSE<br>(sup-simcse-<br>bert-base-uncased) |
| deft-forum        |                          46.8% |                       47.8% |                    41.2% |                                          59.2% |                                        64.8% |
| deft-news         |                          69.8% |                       70.8% |                    69.4% |                                          78.8% |                                        82.4% |
| headlines         |                          69.1% |                       69.7% |                    64.7% |                                          76.9% |                                        79.3% |
| images            |                          83.8% |                       83.0% |                    82.6% |                                          81.5% |                                        89.4% |
| OnWN              |                          85.6% |                       85.3% |                    82.8% |                                          87.9% |                                        89.5% |
| tweet-news        |                          78.4% |                       77.6% |                    70.1% |                                          79.6% |                                        83.6% |
| Mean              |                          72.2% |                       72.4% |                    68.5% |                                          77.3% |                                        81.5% |
|                   |                                |                             |                          |                                                |                                              |
| 2015              | sif-embedding<br>(fastText+WR) | sif-embedding<br>(GloVe+WR) | SIF-ICLR17<br>(GloVe+WR) | SimCSE<br>(unsup-simcse-<br>bert-base-uncased) | SimCSE<br>(sup-simcse-<br>bert-base-uncased) |
| answers-forums    |                          69.5% |                       69.1% |                    63.9% |                                          77.2% |                                        74.5% |
| answers-students  |                          73.3% |                       71.7% |                    70.4% |                                          73.2% |                                        74.4% |
| belief            |                          75.6% |                       75.6% |                    71.8% |                                          81.5% |                                        85.2% |
| headlines         |                          74.1% |                       75.3% |                    70.7% |                                          81.4% |                                        82.1% |
| images            |                          82.5% |                       82.8% |                    81.5% |                                          84.7% |                                        92.7% |
| Mean              |                          75.0% |                       74.9% |                    71.7% |                                          79.6% |                                        81.8% |
|                   |                                |                             |                          |                                                |                                              |
| 2016              | sif-embedding<br>(fastText+WR) | sif-embedding<br>(GloVe+WR) | SIF-ICLR17<br>(GloVe+WR) | SimCSE<br>(unsup-simcse-<br>bert-base-uncased) | SimCSE<br>(sup-simcse-<br>bert-base-uncased) |
| answer-answer     |                          55.6% |                       50.4% |                          |                                          68.3% |                                        76.3% |
| headlines         |                          72.3% |                       72.7% |                          |                                          80.1% |                                        79.5% |
| plagiarism        |                          82.4% |                       81.3% |                          |                                          84.8% |                                        84.3% |
| postediting       |                          83.0% |                       80.8% |                          |                                          84.9% |                                        84.5% |
| question-question |                          72.1% |                       70.4% |                          |                                          70.4% |                                        72.9% |
| Mean              |                          73.1% |                       71.1% |                          |                                          77.7% |                                        79.5% |
