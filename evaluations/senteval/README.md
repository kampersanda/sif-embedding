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

We will use the data under the `data/STS` directory (STS12--16).

### 2. Prepare pretrained word embeddings

You need to prepare pretrained word embeddings in [finalfusion](https://docs.rs/finalfusion/) format.
Prepare a model following [finalfusion-tools/README.md](../../finalfusion-tools/README.md).

Here, we assume that you have `glove.42B.300d.fifu` in the current directory.

### 3. Evaluate

Run the following command:

```shell
$ cargo run --release --features openblas --  -d data/STS -f glove.42B.300d.fifu > score.tsv
```

## Experimental results

We show the actual results obtained by the above procedure using `glove.42B.300d.fifu` (GloVe) or `cc.en.300.fifu` (fastText).
We also show the results obtained from the original SIF paper (Table 5 in [ICLR 2017](https://openreview.net/forum?id=SyK00v5xx)) and the SimCSE paper (Table 5 in [EMNLP 2021](https://aclanthology.org/2021.emnlp-main.552/)).



|                   | sif-embedding | sif-embedding | ICLR 2017 |       EMNLP 2021 |
| ----------------- | ------------: | ------------: | --------: | ---------------: |
| 2012              |   fastText+WR |      GloVe+WR |  GloVe+WR | SimCSE-BERT_base |
| MSRpar            |         35.4% |         39.5% |     35.6% |                  |
| MSRvid            |         84.6% |         84.1% |     83.8% |                  |
| SMTeuroparl       |         49.7% |         51.2% |     49.9% |                  |
| surprise.OnWN     |         72.7% |         71.6% |     66.2% |                  |
| surprise.SMTnews  |         54.7% |         53.4% |     45.6% |                  |
| Avg.              |         59.4% |         60.0% |     56.2% |            68.4% |
|                   |               |               |           |                  |
| 2013              |               |               |           |                  |
| FNWN              |         54.3% |         48.9% |     39.4% |                  |
| headlines         |         72.4% |         73.3% |     69.2% |                  |
| OnWN              |         84.7% |         83.6% |     82.8% |                  |
| SMT               |               |               |     37.9% |                  |
| Avg.              |               |               |     56.6% |            82.4% |
|                   |               |               |           |                  |
| 2014              |               |               |           |                  |
| deft-forum        |         46.8% |         47.8% |     41.2% |                  |
| deft-news         |         69.8% |         70.8% |     69.4% |                  |
| headlines         |         69.1% |         69.7% |     64.7% |                  |
| images            |         83.8% |         83.0% |     82.6% |                  |
| OnWN              |         85.6% |         85.3% |     82.8% |                  |
| tweet-news        |         78.4% |         77.6% |     70.1% |                  |
| Avg.              |         72.2% |         72.4% |     68.5% |            74.4% |
|                   |               |               |           |                  |
| 2015              |               |               |           |                  |
| answers-forums    |         69.5% |         69.1% |     63.9% |                  |
| answers-students  |         73.3% |         71.7% |     70.4% |                  |
| belief            |         75.6% |         75.6% |     71.8% |                  |
| headlines         |         74.1% |         75.3% |     70.7% |                  |
| images            |         82.5% |         82.8% |     81.5% |                  |
| Avg.              |         75.0% |         74.9% |     71.7% |            80.9% |
|                   |               |               |           |                  |
| 2016              |               |               |           |                  |
| answer-answer     |         55.6% |         50.4% |           |                  |
| headlines         |         72.3% |         72.7% |           |                  |
| plagiarism        |         82.4% |         81.3% |           |                  |
| postediting       |         83.0% |         80.8% |           |                  |
| question-question |         72.1% |         70.4% |           |                  |
| Avg.              |         73.1% |         71.1% |           |            78.6% |
