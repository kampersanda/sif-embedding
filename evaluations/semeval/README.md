# SemEval on sif-embedding

Here, we provide a tool to evaluate this library on [SemEval STS Task](https://aclanthology.org/S16-1081/).
We aim to reproduce the experiments at [ICLR 2017](https://openreview.net/forum?id=SyK00v5xx),
but some parts (such as pre-processing) have not been fully reproduced.

## Evaluation steps

We show steps to run the evaluation, assuming you are at directory `sif-embedding/evaluations/semeval`.

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

### 3. Prepare pretrained word embeddings

You need to prepare pretrained word embeddings in [finalfusion](https://docs.rs/finalfusion/) format.
Prepare a model following [finalfusion-tools/README.md](../finalfusion-tools/README.md).

Here, we assume that you have `glove.42B.300d.fifu` in the current directory.

### 4. Conduct evaluation

`cli/semeval_sts` is a command line tool that evaluates `sif_embedding::Sif` using the SemEval STS datasets.
The SIF algorithm requires unigram probabilities.
You can use `auxiliary_data/enwiki_vocab_min200.txt` that has word frequencies (copied from [the authors' repository](https://github.com/PrincetonML/SIF)).

```shell
$ cargo run --release -- -f glove.42B.300d.fifu -w auxiliary_data/enwiki_vocab_min200.txt -c semeval-sts-clean/all -c semeval-sts-clean/all > scores.txt
```

This will report the Pearson correlation coefficient between estimated similarities
(i.e., cosine similarity between sentence embeddings) and gold scores, following the evaluation metric of the task.

## Experimental results

The following table shows the actual results obtained from the above procedure.
The original results by the authors are also shown as a baseline, from Table 5 (GloVe+WR) in [ICLR 2017](https://openreview.net/forum?id=SyK00v5xx).

|                            | ICLR 2017 | sif_embedding | sif_embedding |
|----------------------------|----------:|--------------:|--------------:|
| 2012                       |  GloVe+WR |      GloVe+WR |   fastText+WR |
| MSRpar.test.tsv            |     35.6% |         25.6% |         25.3% |
| OnWN.test.tsv              |     66.2% |         67.3% |         66.3% |
| SMTeuroparl.test.tsv       |     49.9% |         49.7% |         52.3% |
| SMTnews.test.tsv           |     45.6% |         47.5% |         46.1% |
|                            |           |               |               |
| 2013                       |           |               |               |
| FNWN.test.tsv              |     39.4% |         42.8% |         46.7% |
| headlines.test.tsv         |     69.2% |         72.0% |         69.0% |
| OnWN.test.tsv              |     82.8% |         79.7% |         79.2% |
|                            |           |               |               |
| 2014                       |           |               |               |
| deft-forum.test.tsv        |     41.2% |         40.1% |         39.9% |
| deft-news.test.tsv         |     69.4% |         72.1% |         69.5% |
| headlines.test.tsv         |     64.7% |         66.7% |         64.6% |
| images.test.tsv            |     82.6% |         82.2% |         82.7% |
| OnWN.test.tsv              |     82.8% |         82.6% |         81.6% |
|                            |           |               |               |
| 2015                       |           |               |               |
| answers-forums.test.tsv    |     63.9% |         64.5% |         63.9% |
| answers-students.test.tsv  |     70.4% |         72.3% |         72.6% |
| belief.test.tsv            |     71.8% |         73.3% |         70.7% |
| headlines.test.tsv         |     70.7% |         73.6% |         71.4% |
| images.test.tsv            |     81.5% |         82.0% |         80.8% |
|                            |           |               |               |
| 2016                       |           |               |               |
| answer-answer.test.tsv     |           |         51.3% |         54.0% |
| headlines.test.tsv         |           |         70.5% |         70.7% |
| plagiarism.test.tsv        |           |         79.8% |         80.2% |
| postediting.test.tsv       |           |         81.1% |         81.4% |
| question-question.test.tsv |           |         69.0% |         69.7% |

This library is not an exact port of the original code, and the experimental results do not exactly match.
However, similar results were obtained (except for `2012.MSRpar.test.tsv`).
