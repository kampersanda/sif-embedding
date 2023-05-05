# sif-embedding

Smooth inverse frequency

## Evaluation

This repository provides an easy tool to evaluate this library using Semantic Text Similarity

```
$ git clone https://github.com/brmson/dataset-sts.git
$ ln -s dataset-sts/data/sts/semeval-sts
```

```
$ python -m venv venv
$ . ./venv/bin/activate
$ pip install -r scripts/requirements.txt
$ python scripts/clean_semeval_sts.py
```

```
$ wget https://nlp.stanford.edu/data/glove.840B.300d.zip
$ unzip glove.840B.300d.zip
```

```
$ cargo run --release -p cli --bin semeval_sts -- -e path/to/glove.840B.300d.txt -w path/to/enwiki_vocab_min200.txt -c path/to/semeval-sts-clean/all
```

| Dataset                    | ICLR 2017 | `sif_embedding::Sif` |
| -------------------------- | --------- | -------------------- |
| 2012                       |           |                      |
| MSRpar.test.tsv            | 35.6%     | 21.9%                |
| OnWN.test.tsv              | 66.2%     | 66.2%                |
| SMTeuroparl.test.tsv       | 49.9%     | 50.3%                |
| SMTnews.test.tsv           | 45.6%     | 48.7%                |
| 2013                       |           |                      |
| FNWN.test.tsv              | 39.4%     | 40.5%                |
| headlines.test.tsv         | 69.2%     | 70.4%                |
| OnWN.test.tsv              | 82.8%     | 80.1%                |
| 2014                       |           |                      |
| deft-forum.test.tsv        | 41.2%     | 41.1%                |
| deft-news.test.tsv         | 69.4%     | 69.3%                |
| headlines.test.tsv         | 64.7%     | 65.5%                |
| images.test.tsv            | 82.6%     | 82.9%                |
| OnWN.test.tsv              | 82.8%     | 83.1%                |
| 2015                       |           |                      |
| answers-forums.test.tsv    | 63.9%     | 63.9%                |
| answers-students.test.tsv  | 70.4%     | 70.7%                |
| belief.test.tsv            | 71.8%     | 72.5%                |
| headlines.test.tsv         | 70.7%     | 73.5%                |
| images.test.tsv            | 81.5%     | 81.5%                |
| 2016                       |           |                      |
| answer-answer.test.tsv     | NA        | 51.9%                |
| headlines.test.tsv         | NA        | 69.7%                |
| plagiarism.test.tsv        | NA        | 79.4%                |
| postediting.test.tsv       | NA        | 79.4%                |
| question-question.test.tsv | NA        | 69.6%                |

## Links

https://github.com/PrincetonML/SIF
https://github.com/brmson/dataset-sts
