# sif-embedding

Smooth inverse frequency

## Evaluation

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

## Links

https://github.com/PrincetonML/SIF
https://github.com/brmson/dataset-sts
