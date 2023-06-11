# Legal case document similarity

Here, we provide a tool to evaluate this library on [Legal case document similarity](https://arxiv.org/abs/2209.12474).

## Evaluation steps

We show steps to run the evaluation, assuming you are at this directory.

### 1. Download Legal case document similarity datasets

Run the following commands:

```shell
$ git clone https://github.com/Law-AI/document-similarity.git
$ unzip document-similarity/test.zip
```

We will use the data under the `test` directory.

```shell
$ ls test
documents  similarity_scores.csv
```

### 2. Prepare pretrained word embeddings

You need to prepare pretrained word embeddings in [finalfusion](https://docs.rs/finalfusion/) format.
Prepare a model following [finalfusion-tools/README.md](../../finalfusion-tools/README.md).

Here, we assume that you have `glove.42B.300d.fifu` in the current directory.

### 3. Evaluate

Run the following command:

```shell
$ cargo run --release -- -d test -f ~/data/finalfusion/glove.42B.300d.fifu -o score.tsv
```

This commnad will report the similarity results in the three metrics as with the paper:

- Pearson correlation coefficient
- Mean Squared Error (MSE)
- F-Score

## Experimental results

We show the actual results obtained from the above procedure using `glove.42B.300d.fifu` (GloVe) or `cc.en.300.fifu` (fastText).
We also show the results of Doc2Vec obtained from the original paper.

| Method   | Correlation |   MSE | FScore |
| -------- | ----------: | ----: | -----: |
| GloVe    |      -0.053 | 0.086 |  0.579 |
| fastText |      -0.020 | 0.085 |  0.583 |
| Doc2Vec  |       0.701 | 0.036 |  0.682 |

Basically, the performance of SIF was poor.
Especially, there was no correlation between SIF scores and gold scores.
This may suggest the importance of domain-adaptive pretraining.
I'm interested in the results with word embeddings trained in the Legal domain.
