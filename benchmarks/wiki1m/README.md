# Speed benchmark using wiki1m dataset

This directory provides speed benchmark using the [wiki1m dataset](https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse).

## Evaluation steps

### 1. Download dataset

Run the following commands:

```shell
$ wget https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt
```

### 2. Prepare pretrained word embeddings

You need to prepare pretrained word embeddings in [finalfusion](https://docs.rs/finalfusion/) format.
Prepare a model following [finalfusion-tools](../../finalfusion-tools).

Here, we assume that you have `glove.42B.300d.fifu` in the current directory.

### 3. Evaluate

`src/main.rs` provides evaluation for SIF (with `-m sif`) and uSIF (with `-m usif`).

```shell
$ cargo run --release --features openblas -- \
    -d wiki1m_for_simcse.txt \
    -f glove.42B.300d.fifu \
    -m sif
```

## Evaluation results


