# Example usage of sif-embedding + qdrant

This directory provides an example usage of sif-embedding + [qdrant/rust-client](https://github.com/qdrant/rust-client).

## Preparation

### Run qdrant server

Run a Qdrant server enabling gRPC port with the following commands:

```shell
$ docker pull qdrant/qdrant
$ docker run -p 6333:6333 -p 6334:6334 \
    -e QDRANT__SERVICE__GRPC_PORT="6334" \
    qdrant/qdrant
```

### Download dataset

This example uses the [wiki1m dataset](https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse) for the database.
Run the following command:

```shell
$ wget https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt
```

### Prepare pretrained word embeddings

You need to prepare pretrained word embeddings in [finalfusion](https://docs.rs/finalfusion/) format.
Prepare a model following [finalfusion-tools](../finalfusion-tools).

Here, we assume that you have `glove.42B.300d.fifu` in the current directory.

## Example usage

### Indexing

`src/create.rs` is an example code for indexing the dataset.

The following command indexes the dataset in Qdrant and saves the SIF model to `model.sif`:

```shell
$ cargo run --release --bin create --features openblas -- \
    -d ja.wikipedia_100k.txt \
    -f glove.42B.300d.fifu \
    -o model.sif
```

### Querying

`src/search.rs` is an example code for querying.

The following command provides a simple querying demo using the index and the SIF model:

```shell
$ cargo run --release --bin search --features openblas -- \
    -i model.sif \
    -f glove.42B.300d.fifu
```
