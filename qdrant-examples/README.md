# Example usage of sif-embedding + Qdrant

This directory provides an example usage of sif-embedding + [qdrant/rust-client](https://github.com/qdrant/rust-client).

## Preparation

### Run Qdrant server

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

The following is an example output:

```text
...
Enter a sentence to search (or empty to exit):
> UNIX is basically a simple operating system, but you have to be a genius to understand the simplicity.
search_result = SearchResponse {
    result: [
        ScoredPoint {
            ...
                        StringValue(
                            "The Monitor Call API was very much ahead of its time, like most of the operating system, and made system programming on DECsystem-10s simple and powerful.",
                        ),
            ...
            score: 0.73194635,
            ...
        },
        ScoredPoint {
            ...
                        StringValue(
                            "It was a simple, efficient system, very effective primarily because of its simplicity.",
                        ),
            ...
            score: 0.69869876,
            ...
        },
        ScoredPoint {
            ...
                        StringValue(
                            "Erzya has a simple five-vowel system.",
                        ),
            ...
            score: 0.68330246,
            ...
        },
        ScoredPoint {
            ...
                        StringValue(
                            "True BASIC.“ Upon Kemeny's advice, True BASIC was not limited to a single OS or computer system.",
                        ),
            ...
            score: 0.6798639,
            ...
        },
        ScoredPoint {
            ...
                        StringValue(
                            "An open-loop controller is often used in simple processes because of its simplicity and low cost, especially in systems where feedback is not critical.",
                        ),
            ...
            score: 0.6794525,
            ...
        },
    ],
    time: 0.0214875,
}
Enter a sentence to search (or empty to exit):
>
...
```
