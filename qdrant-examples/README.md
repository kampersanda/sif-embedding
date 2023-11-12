# Example usage of sif-embedding + qdrant


```shell
$ docker pull qdrant/qdrant
$ docker run -p 6333:6333 -p 6334:6334 \
    -e QDRANT__SERVICE__GRPC_PORT="6334" \
    qdrant/qdrant
```

```shell
$ wget https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt
```

```shell
$ cargo run --release --bin create --features openblas-system -- -d ja.wikipedia_100k.txt -f ~/data/finalfusion/cc.ja.300.vec.fifu -o model.sif
```
