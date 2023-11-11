# Qdrant


```shell
$ docker pull qdrant/qdrant
$ docker run -p 6333:6333 -p 6334:6334 \
    -e QDRANT__SERVICE__GRPC_PORT="6334" \
    qdrant/qdrant
```

```shell
$ cargo run --release --bin create --features openblas-system -- -d ja.wikipedia_100k.txt -f ~/data/finalfusion/cc.ja.300.vec.fifu -o model.sif
```
