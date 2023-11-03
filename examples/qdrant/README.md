# Qdrant

```shell
$ wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
$ tar xf ldcc-20140209.tar.gz
$ ls -1 text
CHANGES.txt
dokujo-tsushin
it-life-hack
kaden-channel
livedoor-homme
movie-enter
peachy
README.txt
smax
sports-watch
topic-news
```

```shell
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 \
    -e QDRANT__SERVICE__GRPC_PORT="6334" \
    qdrant/qdrant
```

https://github.com/qdrant/rust-client

https://dev.classmethod.jp/articles/qdrant-first-step/
