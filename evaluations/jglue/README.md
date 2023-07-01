# JGLUE STS Task

```
$ git clone https://github.com/yahoojapan/JGLUE.git
$ cd JGLUE
$ git checkout v1.1.0
$ cd ..
```

```
$ cargo run --release --features openblas -- \
    -d JGLUE/datasets/jsts-v1.1/valid-v1.1.json \
    -f ~/data/finalfusion/cc.ja.300.vec.fifu \
    -v ~/data/vibrato/ipadic-mecab-2_7_0/system.dic.zst \
    -m sif
```
