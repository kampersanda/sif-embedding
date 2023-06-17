# Livedoorニュースコーパス

ここでは、[Livedoorニュースコーパス](https://www.rondhuit.com/download.html)を用いて、日本語テキストに対する文埋め込みの評価を行います。
書籍「[BERTによる自然言語処理入門](https://www.ohmsha.co.jp/book/9784274227264/)」の第10章に倣って、文埋め込みに対する最近傍探索によるカテゴリラベルの正解率を評価します。

## 評価手順

このディレクトリにいることを前提として、評価手順を示します。

### 1. データセットのダウンロード

Livedoorニュースコーパスをダウンロードします。

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

### 2. 単語埋め込みの準備

[finalfusion](https://docs.rs/finalfusion/)形式で単語埋め込みを準備します。
ここでは、fastTextによって配布される`cc.ja.300.vec`を用います。
[ホームページ](https://fasttext.cc/docs/en/crawl-vectors.html)からモデルをダウンロードし、[finalfusion-tools](../../finalfusion-tools/)の手順に従ってfinalfusion形式に変換します。

ここでは、カレントディレクトリに`cc.ja.300.vec.fifu`があることを想定します。

### 3. Vibratoモデルの準備

`cc.ja.300.vec`は、前処理としてMeCabを使って単語分割し学習したモデルです。
ここでは同様の単語分割を再現するため、MeCab互換のRustライブラリである[Vibrato](https://github.com/daac-tools/vibrato)を使ってテキストを前処理します。

fastTextのホームページに明記はありませんが、おそらくIPADICを用いて分割したと思われるので、ここでも同じくVibratoのホームページで配されているIPADICモデルを用います。

```
$ wget https://github.com/daac-tools/vibrato/releases/download/v0.5.0/ipadic-mecab-2_7_0.tar.xz
$ tar xf ipadic-mecab-2_7_0.tar.xz
```

### 4. 評価

以下のコマンドで評価します。

```
$ cargo run --release --features openblas -- -d text -f cc.ja.300.vec.fifu -v ipadic-mecab-2_7_0/system.dic.zst
```

## 実験結果
