# Livedoorニュースコーパス

ここでは、[Livedoorニュースコーパス](https://www.rondhuit.com/download.html)を用いて日本語テキストに対する文埋め込みの評価を行います。
書籍「[BERTによる自然言語処理入門](https://www.ohmsha.co.jp/book/9784274227264/)」の第10章に倣って、文埋め込みについてのコサイン類似度を用いた最近傍探索により、ニュースカテゴリラベルの一致での正解率を評価します。

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

上記のコマンドで実際に得られた結果を示します。
また比較対象として、以下の文埋め込みを使って得られた結果も示します。

- 同じ単語埋め込み（`cc.ja.300.vec`）の平均を用いて得た文埋め込み
- GensimのDoc2Vecを用いて得た文埋め込み（Livedoorニュースコーパスから自己教師あり学習）
- BERT（`cl-tohoku/bert-base-japanese-whole-word-masking`）で、全サブワードに対応する最終層の隠れ状態ベクトルの平均値プーリングを用いて得た文埋め込み

これら結果は、同様の実験をしている[このブログ記事](https://kampersanda.hatenablog.jp/entry/2023/01/02/155106)から拝借しました。

結果は以下の通りです。
改めてですがここで示す結果は、各ニュース記事について文埋め込みのコサイン類似度が最も大きくなる他のニュース記事とカテゴリか一致するかを評価し、その正解率を算出したものです。


| 手法             | 正解率 |
| ---------------- | ------ |
| sif-embedding    | 86.7%  |
| 単語埋め込み平均 | 83.5%  |
| Doc2Vec          | 86.9%  |
| BERT             | 83.2%  |

Livedoorニュースコーパスから自己教師あり学習されたDoc2Vecが最も高いスコアとなっています。
僅差でsif-embeddingが続きます。
ナイーブな単語埋め込み平均では3ポイントほど低いスコアとなっており、SIFの重み付けの効果が日本語でも確認できました。
