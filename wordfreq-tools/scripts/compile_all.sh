#!/bin/bash

set -eux

data_dir="wordfreq/wordfreq/data/"

targets=(
    "large_ar"
    "large_bn"
    "large_ca"
    "large_cs"
    "large_de"
    "large_en"
    "large_es"
    "large_fi"
    "large_fr"
    "large_he"
    "large_it"
    "large_ja"
    "large_mk"
    "large_nb"
    "large_nl"
    "large_pl"
    "large_pt"
    "large_ru"
    "large_sv"
    "large_uk"
    "large_zh"
)

output_dir="output"
mkdir ${output_dir}

for target in "${targets[@]}" ; do
    echo "Compiling ${target}"
    python scripts/parse_msgpack.py ${data_dir}/${target}.msgpack.gz > ${output_dir}/${target}.txt
    cargo run --release -- -i ${output_dir}/${target}.txt -o ${output_dir}/${target}.unigram
done
