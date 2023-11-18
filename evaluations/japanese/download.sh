#! /bin/bash

set -eux

mkdir ./data

# Download JGLUE
git clone https://github.com/yahoojapan/JGLUE.git
mv ./JGLUE/datasets/jsts-v1.1 ./data/jsts
rm -rf ./JGLUE

# Download JSICK
git clone git@github.com:verypluming/JSICK.git
mv ./JSICK/jsick ./data/jsick
rm -rf ./JSICK
