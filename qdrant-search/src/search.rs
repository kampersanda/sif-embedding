#[cfg(any(feature = "intel-mkl-static", feature = "intel-mkl-system"))]
extern crate intel_mkl_src as _src;
#[cfg(any(feature = "netlib-static", feature = "netlib-system"))]
extern crate netlib_src as _src;
#[cfg(any(feature = "openblas-static", feature = "openblas-system"))]
extern crate openblas_src as _src;

use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use finalfusion::prelude::*;
use qdrant_client::prelude::*;
use regex::Regex;
use sif_embedding::SentenceEmbedder;
use sif_embedding::Sif;
use sif_embedding::WordProbabilities;
use wordfreq_model::ModelKind;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 's', long)]
    sif_model: PathBuf,

    #[arg(short = 'f', long)]
    fifu_model: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let mut reader = BufReader::new(File::open(&args.fifu_model)?);
    let word_embeddings = Embeddings::<VocabWrap, StorageWrap>::mmap_embeddings(&mut reader)?;
    eprintln!("word_embeddings.len() = {}", word_embeddings.len());
    eprintln!("word_embeddings.dims() = {}", word_embeddings.dims());

    let unigram_lm = wordfreq_model::load_wordfreq(ModelKind::LargeJa)?;
    eprintln!("unigram_lm.n_words() = {}", unigram_lm.n_words());

    let mut data = vec![];
    File::open(&args.sif_model)?.read_to_end(&mut data)?;
    let model = Sif::deserialize(&data, &word_embeddings, &unigram_lm)?;

    let client = QdrantClient::from_url("http://localhost:6334").build()?;
    let collection_name = "wiki-article-dataset";

    let re = Regex::new(r"\s+").unwrap();

    loop {
        println!("単語間に空白を入れてクエリ文を入力して下さい");

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();
        if input.is_empty() {
            break;
        }

        let input = re
            .split(input)
            .collect::<Vec<_>>()
            .join(&sif_embedding::DEFAULT_SEPARATOR.to_string());
        let sent_embedding = model.embeddings([input])?;
        let search_point = SearchPoints {
            collection_name: collection_name.into(),
            vector: sent_embedding.row(0).to_vec(),
            limit: 5,
            with_payload: Some(true.into()),
            ..Default::default()
        };
        let search_result = client.search_points(&search_point).await?;
        println!("search_result = {:#?}", search_result);
    }

    Ok(())
}
