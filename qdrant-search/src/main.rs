#[cfg(any(feature = "intel-mkl-static", feature = "intel-mkl-system"))]
extern crate intel_mkl_src as _src;
#[cfg(any(feature = "netlib-static", feature = "netlib-system"))]
extern crate netlib_src as _src;
#[cfg(any(feature = "openblas-static", feature = "openblas-system"))]
extern crate openblas_src as _src;

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use finalfusion::prelude::*;
use ndarray::Axis;
use qdrant_client::prelude::*;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::VectorParams;
use qdrant_client::qdrant::VectorsConfig;
use rand::prelude::*;
use serde_json::json;
use sif_embedding::SentenceEmbedder;
use sif_embedding::Sif;
use sif_embedding::WordProbabilities;
use wordfreq_model::ModelKind;

const BATCH_SIZE: usize = 1 << 16;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'd', long)]
    dataset_file: PathBuf,

    #[arg(short = 'f', long)]
    fifu_model: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let sentences = load_wiki_article_dataset(&args.dataset_file)?;
    let sentences = sentences[..10000].to_vec();
    eprintln!("Loaded {} sentences", sentences.len());

    let mut reader = BufReader::new(File::open(&args.fifu_model)?);
    let word_embeddings = Embeddings::<VocabWrap, StorageWrap>::mmap_embeddings(&mut reader)?;
    eprintln!("word_embeddings.len() = {}", word_embeddings.len());
    eprintln!("word_embeddings.dims() = {}", word_embeddings.dims());

    let unigram_lm = wordfreq_model::load_wordfreq(ModelKind::LargeJa)?;
    eprintln!("unigram_lm.n_words() = {}", unigram_lm.n_words());

    let model = Sif::new(&word_embeddings, &unigram_lm);
    let model = model.fit(&sentences)?;

    let client = QdrantClient::from_url("http://localhost:6334").build()?;
    let collection_name = "wiki-article-dataset";
    client.delete_collection(collection_name).await?;

    // 4. Upload embeddings
    let collection = CreateCollection {
        collection_name: collection_name.into(),
        vectors_config: Some(VectorsConfig {
            config: Some(Config::Params(VectorParams {
                size: model.embedding_size() as u64,
                distance: Distance::Cosine.into(),
                ..Default::default()
            })),
        }),
        ..Default::default()
    };
    client.create_collection(&collection).await?;

    let n_batches = if sentences.len() % BATCH_SIZE == 0 {
        sentences.len() / BATCH_SIZE
    } else {
        sentences.len() / BATCH_SIZE + 1
    };

    let mut id = 0;
    for (i, batch) in sentences.chunks(BATCH_SIZE).enumerate() {
        eprintln!("Uploading batch {}/{}", i + 1, n_batches);

        let sent_embeddings = model.embeddings(batch)?;
        let mut points = vec![];
        for (embedding, sentence) in sent_embeddings.axis_iter(Axis(0)).zip(batch.iter()) {
            let payload: Payload = json!({"sentence": sentence}).try_into().unwrap();
            points.push(PointStruct::new(id as u64, embedding.to_vec(), payload));
            id += 1;
        }
        client
            .upsert_points_blocking(collection_name, points, None)
            .await?;
    }

    // Search
    let idx = thread_rng().gen_range(0..sentences.len());
    let sent_embedding = model.embeddings(&sentences[idx..idx + 1])?;
    let search_point = SearchPoints {
        collection_name: collection_name.into(),
        vector: sent_embedding.row(0).to_vec(),
        limit: 4, // Top3 + itself
        with_payload: Some(true.into()),
        ..Default::default()
    };
    let search_result = client.search_points(&search_point).await?;
    println!("search_result = {:#?}", search_result);

    Ok(())
}

/// https://github.com/Hironsan/wiki-article-dataset
fn load_wiki_article_dataset<P: AsRef<Path>>(dataset_file: P) -> Result<Vec<String>> {
    let file = File::open(dataset_file)?;
    let reader = BufReader::new(file);
    let mut sentences = vec![];
    for line in reader.lines() {
        let line = line?;
        sentences.extend(line.split('\t').map(|s| s.to_string()));
    }
    Ok(sentences)
}
