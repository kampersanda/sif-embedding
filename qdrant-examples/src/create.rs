#[cfg(any(feature = "intel-mkl-static", feature = "intel-mkl-system"))]
extern crate intel_mkl_src as _src;
#[cfg(any(feature = "netlib-static", feature = "netlib-system"))]
extern crate netlib_src as _src;
#[cfg(any(feature = "openblas-static", feature = "openblas-system"))]
extern crate openblas_src as _src;

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use finalfusion::prelude::*;
use ndarray::Axis;
use qdrant_client::prelude::*;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::OptimizersConfigDiff;
use qdrant_client::qdrant::VectorParams;
use qdrant_client::qdrant::VectorsConfig;
use serde_json::json;
use sif_embedding::SentenceEmbedder;
use sif_embedding::Sif;
use sif_embedding::WordProbabilities;
use vtext::tokenize::Tokenizer;
use vtext::tokenize::VTextTokenizerParams;
use wordfreq_model::ModelKind;

const BATCH_SIZE: usize = 10000;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'd', long)]
    dataset_file: PathBuf,

    #[arg(short = 'f', long)]
    fifu_model: PathBuf,

    #[arg(short = 'o', long)]
    output_model: PathBuf,

    #[arg(short = 'b', long)]
    batch_size: Option<usize>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let (orig_sentences, proc_sentences) = load_wiki1m_dataset(&args.dataset_file)?;
    eprintln!("Number of loaded sentences: {}", orig_sentences.len());

    let mut reader = BufReader::new(File::open(&args.fifu_model)?);
    let word_embeddings = Embeddings::<VocabWrap, StorageWrap>::mmap_embeddings(&mut reader)?;
    eprintln!("word_embeddings.len() = {}", word_embeddings.len());
    eprintln!("word_embeddings.dims() = {}", word_embeddings.dims());

    let unigram_lm = wordfreq_model::load_wordfreq(ModelKind::LargeEn)?;
    eprintln!("unigram_lm.n_words() = {}", unigram_lm.n_words());

    let model = Sif::new(&word_embeddings, &unigram_lm);
    let model = model.fit(&proc_sentences)?;

    let client = QdrantClient::from_url("http://localhost:6334").build()?;
    let collection_name = "wiki1m";
    client.delete_collection(collection_name).await?;

    let collection = CreateCollection {
        collection_name: collection_name.into(),
        vectors_config: Some(VectorsConfig {
            config: Some(Config::Params(VectorParams {
                size: model.embedding_size() as u64,
                distance: Distance::Cosine.into(),
                ..Default::default()
            })),
        }),
        // Disable indexing during upload for speed up.
        // https://qdrant.tech/documentation/tutorials/bulk-upload/
        optimizers_config: Some(OptimizersConfigDiff {
            indexing_threshold: Some(0),
            ..Default::default()
        }),
        ..Default::default()
    };
    client.create_collection(&collection).await?;

    let start = Instant::now();
    {
        let batch_size = args.batch_size.unwrap_or(BATCH_SIZE);
        let n_batches = if proc_sentences.len() % batch_size == 0 {
            proc_sentences.len() / batch_size
        } else {
            proc_sentences.len() / batch_size + 1
        };

        let mut id = 0;
        for (i, (orig_batch, proc_batch)) in orig_sentences
            .chunks(batch_size)
            .zip(proc_sentences.chunks(batch_size))
            .enumerate()
        {
            eprintln!("Uploading batch {}/{}", i + 1, n_batches);

            let sent_embeddings = model.embeddings(proc_batch)?;
            let mut points = vec![];
            for (embedding, sentence) in sent_embeddings.axis_iter(Axis(0)).zip(orig_batch.iter()) {
                let payload: Payload = json!({"sentence": sentence}).try_into().unwrap();
                points.push(PointStruct::new(id as u64, embedding.to_vec(), payload));
                id += 1;
            }
            client.upsert_points(collection_name, points, None).await?;
        }

        // Post-create the index.
        // https://qdrant.tech/documentation/tutorials/bulk-upload/
        client
            .update_collection(
                collection_name,
                &OptimizersConfigDiff {
                    indexing_threshold: Some(20000),
                    ..Default::default()
                },
            )
            .await?;
    }
    let elapsed = start.elapsed();
    eprintln!("Indexing time: {:?}", elapsed);

    let data = model.serialize()?;
    let mut model = File::create(&args.output_model)?;
    model.write_all(&data)?;

    Ok(())
}

/// https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse
fn load_wiki1m_dataset<P: AsRef<Path>>(dataset_file: P) -> Result<(Vec<String>, Vec<String>)> {
    let reader = BufReader::new(File::open(dataset_file)?);
    let tokenizer = VTextTokenizerParams::default().lang("en").build()?;
    let separator = sif_embedding::DEFAULT_SEPARATOR.to_string();

    let mut orig_sentences = vec![];
    let mut proc_sentences = vec![];

    for line in reader.lines() {
        let orig_sentence = line?;
        let proc_sentence = tokenizer
            .tokenize(&orig_sentence)
            .collect::<Vec<_>>()
            .join(&separator)
            .to_lowercase();
        orig_sentences.push(orig_sentence);
        proc_sentences.push(proc_sentence);
    }

    Ok((orig_sentences, proc_sentences))
}
