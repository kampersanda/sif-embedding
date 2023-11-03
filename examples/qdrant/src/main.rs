#[cfg(any(feature = "intel-mkl-static", feature = "intel-mkl-system"))]
extern crate intel_mkl_src as _src;
#[cfg(any(feature = "netlib-static", feature = "netlib-system"))]
extern crate netlib_src as _src;
#[cfg(any(feature = "openblas-static", feature = "openblas-system"))]
extern crate openblas_src as _src;

use std::cell::RefCell;
use std::error::Error;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;

use clap::Parser;
use finalfusion::prelude::*;
use qdrant_client::prelude::*;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::VectorParams;
use qdrant_client::qdrant::VectorsConfig;
use sif_embedding::SentenceEmbedder;
use sif_embedding::USif;
use sif_embedding::WordProbabilities;
use unicode_normalization::UnicodeNormalization;
use vibrato::dictionary::Dictionary;
use vibrato::tokenizer::worker::Worker as VibratoWorker;
use vibrato::Tokenizer;
use wordfreq_model::ModelKind;

const CATEGORIES: &[&str] = &[
    "dokujo-tsushin",
    "it-life-hack",
    "kaden-channel",
    "livedoor-homme",
    "movie-enter",
    "peachy",
    "smax",
    "sports-watch",
    "topic-news",
];

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'd', long)]
    dataset_dir: PathBuf,

    #[arg(short = 'f', long)]
    fifu_model: PathBuf,

    #[arg(short = 'v', long)]
    vibrato_model: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // 1. Load dataset
    let (sentences, labels) = load_livedoor_data(&args.dataset_dir)?;
    eprintln!("{} sentences", sentences.len());
    eprintln!("{} labels", labels.len());

    // 2. Tokenize sentences
    let tokenizer = {
        let reader = zstd::Decoder::new(File::open(args.vibrato_model)?)?;
        let dict = Dictionary::read(reader)?;
        Tokenizer::new(dict)
            .ignore_space(true)?
            .max_grouping_len(24)
    };
    let worker = RefCell::new(tokenizer.new_worker());
    let tokenized: Vec<String> = sentences.iter().map(|s| tokenize(s, &worker)).collect();

    // 3. Load models
    let word_embeddings = {
        let mut reader = BufReader::new(File::open(&args.fifu_model)?);
        Embeddings::<VocabWrap, StorageWrap>::mmap_embeddings(&mut reader)?
    };
    eprintln!("word_embeddings.len() = {}", word_embeddings.len());
    eprintln!("word_embeddings.dims() = {}", word_embeddings.dims());

    let unigram_lm = wordfreq_model::load_wordfreq(ModelKind::LargeJa)?;
    eprintln!("unigram_lm.n_words() = {}", unigram_lm.n_words());

    let model = USif::new(&word_embeddings, &unigram_lm);
    let (sent_embeddings, model) = model.fit_embeddings(&tokenized)?;
    eprintln!("sent_embeddings.shape() = {:?}", sent_embeddings.shape());

    // 4. Upload embeddings
    let client = QdrantClient::from_url("http://localhost:6334").build()?;
    let collection_name = "livedoor";
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

    let collection_info = client.collection_info(collection_name).await?;
    dbg!(collection_info);

    Ok(())
}

fn load_livedoor_data<P: AsRef<Path>>(
    data_dir: P,
) -> Result<(Vec<String>, Vec<usize>), Box<dyn Error>> {
    let data_dir = data_dir.as_ref().to_str().unwrap();

    let mut sentences = vec![];
    let mut labels = vec![];
    for (label, &cate) in CATEGORIES.iter().enumerate() {
        for filepath in glob::glob(&format!("{data_dir}/{cate}/{cate}-*.txt"))? {
            let filepath = filepath?;
            let sentence = read_sentence(&filepath)?;
            sentences.push(sentence);
            labels.push(label);
        }
    }
    Ok((sentences, labels))
}

fn read_sentence<P: AsRef<Path>>(filepath: P) -> Result<String, Box<dyn Error>> {
    let file = File::open(filepath)?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();
    // Skip the first three lines for the header.
    Ok(lines[3..].join("\n"))
}

fn tokenize(sentence: &str, worker: &RefCell<VibratoWorker>) -> String {
    let mut surfaces = vec![];
    for line in sentence.split('\n') {
        let line = line.nfkc().collect::<String>();
        let mut worker = worker.borrow_mut();
        worker.reset_sentence(line);
        worker.tokenize();
        surfaces.extend(worker.token_iter().map(|t| t.surface().to_string()));
    }
    surfaces.join(" ")
}
