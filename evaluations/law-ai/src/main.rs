extern crate openblas_src;

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use clap::Parser;
use finalfusion::prelude::*;
use ndarray::Array2;
use ndarray_stats::CorrelationExt;
use sif_embedding::Sif;
use tantivy::tokenizer::*;
use wordfreq_model::{self, ModelKind};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'f', long)]
    input_fifu: PathBuf,

    #[arg(short = 'd', long)]
    data_dir: PathBuf,
}

struct EnglishAnalyzer {
    analyzer: tantivy::tokenizer::TextAnalyzer,
}

impl EnglishAnalyzer {
    fn new() -> Self {
        let en_stem = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(RemoveLongFilter::limit(40))
            .filter(LowerCaser)
            .build();
        Self { analyzer: en_stem }
    }

    fn analyze(&mut self, text: &str) -> String {
        let mut tokens = self.analyzer.token_stream(text);
        let mut token_texts = Vec::new();
        while let Some(tok) = tokens.next() {
            token_texts.push(tok.text.to_string());
        }
        token_texts.join(" ")
    }
}

struct DatasetHandler {
    document_files: Vec<PathBuf>,
    gold_scores: Vec<(usize, usize, f32)>,
}

impl DatasetHandler {
    fn new(data_dir: &Path) -> Self {
        let data_dir = data_dir.to_str().unwrap();
        let document_files = glob::glob(&format!("{data_dir}/documents/*.txt"))
            .unwrap()
            .filter_map(Result::ok)
            .collect::<Vec<_>>();

        let file_name_map = document_files
            .iter()
            .enumerate()
            .map(|(i, file)| (file.file_stem().unwrap().to_str().unwrap().to_string(), i))
            .collect::<HashMap<_, _>>();

        let mut gold_scores = vec![];
        let similarity_scores_reader =
            BufReader::new(File::open(&format!("{data_dir}/similarity_scores.csv")).unwrap());
        for line in similarity_scores_reader.lines() {
            let line = line.unwrap();
            let mut cols = line.split(",");
            let file_id_1 = file_name_map[cols.next().unwrap()];
            let file_id_2 = file_name_map[cols.next().unwrap()];
            let score = cols.next().unwrap().parse::<f32>().unwrap();
            gold_scores.push((file_id_1, file_id_2, score));
        }

        Self {
            document_files,
            gold_scores,
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let dataset = DatasetHandler::new(&args.data_dir);

    let word_embeddings = {
        let mut reader = BufReader::new(File::open(&args.input_fifu)?);
        Embeddings::<VocabWrap, StorageWrap>::mmap_embeddings(&mut reader)?
    };
    let unigram_lm = wordfreq_model::load_wordfreq(ModelKind::LargeEn)?;
    let sif = Sif::new(&word_embeddings, &unigram_lm);

    let mut analyzer = EnglishAnalyzer::new();
    let sent_embeddings = sif.embeddings(dataset.document_files.iter().map(|file| {
        let mut reader = BufReader::new(File::open(file).unwrap());
        let mut text = String::new();
        reader.read_to_string(&mut text).unwrap();
        analyzer.analyze(&text)
    }));

    let mut pred_scores = vec![];
    let mut gold_scores = vec![];

    for &(file_id_1, file_id_2, gold_score) in &dataset.gold_scores {
        let e1 = &sent_embeddings.row(file_id_1);
        let e2 = &sent_embeddings.row(file_id_2);
        let pred_score = sif_embedding::util::cosine_similarity(e1, e2).unwrap_or(0.);
        let pred_score = (pred_score + 1.) / 2.; // normalized to [0,1]
        pred_scores.push(pred_score);
        gold_scores.push(gold_score);
    }

    let r = pearson_correlation(&pred_scores, &gold_scores);
    println!("Pearson correlation: {}", r);

    Ok(())
}

fn pearson_correlation(s1: &[f32], s2: &[f32]) -> f32 {
    assert_eq!(s1.len(), s2.len());
    let concat = [s1, s2].concat();
    let scores = Array2::from_shape_vec((2, s1.len()), concat).unwrap();
    let corr = scores.pearson_correlation().unwrap();
    corr[[0, 1]]
}
