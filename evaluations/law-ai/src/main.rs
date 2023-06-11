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

struct Preprocessor {
    analyzer: tantivy::tokenizer::TextAnalyzer,
}

impl Preprocessor {
    fn new() -> Self {
        let analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(RemoveLongFilter::limit(40))
            .filter(LowerCaser)
            .filter(StopWordFilter::new(Language::English).unwrap())
            .filter(Stemmer::new(Language::English))
            .build();
        Self { analyzer }
    }

    fn apply(&mut self, text: &str) -> String {
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

    let mut analyzer = Preprocessor::new();
    let sent_embeddings = sif.embeddings(dataset.document_files.iter().map(|file| {
        let mut reader = BufReader::new(File::open(file).unwrap());
        let mut text = String::new();
        reader.read_to_string(&mut text).unwrap();
        analyzer.apply(&text)
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
    let mse = mean_squared_error(&pred_scores, &gold_scores);
    println!("Mean squared error: {}", mse);
    let f1 = f_score(&pred_scores, &gold_scores);
    println!("F1 score: {}", f1);

    Ok(())
}

fn pearson_correlation(s1: &[f32], s2: &[f32]) -> f32 {
    assert_eq!(s1.len(), s2.len());
    let concat = [s1, s2].concat();
    let scores = Array2::from_shape_vec((2, s1.len()), concat).unwrap();
    let corr = scores.pearson_correlation().unwrap();
    corr[[0, 1]]
}

fn mean_squared_error(s1: &[f32], s2: &[f32]) -> f32 {
    assert_eq!(s1.len(), s2.len());
    let mut sum = 0.;
    for (x, y) in s1.iter().zip(s2.iter()) {
        sum += (x - y).powi(2);
    }
    sum / s1.len() as f32
}

fn f_score(pred_scores: &[f32], gold_scores: &[f32]) -> f32 {
    assert_eq!(pred_scores.len(), gold_scores.len());
    let mut true_positives = 0.;
    let mut false_positives = 0.0;
    let mut false_negatives = 0.0;

    for (&pred, &gold) in pred_scores.iter().zip(gold_scores) {
        let pred_label = pred > 0.5;
        let gold_label = gold > 0.5;
        if pred_label && gold_label {
            true_positives += 1.0;
        } else if pred_label && !gold_label {
            false_positives += 1.0;
        } else if !pred_label && gold_label {
            false_negatives += 1.0;
        }
    }

    let precision = if true_positives + false_positives != 0.0 {
        true_positives / (true_positives + false_positives)
    } else {
        0.0
    };

    let recall = if true_positives + false_negatives != 0.0 {
        true_positives / (true_positives + false_negatives)
    } else {
        0.0
    };

    if precision + recall != 0.0 {
        2.0 * (precision * recall) / (precision + recall)
    } else {
        0.0
    }
}
