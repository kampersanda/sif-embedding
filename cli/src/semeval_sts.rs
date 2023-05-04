use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use clap::Parser;
use ndarray::Array2;
use ndarray_stats::CorrelationExt;

use sif_embedding::util;
use sif_embedding::{Float, Sif, WordEmbeddings};

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'e', long)]
    word_embedding: PathBuf,

    #[arg(short = 'w', long)]
    word_weights: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let word_embeddings = {
        let reader = BufReader::new(File::open(&args.word_embedding)?);
        WordEmbeddings::from_text(reader)?
    };
    let word_weights = {
        let reader = BufReader::new(File::open(&args.word_weights)?);
        util::word_weights_from_text(reader)?
    };

    println!("word_embeddings.len() = {}", word_embeddings.len());
    println!(
        "word_embeddings.embedding_size() = {}",
        word_embeddings.embedding_size()
    );
    println!("word_weights.len() = {}", word_weights.len());

    let corpora = vec![
        (
            "semeval-sts/2012",
            vec![
                "MSRpar.test.tsv",
                "OnWN.test.tsv",
                "SMTeuroparl.test.tsv",
                "SMTnews.test.tsv",
            ],
        ),
        (
            "semeval-sts/2013",
            vec!["FNWN.test.tsv", "headlines.test.tsv", "OnWN.test.tsv"],
        ),
        (
            "semeval-sts/2014",
            vec![
                "deft-forum.test.tsv",
                "deft-news.test.tsv",
                "headlines.test.tsv",
                "images.test.tsv",
                "OnWN.test.tsv",
                "tweet-news.test.tsv",
            ],
        ),
        (
            "semeval-sts/2015",
            vec![
                "answers-forums.test.tsv",
                "answers-students.test.tsv",
                "belief.test.tsv",
                "headlines.test.tsv",
                "images.test.tsv",
            ],
        ),
        (
            "semeval-sts/2016",
            vec![
                "answer-answer.test.tsv",
                "headlines.test.tsv",
                "plagiarism.test.tsv",
                "postediting.test.tsv",
                "question-question.test.tsv",
            ],
        ),
    ];

    let sif = Sif::new(word_embeddings, &word_weights);
    for (dir, files) in corpora {
        let mut sum_curr = 0.;
        for file in &files {
            let corr = simeval(sif.clone(), &format!("{dir}/{file}"))?;
            sum_curr += corr;
        }
        println!(
            "Average Pearson Correlation: {:.3}",
            sum_curr / files.len() as Float
        );
    }

    Ok(())
}

fn simeval(sif: Sif, curpus: &str) -> Result<Float, Box<dyn Error>> {
    println!("[{curpus}]");

    let mut gold_scores = vec![];
    let mut sentences = vec![];

    let reader = BufReader::new(File::open(curpus)?);
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        let cols: Vec<_> = line.split('\t').collect();
        assert_eq!(cols.len(), 3);
        if cols[0].is_empty() {
            // That pair was not included in the official scoring.
            continue;
        }
        gold_scores.push(
            cols[0]
                .parse::<Float>()
                .map_err(|e| format!("{e} at Line {i}: {line}"))?,
        );
        sentences.push(cols[1].to_string());
        sentences.push(cols[2].to_string());
    }
    let n_examples = gold_scores.len();

    println!("n_examples = {}", n_examples);
    println!("sentences.len() = {}", sentences.len());

    let (sent_embeddings, _) = sif.embeddings(&sentences);
    let mut pred_scores = Vec::with_capacity(sentences.len());

    for i in 0..n_examples {
        let e1 = &sent_embeddings.row(i * 2);
        let e2 = &sent_embeddings.row(i * 2 + 1);
        let score = util::cosine_similarity(e1, e2).unwrap_or(0.); // ok?
        pred_scores.push(score);
    }

    let corr = pearson_correlation(&pred_scores, &gold_scores);
    println!("{:?}", corr);

    Ok(corr)
}

fn pearson_correlation(s1: &[Float], s2: &[Float]) -> Float {
    assert_eq!(s1.len(), s2.len());
    let concat = [s1, s2].concat();
    let scores = Array2::from_shape_vec((2, s1.len()), concat).unwrap();
    let corr = scores.pearson_correlation().unwrap();
    corr[[0, 1]]
}
