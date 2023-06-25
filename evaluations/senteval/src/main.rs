#[cfg(any(feature = "intel-mkl-static", feature = "intel-mkl-system"))]
extern crate intel_mkl_src as _src;
#[cfg(any(feature = "netlib-static", feature = "netlib-system"))]
extern crate netlib_src as _src;
#[cfg(any(feature = "openblas-static", feature = "openblas-system"))]
extern crate openblas_src as _src;

use std::error::Error;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::str::FromStr;

use clap::Parser;
use finalfusion::prelude::*;
use ndarray::Array2;
use ndarray_stats::CorrelationExt;
use sif_embedding::util;
use sif_embedding::Float;
use sif_embedding::SentenceEmbedder;
use sif_embedding::Sif;
use sif_embedding::USif;
use tantivy::tokenizer::*;
use wordfreq_model::ModelKind;

#[derive(Clone, Debug)]
enum MethodKind {
    Sif,
    USif,
}

impl FromStr for MethodKind {
    type Err = &'static str;
    fn from_str(mode: &str) -> Result<Self, Self::Err> {
        match mode {
            "sif" => Ok(Self::Sif),
            "usif" => Ok(Self::USif),
            _ => Err("Could not parse a mode"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'd', long)]
    data_dir: String,

    #[arg(short = 'f', long)]
    input_fifu: String,

    #[arg(short = 'm', long, default_value = "sif")]
    method: MethodKind,

    #[arg(short = 'n', long)]
    n_components: Option<usize>,
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
            // .filter(Stemmer::new(Language::English))
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

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let word_embeddings = {
        let mut reader = BufReader::new(File::open(&args.input_fifu)?);
        Embeddings::<VocabWrap, StorageWrap>::mmap_embeddings(&mut reader)?
    };
    eprintln!("word_embeddings.len() = {}", word_embeddings.len());
    eprintln!("word_embeddings.dims() = {}", word_embeddings.dims());

    let unigram_lm = wordfreq_model::load_wordfreq(ModelKind::LargeEn)?;

    let data_dir = args.data_dir;
    let corpora = vec![
        (
            "STS12-en-test",
            vec![
                "MSRpar",
                "MSRvid",
                "SMTeuroparl",
                "surprise.OnWN",
                "surprise.SMTnews",
            ],
        ),
        ("STS13-en-test", vec!["FNWN", "headlines", "OnWN"]),
        (
            "STS14-en-test",
            vec![
                "deft-forum",
                "deft-news",
                "headlines",
                "images",
                "OnWN",
                "tweet-news",
            ],
        ),
        (
            "STS15-en-test",
            vec![
                "answers-forums",
                "answers-students",
                "belief",
                "headlines",
                "images",
            ],
        ),
        (
            "STS16-en-test",
            vec![
                "answer-answer",
                "headlines",
                "plagiarism",
                "postediting",
                "question-question",
            ],
        ),
    ];

    let mut preprocessor = Preprocessor::new();
    for (year, files) in corpora {
        println!("{year}");
        let mut corrs = vec![];
        for &file in &files {
            let gs_file = format!("{data_dir}/{year}/STS.gs.{file}.txt");
            let input_file = format!("{data_dir}/{year}/STS.input.{file}.txt");
            let (gold_scores, sentences) = load_sts_data(&gs_file, &input_file)?;
            eprintln!("file = {}, n_examples = {}", file, gold_scores.len());
            let sentences: Vec<_> = sentences.iter().map(|s| preprocessor.apply(s)).collect();
            let corr = match args.method {
                MethodKind::Sif => {
                    let mut model = Sif::new(&word_embeddings, &unigram_lm);
                    if let Some(n_components) = args.n_components {
                        model = model.n_components(n_components)?;
                    }
                    evaluate(model, &gold_scores, &sentences)?
                }
                MethodKind::USif => {
                    let mut model = USif::new(&word_embeddings, &unigram_lm);
                    if let Some(n_components) = args.n_components {
                        model = model.n_components(n_components)?;
                    }
                    evaluate(model, &gold_scores, &sentences)?
                }
            };
            corrs.push(corr);
            println!("{file}\t{corr}");
        }
        let mean = corrs.iter().sum::<Float>() / corrs.len() as Float;
        println!("Avg.\t{mean}");
        println!();
    }

    Ok(())
}

fn load_sts_data(
    gs_path: &str,
    input_path: &str,
) -> Result<(Vec<Float>, Vec<String>), Box<dyn Error>> {
    let gs_lines: Vec<String> = BufReader::new(File::open(gs_path)?)
        .lines()
        .map(|l| l.unwrap())
        .collect();
    let input_lines: Vec<String> = BufReader::new(File::open(input_path)?)
        .lines()
        .map(|l| l.unwrap())
        .collect();
    assert_eq!(gs_lines.len(), input_lines.len());

    let mut gold_scores = vec![];
    let mut sentences = vec![];
    for (gs_line, input_line) in gs_lines.iter().zip(input_lines.iter()) {
        if gs_line.is_empty() {
            continue;
        }
        gold_scores.push(gs_line.parse::<Float>()?);
        let cols: Vec<_> = input_line.split('\t').collect();
        assert_eq!(cols.len(), 2);
        sentences.push(cols[0].to_string());
        sentences.push(cols[1].to_string());
    }
    Ok((gold_scores, sentences))
}

fn evaluate<M>(
    mut model: M,
    gold_scores: &[Float],
    sentences: &[String],
) -> Result<Float, Box<dyn Error>>
where
    M: SentenceEmbedder,
{
    let sent_embeddings = model.fit_embeddings(sentences)?;
    let n_examples = gold_scores.len();
    let mut pred_scores = Vec::with_capacity(n_examples);
    for i in 0..n_examples {
        let e1 = &sent_embeddings.row(i * 2);
        let e2 = &sent_embeddings.row(i * 2 + 1);
        let score = util::cosine_similarity(e1, e2).unwrap_or(0.); // ok?
        pred_scores.push(score);
    }
    Ok(pearson_correlation(&pred_scores, gold_scores))
}

fn pearson_correlation(s1: &[Float], s2: &[Float]) -> Float {
    assert_eq!(s1.len(), s2.len());
    let concat = [s1, s2].concat();
    let scores = Array2::from_shape_vec((2, s1.len()), concat).unwrap();
    let corr = scores.pearson_correlation().unwrap();
    corr[[0, 1]]
}
