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
use sif_embedding::util;
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
        // NOTE: Since senteval.tar provided at https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse
        // has been preprocessed (such as lowercasing), we only need simple tokenization for punctuation.
        let analyzer = TextAnalyzer::builder(SimpleTokenizer::default()).build();
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
        (
            "STS13-en-test",
            vec![
                "FNWN",
                "headlines",
                "OnWN",
                // This dataset is not provided due to the license issue.
                // "SMT",
            ],
        ),
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
        let mut pearsons = vec![];
        let mut spearmans = vec![];
        println!("year\tfile\tpearson\tspearman");
        for &file in &files {
            let gs_file = format!("{data_dir}/{year}/STS.gs.{file}.txt");
            let input_file = format!("{data_dir}/{year}/STS.input.{file}.txt");
            let (gold_scores, sentences) = load_sts_data(&gs_file, &input_file)?;
            eprintln!("file = {}, n_examples = {}", file, gold_scores.len());
            let sentences: Vec<_> = sentences.iter().map(|s| preprocessor.apply(s)).collect();
            let (pearson, spearman) = match args.method {
                MethodKind::Sif => {
                    let param_a = sif_embedding::sif::DEFAULT_PARAM_A;
                    let n_components = args
                        .n_components
                        .unwrap_or(sif_embedding::sif::DEFAULT_N_COMPONENTS);
                    let model =
                        Sif::with_parameters(&word_embeddings, &unigram_lm, param_a, n_components)?;
                    evaluate(model, &gold_scores, &sentences)?
                }
                MethodKind::USif => {
                    let n_components = args
                        .n_components
                        .unwrap_or(sif_embedding::usif::DEFAULT_N_COMPONENTS);
                    let model = USif::with_parameters(&word_embeddings, &unigram_lm, n_components);
                    evaluate(model, &gold_scores, &sentences)?
                }
            };
            pearsons.push(pearson);
            spearmans.push(spearman);
            println!("{year}\t{file}\t{pearson}\t{spearman}");
        }
        let mean_pearson = pearsons.iter().sum::<f64>() / pearsons.len() as f64;
        let mean_spearman = spearmans.iter().sum::<f64>() / spearmans.len() as f64;
        println!("{year}\tAvg.\t{mean_pearson}\t{mean_spearman}");
        println!("")
    }

    Ok(())
}

fn load_sts_data(
    gs_path: &str,
    input_path: &str,
) -> Result<(Vec<f64>, Vec<String>), Box<dyn Error>> {
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
        gold_scores.push(gs_line.parse::<f64>()?);
        let cols: Vec<_> = input_line.split('\t').collect();
        assert_eq!(cols.len(), 2);
        sentences.push(cols[0].to_string());
        sentences.push(cols[1].to_string());
    }
    Ok((gold_scores, sentences))
}

fn evaluate<M>(
    model: M,
    gold_scores: &[f64],
    sentences: &[String],
) -> Result<(f64, f64), Box<dyn Error>>
where
    M: SentenceEmbedder,
{
    let (sent_embeddings, _) = model.fit_embeddings(sentences)?;
    let n_examples = gold_scores.len();
    let mut pred_scores = Vec::with_capacity(n_examples);
    for i in 0..n_examples {
        let e1 = &sent_embeddings.row(i * 2);
        let e2 = &sent_embeddings.row(i * 2 + 1);
        let score = util::cosine_similarity(e1, e2).unwrap_or(-1.) as f64; // ok?
        pred_scores.push(score);
    }
    let pearson = pearson_correlation(&pred_scores, gold_scores);
    let spearman = spearman_correlation(&pred_scores, gold_scores);
    Ok((pearson, spearman))
}

fn pearson_correlation(s1: &[f64], s2: &[f64]) -> f64 {
    assert_eq!(s1.len(), s2.len());
    rgsl::statistics::correlation(s1, 1, s2, 1, s1.len())
}

fn spearman_correlation(s1: &[f64], s2: &[f64]) -> f64 {
    assert_eq!(s1.len(), s2.len());
    let mut work = Vec::with_capacity(2 * s1.len());
    rgsl::statistics::spearman(s1, 1, s2, 1, s1.len(), &mut work)
}
