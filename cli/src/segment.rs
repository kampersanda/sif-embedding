use tokenizers::tokenizer::{Result, Tokenizer};

fn main() -> Result<()> {
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None)?;

    let encoding = tokenizer.encode("The problem likely will mean corrective changes before the shuttle fleet starts flying again.", false)?;
    println!("{:?}", encoding.get_tokens());

    Ok(())
}
