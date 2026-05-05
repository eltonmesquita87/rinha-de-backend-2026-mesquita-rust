use rinha_fraude_vetorial::{load_references_json_gz, save_references_bin};
use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    let input =
        flag_value(&args, "--input").unwrap_or_else(|| PathBuf::from("data/references.json.gz"));
    let output =
        flag_value(&args, "--output").unwrap_or_else(|| PathBuf::from("data/references.bin"));

    let dataset = load_references_json_gz(&input)?;
    save_references_bin(&output, &dataset)?;

    eprintln!("wrote {} records to {}", dataset.len(), output.display());

    Ok(())
}

fn flag_value(args: &[String], flag: &str) -> Option<PathBuf> {
    args.windows(2)
        .find(|window| window[0] == flag)
        .map(|window| PathBuf::from(&window[1]))
}
