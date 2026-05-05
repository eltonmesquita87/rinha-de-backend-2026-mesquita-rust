use rinha_fraude_vetorial::{load_references_json_gz, save_references_bin};
use std::error::Error;
use std::path::PathBuf;

const DEFAULT_PER_CLASS: usize = 15_000;
const DEFAULT_SEED: u64 = 0xC0FF_EEDE_ADBE_EF42;

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    let input = flag_str(&args, "--input")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/references.json.gz"));
    let output = flag_str(&args, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/references.bin"));
    let per_class: usize = flag_str(&args, "--max-per-class")
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_PER_CLASS);
    let seed: u64 = flag_str(&args, "--seed")
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_SEED);

    let full = load_references_json_gz(&input)?;
    eprintln!("loaded {} reference records from {}", full.len(), input.display());

    let sampled = if per_class == 0 {
        full
    } else {
        let sampled = full.stratified_subsample(per_class, seed);
        eprintln!(
            "stratified subsample to {} records (per_class={}, seed=0x{:x})",
            sampled.len(),
            per_class,
            seed
        );
        sampled
    };

    save_references_bin(&output, &sampled)?;
    eprintln!("wrote {} records to {}", sampled.len(), output.display());

    Ok(())
}

fn flag_str<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|window| window[0] == flag)
        .map(|window| window[1].as_str())
}
