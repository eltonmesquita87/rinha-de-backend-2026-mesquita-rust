use chrono::{DateTime, Datelike, Timelike, Utc};
use flate2::read::GzDecoder;
use serde::de::{SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

pub const VECTOR_DIMENSIONS: usize = 14;
pub const TOP_K: usize = 5;
pub const QUANTIZATION_SCALE: f32 = 10_000.0;
const BINARY_MAGIC: &[u8; 8] = b"RFVEC01\0";
const DEFAULT_REFERENCE_HINT: usize = 3_000_000;

pub type MccRisk = HashMap<String, f32>;
pub type BoxError = Box<dyn Error + Send + Sync>;

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Normalization {
    pub max_amount: f32,
    pub max_installments: f32,
    pub amount_vs_avg_ratio: f32,
    pub max_minutes: f32,
    pub max_km: f32,
    pub max_tx_count_24h: f32,
    pub max_merchant_avg_amount: f32,
}

impl Default for Normalization {
    fn default() -> Self {
        Self {
            max_amount: 10_000.0,
            max_installments: 12.0,
            amount_vs_avg_ratio: 10.0,
            max_minutes: 1_440.0,
            max_km: 1_000.0,
            max_tx_count_24h: 20.0,
            max_merchant_avg_amount: 10_000.0,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct FraudRequest {
    pub transaction: Transaction,
    pub customer: Customer,
    pub merchant: Merchant,
    pub terminal: Terminal,
    #[serde(default)]
    pub last_transaction: Option<LastTransaction>,
}

#[derive(Debug, Deserialize)]
pub struct Transaction {
    pub amount: f32,
    pub installments: f32,
    pub requested_at: String,
}

#[derive(Debug, Deserialize)]
pub struct Customer {
    pub avg_amount: f32,
    pub tx_count_24h: f32,
    pub known_merchants: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct Merchant {
    pub id: String,
    pub mcc: String,
    pub avg_amount: f32,
}

#[derive(Debug, Deserialize)]
pub struct Terminal {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f32,
}

#[derive(Debug, Deserialize)]
pub struct LastTransaction {
    pub timestamp: String,
    pub km_from_current: f32,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct FraudResponse {
    pub approved: bool,
    pub fraud_score: f32,
}

#[derive(Debug)]
pub struct VectorizeError {
    field: &'static str,
    value: String,
}

impl VectorizeError {
    fn invalid_timestamp(field: &'static str, value: &str) -> Self {
        Self {
            field,
            value: value.to_owned(),
        }
    }
}

impl fmt::Display for VectorizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid timestamp in {}: {}", self.field, self.value)
    }
}

impl Error for VectorizeError {}

#[derive(Debug)]
pub struct ReferenceDataset {
    vectors: Vec<i16>,
    labels: Vec<u8>,
}

impl ReferenceDataset {
    pub fn with_capacity(records: usize) -> Self {
        Self {
            vectors: Vec::with_capacity(records.saturating_mul(VECTOR_DIMENSIONS)),
            labels: Vec::with_capacity(records),
        }
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn memory_bytes(&self) -> usize {
        self.vectors.len() * std::mem::size_of::<i16>() + self.labels.len()
    }

    pub fn push_float_vector(&mut self, vector: &[f32; VECTOR_DIMENSIONS], label: u8) {
        let quantized = quantize_vector(vector);
        self.push_quantized_vector(&quantized, label);
    }

    pub fn push_quantized_vector(&mut self, vector: &[i16; VECTOR_DIMENSIONS], label: u8) {
        self.vectors.extend_from_slice(vector);
        self.labels.push(u8::from(label != 0));
    }

    pub fn top5_labels(&self, query: &[i16; VECTOR_DIMENSIONS]) -> Option<[u8; TOP_K]> {
        if self.labels.len() < TOP_K {
            return None;
        }

        let mut top_distances = [i64::MAX; TOP_K];
        let mut top_labels = [0u8; TOP_K];
        let mut worst_index = 0usize;
        let mut worst_distance = i64::MAX;

        let labels = self.labels.as_slice();
        for (record_index, candidate) in
            self.vectors.chunks_exact(VECTOR_DIMENSIONS).enumerate()
        {
            let distance = squared_distance_i16(query, candidate);

            if distance < worst_distance {
                top_distances[worst_index] = distance;
                top_labels[worst_index] = labels[record_index];

                worst_index = 0;
                worst_distance = top_distances[0];
                for slot in 1..TOP_K {
                    if top_distances[slot] > worst_distance {
                        worst_index = slot;
                        worst_distance = top_distances[slot];
                    }
                }
            }
        }

        Some(top_labels)
    }

    pub fn fraud_count_top5(&self, query: &[i16; VECTOR_DIMENSIONS]) -> Option<u8> {
        self.top5_labels(query)
            .map(|labels| labels.into_iter().map(|label| u8::from(label != 0)).sum())
    }

    pub fn stratified_subsample(&self, per_class: usize, seed: u64) -> Self {
        let mut fraud_indices = Vec::new();
        let mut legit_indices = Vec::new();
        for (index, &label) in self.labels.iter().enumerate() {
            if label != 0 {
                fraud_indices.push(index);
            } else {
                legit_indices.push(index);
            }
        }

        let mut rng = SplitMix64::new(seed);
        partial_fisher_yates(&mut fraud_indices, per_class, &mut rng);
        partial_fisher_yates(&mut legit_indices, per_class, &mut rng);

        let take_fraud = fraud_indices.len().min(per_class);
        let take_legit = legit_indices.len().min(per_class);
        let total = take_fraud + take_legit;

        let mut output = Self::with_capacity(total);
        for &source in fraud_indices[..take_fraud]
            .iter()
            .chain(legit_indices[..take_legit].iter())
        {
            let offset = source * VECTOR_DIMENSIONS;
            let mut vector = [0i16; VECTOR_DIMENSIONS];
            vector.copy_from_slice(&self.vectors[offset..offset + VECTOR_DIMENSIONS]);
            output.push_quantized_vector(&vector, self.labels[source]);
        }
        output
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn partial_fisher_yates(indices: &mut [usize], take: usize, rng: &mut SplitMix64) {
    let take = take.min(indices.len());
    for slot in 0..take {
        let remaining = (indices.len() - slot) as u64;
        let pick = slot + (rng.next_u64() % remaining) as usize;
        indices.swap(slot, pick);
    }
}

impl<'de> Deserialize<'de> for ReferenceDataset {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(ReferenceDatasetVisitor)
    }
}

struct ReferenceDatasetVisitor;

impl<'de> Visitor<'de> for ReferenceDatasetVisitor {
    type Value = ReferenceDataset;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON array of reference vectors")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut dataset = ReferenceDataset::with_capacity(
            seq.size_hint().unwrap_or(DEFAULT_REFERENCE_HINT),
        );

        while let Some(item) = seq.next_element::<ReferenceItem>()? {
            dataset.push_float_vector(&item.vector, item.label.as_u8());
        }

        Ok(dataset)
    }
}

#[derive(Debug, Deserialize)]
struct ReferenceItem {
    vector: [f32; VECTOR_DIMENSIONS],
    label: ReferenceLabel,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ReferenceLabel {
    Fraud,
    Legit,
}

impl ReferenceLabel {
    fn as_u8(&self) -> u8 {
        match self {
            Self::Fraud => 1,
            Self::Legit => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResourcePaths {
    pub references_bin: PathBuf,
    pub references_json_gz: PathBuf,
    pub mcc_risk: PathBuf,
    pub normalization: PathBuf,
}

impl ResourcePaths {
    pub fn from_env() -> Self {
        let data_dir = std::env::var_os("DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("data"));
        Self::from_data_dir(data_dir)
    }

    pub fn from_data_dir(data_dir: impl Into<PathBuf>) -> Self {
        let data_dir = data_dir.into();
        Self {
            references_bin: data_dir.join("references.bin"),
            references_json_gz: data_dir.join("references.json.gz"),
            mcc_risk: data_dir.join("mcc_risk.json"),
            normalization: data_dir.join("normalization.json"),
        }
    }
}

#[derive(Debug)]
pub struct FraudEngine {
    dataset: ReferenceDataset,
    normalization: Normalization,
    mcc_risk: MccRisk,
}

impl FraudEngine {
    pub fn load(paths: &ResourcePaths) -> Result<Self, BoxError> {
        let normalization = load_normalization(&paths.normalization)?;
        let mcc_risk = load_mcc_risk(&paths.mcc_risk)?;
        let dataset = if paths.references_bin.exists() {
            load_references_bin(&paths.references_bin)?
        } else {
            load_references_json_gz(&paths.references_json_gz)?
        };

        if dataset.len() < TOP_K {
            return Err(format!(
                "reference dataset must contain at least {} records, found {}",
                TOP_K,
                dataset.len()
            )
            .into());
        }

        Ok(Self {
            dataset,
            normalization,
            mcc_risk,
        })
    }

    pub fn from_parts(
        dataset: ReferenceDataset,
        normalization: Normalization,
        mcc_risk: MccRisk,
    ) -> Self {
        Self {
            dataset,
            normalization,
            mcc_risk,
        }
    }

    pub fn reference_count(&self) -> usize {
        self.dataset.len()
    }

    pub fn reference_memory_bytes(&self) -> usize {
        self.dataset.memory_bytes()
    }

    pub fn score(&self, request: &FraudRequest) -> Result<FraudResponse, VectorizeError> {
        let vector = vectorize_transaction(request, &self.mcc_risk, &self.normalization)?;
        let query = quantize_vector(&vector);
        let fraud_count = self
            .dataset
            .fraud_count_top5(&query)
            .expect("dataset is validated with at least TOP_K records");
        Ok(decision_from_fraud_count(fraud_count))
    }
}

pub fn load_normalization(path: impl AsRef<Path>) -> Result<Normalization, BoxError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    Ok(serde_json::from_reader(reader)?)
}

pub fn load_mcc_risk(path: impl AsRef<Path>) -> Result<MccRisk, BoxError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut risks: MccRisk = serde_json::from_reader(reader)?;
    for value in risks.values_mut() {
        *value = clamp01(*value);
    }
    Ok(risks)
}

pub fn load_references_json_gz(path: impl AsRef<Path>) -> Result<ReferenceDataset, BoxError> {
    let file = File::open(path)?;
    let decoder = GzDecoder::new(file);
    let reader = BufReader::with_capacity(1024 * 1024, decoder);
    let mut deserializer = serde_json::Deserializer::from_reader(reader);
    Ok(ReferenceDataset::deserialize(&mut deserializer)?)
}

pub fn load_references_bin(path: impl AsRef<Path>) -> Result<ReferenceDataset, BoxError> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);

    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != BINARY_MAGIC {
        return Err("invalid references.bin magic header".into());
    }

    let mut count_bytes = [0u8; 8];
    reader.read_exact(&mut count_bytes)?;
    let count = u64::from_le_bytes(count_bytes);
    if count > (usize::MAX / VECTOR_DIMENSIONS) as u64 {
        return Err("references.bin record count is too large for this platform".into());
    }

    let count = count as usize;
    let mut dataset = ReferenceDataset::with_capacity(count);
    let mut vector_bytes = [0u8; VECTOR_DIMENSIONS * 2];
    let mut label = [0u8; 1];

    for _ in 0..count {
        reader.read_exact(&mut vector_bytes)?;
        let mut vector = [0i16; VECTOR_DIMENSIONS];
        for dimension in 0..VECTOR_DIMENSIONS {
            let offset = dimension * 2;
            vector[dimension] =
                i16::from_le_bytes([vector_bytes[offset], vector_bytes[offset + 1]]);
        }

        reader.read_exact(&mut label)?;
        dataset.push_quantized_vector(&vector, label[0]);
    }

    Ok(dataset)
}

pub fn save_references_bin(
    path: impl AsRef<Path>,
    dataset: &ReferenceDataset,
) -> Result<(), BoxError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, file);

    writer.write_all(BINARY_MAGIC)?;
    writer.write_all(&(dataset.len() as u64).to_le_bytes())?;

    for record_index in 0..dataset.len() {
        let offset = record_index * VECTOR_DIMENSIONS;
        for value in &dataset.vectors[offset..offset + VECTOR_DIMENSIONS] {
            writer.write_all(&value.to_le_bytes())?;
        }
        writer.write_all(&[dataset.labels[record_index]])?;
    }

    writer.flush()?;
    Ok(())
}

pub fn clamp01(value: f32) -> f32 {
    if !value.is_finite() || value <= 0.0 {
        0.0
    } else if value >= 1.0 {
        1.0
    } else {
        value
    }
}

pub fn parse_utc_timestamp(value: &str) -> Result<DateTime<Utc>, VectorizeError> {
    DateTime::parse_from_rfc3339(value)
        .map(|datetime| datetime.with_timezone(&Utc))
        .map_err(|_| VectorizeError::invalid_timestamp("timestamp", value))
}

pub fn hour_utc(value: &str) -> Result<u32, VectorizeError> {
    Ok(parse_utc_timestamp(value)?.hour())
}

pub fn day_of_week_monday0(value: &str) -> Result<u32, VectorizeError> {
    Ok(parse_utc_timestamp(value)?.weekday().num_days_from_monday())
}

pub fn vectorize_transaction(
    request: &FraudRequest,
    mcc_risk: &MccRisk,
    normalization: &Normalization,
) -> Result<[f32; VECTOR_DIMENSIONS], VectorizeError> {
    let requested_at = DateTime::parse_from_rfc3339(&request.transaction.requested_at)
        .map(|datetime| datetime.with_timezone(&Utc))
        .map_err(|_| {
            VectorizeError::invalid_timestamp(
                "transaction.requested_at",
                &request.transaction.requested_at,
            )
        })?;

    let (minutes_since_last, km_from_last_tx) =
        if let Some(last_transaction) = &request.last_transaction {
            let last_at = DateTime::parse_from_rfc3339(&last_transaction.timestamp)
                .map(|datetime| datetime.with_timezone(&Utc))
                .map_err(|_| {
                    VectorizeError::invalid_timestamp(
                        "last_transaction.timestamp",
                        &last_transaction.timestamp,
                    )
                })?;
            let minutes = requested_at.signed_duration_since(last_at).num_seconds() as f32 / 60.0;
            (
                normalize_positive(minutes, normalization.max_minutes),
                normalize_positive(last_transaction.km_from_current, normalization.max_km),
            )
        } else {
            (-1.0, -1.0)
        };

    let amount = request.transaction.amount;
    let customer_avg = request.customer.avg_amount;
    let amount_vs_avg = if !amount.is_finite() || amount <= 0.0 {
        0.0
    } else if !customer_avg.is_finite()
        || customer_avg <= 0.0
        || !normalization.amount_vs_avg_ratio.is_finite()
        || normalization.amount_vs_avg_ratio <= 0.0
    {
        1.0
    } else {
        clamp01((amount / customer_avg) / normalization.amount_vs_avg_ratio)
    };

    let unknown_merchant = if request
        .customer
        .known_merchants
        .iter()
        .any(|known| known == &request.merchant.id)
    {
        0.0
    } else {
        1.0
    };

    let mcc_risk = mcc_risk
        .get(&request.merchant.mcc)
        .copied()
        .map(clamp01)
        .unwrap_or(0.5);

    Ok([
        normalize_positive(amount, normalization.max_amount),
        normalize_positive(
            request.transaction.installments,
            normalization.max_installments,
        ),
        amount_vs_avg,
        requested_at.hour() as f32 / 23.0,
        requested_at.weekday().num_days_from_monday() as f32 / 6.0,
        minutes_since_last,
        km_from_last_tx,
        normalize_positive(request.terminal.km_from_home, normalization.max_km),
        normalize_positive(
            request.customer.tx_count_24h,
            normalization.max_tx_count_24h,
        ),
        f32::from(request.terminal.is_online),
        f32::from(request.terminal.card_present),
        unknown_merchant,
        mcc_risk,
        normalize_positive(
            request.merchant.avg_amount,
            normalization.max_merchant_avg_amount,
        ),
    ])
}

pub fn quantize_value(value: f32) -> i16 {
    let bounded = if value.is_finite() {
        value.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    (bounded * QUANTIZATION_SCALE).round() as i16
}

pub fn quantize_vector(vector: &[f32; VECTOR_DIMENSIONS]) -> [i16; VECTOR_DIMENSIONS] {
    let mut output = [0i16; VECTOR_DIMENSIONS];
    for dimension in 0..VECTOR_DIMENSIONS {
        output[dimension] = quantize_value(vector[dimension]);
    }
    output
}

pub fn squared_distance_i16(query: &[i16; VECTOR_DIMENSIONS], candidate: &[i16]) -> i64 {
    debug_assert_eq!(candidate.len(), VECTOR_DIMENSIONS);
    let mut distance: i64 = 0;
    for dimension in 0..VECTOR_DIMENSIONS {
        let delta = query[dimension] as i32 - candidate[dimension] as i32;
        distance += (delta * delta) as i64;
    }
    distance
}

pub fn fraud_score_from_count(fraud_count: u8) -> f32 {
    match fraud_count {
        0 => 0.0,
        1 => 0.2,
        2 => 0.4,
        3 => 0.6,
        4 => 0.8,
        _ => 1.0,
    }
}

pub fn approved_from_score(fraud_score: f32) -> bool {
    fraud_score < 0.6
}

pub fn decision_from_fraud_count(fraud_count: u8) -> FraudResponse {
    let fraud_score = fraud_score_from_count(fraud_count);
    FraudResponse {
        approved: approved_from_score(fraud_score),
        fraud_score,
    }
}

fn normalize_positive(value: f32, max: f32) -> f32 {
    if !max.is_finite() || max <= 0.0 {
        return 0.0;
    }
    clamp01(value / max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_request(
        last_transaction: Option<LastTransaction>,
        merchant_id: &str,
        mcc: &str,
    ) -> FraudRequest {
        FraudRequest {
            transaction: Transaction {
                amount: 384.88,
                installments: 3.0,
                requested_at: "2026-03-11T20:23:35Z".to_owned(),
            },
            customer: Customer {
                avg_amount: 769.76,
                tx_count_24h: 3.0,
                known_merchants: vec![
                    "MERC-009".to_owned(),
                    "MERC-001".to_owned(),
                    "MERC-001".to_owned(),
                ],
            },
            merchant: Merchant {
                id: merchant_id.to_owned(),
                mcc: mcc.to_owned(),
                avg_amount: 298.95,
            },
            terminal: Terminal {
                is_online: false,
                card_present: true,
                km_from_home: 13.709_052,
            },
            last_transaction,
        }
    }

    fn sample_risks() -> MccRisk {
        HashMap::from([("5912".to_owned(), 0.7)])
    }

    fn approx_eq(left: f32, right: f32) {
        assert!(
            (left - right).abs() < 0.000_01,
            "left={left}, right={right}"
        );
    }

    #[test]
    fn clamp_handles_bounds_and_invalid_numbers() {
        approx_eq(clamp01(-0.2), 0.0);
        approx_eq(clamp01(0.4), 0.4);
        approx_eq(clamp01(2.0), 1.0);
        approx_eq(clamp01(f32::NAN), 0.0);
    }

    #[test]
    fn extracts_utc_hour() {
        assert_eq!(hour_utc("2026-03-11T20:23:35Z").unwrap(), 20);
    }

    #[test]
    fn calculates_day_of_week_with_monday_as_zero() {
        assert_eq!(day_of_week_monday0("2026-03-11T20:23:35Z").unwrap(), 2);
    }

    #[test]
    fn last_transaction_null_uses_negative_one_sentinel() {
        let request = sample_request(None, "MERC-001", "5912");
        let vector =
            vectorize_transaction(&request, &sample_risks(), &Normalization::default()).unwrap();

        approx_eq(vector[5], -1.0);
        approx_eq(vector[6], -1.0);
    }

    #[test]
    fn unknown_merchant_is_inverted_flag() {
        let known = sample_request(None, "MERC-001", "5912");
        let unknown = sample_request(None, "MERC-999", "5912");

        let known_vector =
            vectorize_transaction(&known, &sample_risks(), &Normalization::default()).unwrap();
        let unknown_vector =
            vectorize_transaction(&unknown, &sample_risks(), &Normalization::default()).unwrap();

        approx_eq(known_vector[11], 0.0);
        approx_eq(unknown_vector[11], 1.0);
    }

    #[test]
    fn unknown_mcc_uses_half_risk_fallback() {
        let request = sample_request(None, "MERC-001", "0000");
        let vector =
            vectorize_transaction(&request, &sample_risks(), &Normalization::default()).unwrap();

        approx_eq(vector[12], 0.5);
    }

    #[test]
    fn vectorizes_full_known_example() {
        let request = sample_request(
            Some(LastTransaction {
                timestamp: "2026-03-11T14:58:35Z".to_owned(),
                km_from_current: 18.862_648,
            }),
            "MERC-001",
            "5912",
        );
        let vector =
            vectorize_transaction(&request, &sample_risks(), &Normalization::default()).unwrap();

        approx_eq(vector[0], 0.038_488);
        approx_eq(vector[1], 0.25);
        approx_eq(vector[2], 0.05);
        approx_eq(vector[3], 20.0 / 23.0);
        approx_eq(vector[4], 2.0 / 6.0);
        approx_eq(vector[5], 325.0 / 1440.0);
        approx_eq(vector[6], 0.018_862_648);
        approx_eq(vector[7], 0.013_709_052);
        approx_eq(vector[8], 0.15);
        approx_eq(vector[9], 0.0);
        approx_eq(vector[10], 1.0);
        approx_eq(vector[11], 0.0);
        approx_eq(vector[12], 0.7);
        approx_eq(vector[13], 0.029_895);
    }

    #[test]
    fn calculates_fraud_score_from_top5_count() {
        approx_eq(fraud_score_from_count(0), 0.0);
        approx_eq(fraud_score_from_count(3), 0.6);
        approx_eq(fraud_score_from_count(5), 1.0);
    }

    #[test]
    fn approval_requires_score_below_point_six() {
        assert!(approved_from_score(0.4));
        assert!(!approved_from_score(0.6));
        assert!(!approved_from_score(0.8));
    }

    #[test]
    fn sentinel_negative_one_dominates_distance_to_zero_query() {
        let mut sentinel_vector = [0.5f32; VECTOR_DIMENSIONS];
        sentinel_vector[5] = -1.0;
        sentinel_vector[6] = -1.0;

        let quantized = quantize_vector(&sentinel_vector);
        assert_eq!(quantized[5], -10_000);
        assert_eq!(quantized[6], -10_000);

        let query = quantize_vector(&[0.5f32; VECTOR_DIMENSIONS]);
        let distance = squared_distance_i16(&query, &quantized);
        // sentinela na q (5000) vs sentinela no candidato (-10000) -> delta 15000 nas dims 5 e 6.
        let expected = 2 * (15_000i64 * 15_000i64);
        assert_eq!(distance, expected);
    }

    #[test]
    fn stratified_subsample_balances_classes_and_is_deterministic() {
        let mut dataset = ReferenceDataset::with_capacity(1_000);
        for index in 0..600 {
            let mut vector = [0.0; VECTOR_DIMENSIONS];
            vector[0] = (index as f32) / 600.0;
            dataset.push_float_vector(&vector, 1);
        }
        for index in 0..400 {
            let mut vector = [0.0; VECTOR_DIMENSIONS];
            vector[0] = (index as f32) / 400.0;
            dataset.push_float_vector(&vector, 0);
        }

        let first = dataset.stratified_subsample(50, 42);
        let second = dataset.stratified_subsample(50, 42);
        let third = dataset.stratified_subsample(50, 99);

        assert_eq!(first.len(), 100);
        let fraud_count: usize =
            first.labels.iter().map(|&label| (label != 0) as usize).sum();
        assert_eq!(fraud_count, 50);
        assert_eq!(first.vectors, second.vectors);
        assert_ne!(first.vectors, third.vectors);
    }

    #[test]
    fn stratified_subsample_caps_at_available_records_per_class() {
        let mut dataset = ReferenceDataset::with_capacity(20);
        for _ in 0..5 {
            dataset.push_float_vector(&[0.1; VECTOR_DIMENSIONS], 1);
        }
        for _ in 0..15 {
            dataset.push_float_vector(&[0.2; VECTOR_DIMENSIONS], 0);
        }

        let sampled = dataset.stratified_subsample(10, 7);
        assert_eq!(sampled.len(), 5 + 10);
    }

    #[test]
    fn top5_neighbors_are_kept_without_full_sort() {
        let mut dataset = ReferenceDataset::with_capacity(6);
        for (value, label) in [
            (0.90, 0),
            (0.40, 1),
            (0.10, 1),
            (0.30, 0),
            (0.20, 1),
            (0.50, 0),
        ] {
            let mut vector = [0.0; VECTOR_DIMENSIONS];
            vector[0] = value;
            dataset.push_float_vector(&vector, label);
        }

        let query = quantize_vector(&[0.0; VECTOR_DIMENSIONS]);
        let fraud_count = dataset.fraud_count_top5(&query).unwrap();

        assert_eq!(fraud_count, 3);
        assert_eq!(
            decision_from_fraud_count(fraud_count),
            FraudResponse {
                approved: false,
                fraud_score: 0.6
            }
        );
    }
}
