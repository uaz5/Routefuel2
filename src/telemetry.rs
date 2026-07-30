use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;
use thiserror::Error;

// ============================================================================
// ERROR HANDLING
// ============================================================================

#[derive(Error, Debug)]
pub enum TelemetryError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::error::Error),

    #[error("Invalid path: {0}")]
    InvalidPath(String),

    #[error("Buffer full - telemetry dropped")]
    BufferFull,
}

pub type TelemetryResult<T> = Result<T, TelemetryError>;

// ============================================================================
// TELEMETRY DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryData {
    /// Unique request identifier
    pub request_id: String,

    /// Timestamp (Unix seconds)
    pub timestamp: i64,

    /// The model that was chosen by the router
    pub chosen_model: String,

    /// The fallback/shadow model (if any)
    pub shadow_model: Option<String>,

    /// Latency of the chosen model in milliseconds
    pub latency_ms: u64,

    /// Latency of shadow model (if tested)
    pub shadow_latency_ms: Option<u64>,

    /// Cost of chosen model in cents
    pub chosen_cost: f64,

    /// Cost of baseline model in cents
    pub baseline_cost: f64,

    /// Cost delta (baseline - chosen) in cents
    pub cost_delta: f64,

    /// Savings percentage
    pub savings_percentage: f64,

    /// Request priority
    pub priority: String,

    /// User tier
    pub user_tier: String,

    /// Was failover triggered
    pub failover_triggered: bool,

    /// Provider used
    pub provider: String,

    /// Success flag
    pub success: bool,

    /// Error message if applicable
    pub error: Option<String>,

    /// Tokens used
    pub tokens_used: u32,

    /// Client ID (from API key)
    pub client_id: String,
}

impl TelemetryData {
    pub fn new(
        request_id: String,
        chosen_model: String,
        chosen_cost: f64,
        baseline_cost: f64,
        client_id: String,
    ) -> Self {
        let cost_delta = baseline_cost - chosen_cost;
        let savings_percentage = if baseline_cost > 0.0 {
            (cost_delta / baseline_cost) * 100.0
        } else {
            0.0
        };

        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        Self {
            request_id,
            timestamp,
            chosen_model,
            shadow_model: None,
            latency_ms: 0,
            shadow_latency_ms: None,
            chosen_cost,
            baseline_cost,
            cost_delta,
            savings_percentage,
            priority: String::new(),
            user_tier: String::new(),
            failover_triggered: false,
            provider: String::new(),
            success: false,
            error: None,
            tokens_used: 0,
            client_id,
        }
    }
}

// ============================================================================
// ROI REPORT GENERATION
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct RoiReport {
    pub period_start: i64,
    pub period_end: i64,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_cost_cents: f64,
    pub total_savings_cents: f64,
    pub average_latency_ms: u64,
    pub top_models: Vec<ModelUsage>,
    pub cost_breakdown_by_tier: Vec<TierBreakdown>,
    pub roi_percentage: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelUsage {
    pub model_name: String,
    pub usage_count: u64,
    pub total_cost: f64,
    pub total_savings: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TierBreakdown {
    pub tier: String,
    pub request_count: u64,
    pub total_cost: f64,
    pub average_cost_per_request: f64,
}

// ============================================================================
// TELEMETRY RECORDER
// ============================================================================

pub struct TelemetryRecorder {
    /// Output directory for telemetry files
    output_dir: String,

    /// In-memory buffer for batching writes
    buffer: Arc<RwLock<Vec<TelemetryData>>>,

    /// Maximum buffer size before forced flush
    max_buffer_size: usize,

    /// Enable telemetry recording
    enabled: bool,
}

impl TelemetryRecorder {
    pub fn new(output_dir: &str, max_buffer_size: usize) -> TelemetryResult<Self> {
        // Verify directory exists or create it
        let path = Path::new(output_dir);
        if !path.exists() {
            std::fs::create_dir_all(path)?;
        }

        Ok(Self {
            output_dir: output_dir.to_string(),
            buffer: Arc::new(RwLock::new(Vec::with_capacity(max_buffer_size))),
            max_buffer_size,
            enabled: true,
        })
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Record a telemetry event
    pub async fn record(&self, data: TelemetryData) -> TelemetryResult<()> {
        if !self.enabled {
            return Ok(());
        }

        let mut buffer = self.buffer.write().await;

        buffer.push(data);

        // Auto-flush if buffer is full
        if buffer.len() >= self.max_buffer_size {
            drop(buffer); // Release lock before flushing
            self.flush().await?;
        }

        Ok(())
    }

    /// Flush buffer to disk
    pub async fn flush(&self) -> TelemetryResult<()> {
        if !self.enabled {
            return Ok(());
        }

        let mut buffer = self.buffer.write().await;

        if buffer.is_empty() {
            return Ok(());
        }

        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let filename = format!("{}/telemetry-{}.jsonl", self.output_dir, timestamp);

        // Write to JSONL format (one JSON object per line)
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&filename)?;

        let mut writer = std::io::BufWriter::new(file);

        for data in buffer.iter() {
            let json = serde_json::to_string(data)?;
            writeln!(writer, "{}", json)?;
        }

        writer.flush()?;
        let flushed_count = buffer.len();
        buffer.clear();

        tracing::info!("Flushed {} telemetry records to {}", flushed_count, filename);

        Ok(())
    }

    /// Generate ROI report from telemetry data
    pub async fn generate_roi_report(
        &self,
        start_timestamp: i64,
        end_timestamp: i64,
    ) -> TelemetryResult<RoiReport> {
        use std::collections::HashMap;

        // File scanning is blocking I/O — run it on a blocking thread so it
        // doesn't stall the async runtime (this can read many files).
        let output_dir = self.output_dir.clone();
        let all_records: Vec<TelemetryData> = tokio::task::spawn_blocking(move || {
            let mut records = Vec::new();
            let entries = std::fs::read_dir(&output_dir)?;
            for entry in entries {
                let entry = entry?;
                let path = entry.path();

                if path.extension().map_or(false, |ext| ext == "jsonl") {
                    let content = std::fs::read_to_string(&path)?;
                    for line in content.lines() {
                        if let Ok(record) = serde_json::from_str::<TelemetryData>(line) {
                            if record.timestamp >= start_timestamp && record.timestamp <= end_timestamp {
                                records.push(record);
                            }
                        }
                    }
                }
            }
            Ok::<Vec<TelemetryData>, std::io::Error>(records)
        })
        .await
        .map_err(|e| TelemetryError::InvalidPath(format!("blocking task panicked: {e}")))??;

        // Calculate metrics
        let total_requests = all_records.len() as u64;
        let successful_requests = all_records.iter().filter(|r| r.success).count() as u64;
        let failed_requests = total_requests - successful_requests;

        let total_cost_cents: f64 = all_records.iter().map(|r| r.chosen_cost).sum();
        let total_savings_cents: f64 = all_records.iter().map(|r| r.cost_delta).sum();
        let average_latency_ms = if !all_records.is_empty() {
            all_records.iter().map(|r| r.latency_ms).sum::<u64>() / all_records.len() as u64
        } else {
            0
        };

        // Model usage breakdown
        let mut model_usage_map: HashMap<String, (u64, f64, f64)> = HashMap::new();
        for record in &all_records {
            let entry = model_usage_map
                .entry(record.chosen_model.clone())
                .or_insert((0, 0.0, 0.0));
            entry.0 += 1;
            entry.1 += record.chosen_cost;
            entry.2 += record.cost_delta;
        }

        let mut top_models: Vec<ModelUsage> = model_usage_map
            .into_iter()
            .map(|(model, (count, cost, savings))| ModelUsage {
                model_name: model,
                usage_count: count,
                total_cost: cost,
                total_savings: savings,
            })
            .collect();

        top_models.sort_by(|a, b| b.usage_count.cmp(&a.usage_count));
        top_models.truncate(10); // Top 10 models

        // Tier breakdown
        let mut tier_map: HashMap<String, (u64, f64)> = HashMap::new();
        for record in &all_records {
            let entry = tier_map
                .entry(record.user_tier.clone())
                .or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += record.chosen_cost;
        }

        let cost_breakdown_by_tier: Vec<TierBreakdown> = tier_map
            .into_iter()
            .map(|(tier, (count, cost))| TierBreakdown {
                tier,
                request_count: count,
                total_cost: cost,
                average_cost_per_request: cost / count as f64,
            })
            .collect();

        let roi_percentage = if total_cost_cents > 0.0 {
            (total_savings_cents / total_cost_cents) * 100.0
        } else {
            0.0
        };

        Ok(RoiReport {
            period_start: start_timestamp,
            period_end: end_timestamp,
            total_requests,
            successful_requests,
            failed_requests,
            total_cost_cents,
            total_savings_cents,
            average_latency_ms,
            top_models,
            cost_breakdown_by_tier,
            roi_percentage,
        })
    }

    /// Export telemetry as CSV for analysis
    fn csv_escape(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}
    pub async fn export_as_csv(&self, output_path: &str) -> TelemetryResult<()> {
        let buffer = self.buffer.read().await;

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(output_path)?;

        // Write CSV header
        writeln!(
            file,
            "request_id,timestamp,chosen_model,shadow_model,latency_ms,chosen_cost,baseline_cost,cost_delta,priority,user_tier,failover,provider,success,tokens,client_id"
        )?;

        // Write data rows
       for record in buffer.iter() {
            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                Self::csv_escape(&record.request_id),
                record.timestamp,
                Self::csv_escape(&record.chosen_model),
                Self::csv_escape(record.shadow_model.as_deref().unwrap_or("")),
                record.latency_ms,
                record.chosen_cost,
                record.baseline_cost,
                record.cost_delta,
                Self::csv_escape(&record.priority),
                Self::csv_escape(&record.user_tier),
                record.failover_triggered,
                Self::csv_escape(&record.provider),
                record.success,
                record.tokens_used,
                Self::csv_escape(&record.client_id)
            )?;
        }

        file.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_telemetry_recorder_creation() {
        let recorder = TelemetryRecorder::new("/tmp/telemetry-test", 100);
        assert!(recorder.is_ok());
    }

    #[tokio::test]
    async fn test_telemetry_recording() {
        let recorder = TelemetryRecorder::new("/tmp/telemetry-test-2", 100).unwrap();

        let data = TelemetryData::new(
            "test-req-1".to_string(),
            "gpt-4o".to_string(),
            100.0,
            200.0,
            "client-1".to_string(),
        );

        let result = recorder.record(data).await;
        assert!(result.is_ok());
    }
}
