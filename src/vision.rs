// =============================================================================
// src/vision.rs  — RouterFuel v0.6
//
// Vision / Multimodal Support
//
// Extends the ChatMessage schema to carry:
//   - text content (existing)
//   - image_url    (URL pointing to an image)
//   - base64       (inline image data)
//
// Vision capability is no longer a separate hardcoded list here — it's the
// `supports_vision` field on each ModelConfig in route_engine.rs, so adding
// or updating a model in one place keeps routing and vision-filtering in
// sync automatically.
// =============================================================================

use serde::{Deserialize, Serialize};

// =============================================================================
// Extended message content types
// =============================================================================

/// A chat message that can carry either text or image content.
/// Replaces the plain `ChatMessage { role, content: String }` for multimodal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalMessage {
    pub role:    String,
    pub content: MessageContent,
}

/// Content can be a plain string (text-only) or a list of content parts
/// (text + one or more images).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text-only message — same as before
    Text(String),
    /// Multimodal: list of text and/or image parts
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Extract all text segments as a single string (for token counting / caching)
    pub fn as_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match &p.kind {
                    PartKind::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
        }
    }

    /// Returns true if this message contains at least one image
    pub fn has_image(&self) -> bool {
        match self {
            MessageContent::Text(_) => false,
            MessageContent::Parts(parts) => parts.iter().any(|p| !matches!(p.kind, PartKind::Text { .. })),
        }
    }
}

/// A single content part in a multimodal message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub part_type: String,
    #[serde(flatten)]
    pub kind: PartKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PartKind {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: ImageUrl,
    },
    ImageBase64 {
        image_data: ImageBase64,
    },
}

/// Reference to an image by URL (must be publicly accessible or data URI)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url:    String,
    /// "low" | "high" | "auto" — controls detail level and cost
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Inline base64-encoded image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageBase64 {
    /// MIME type: "image/jpeg" | "image/png" | "image/gif" | "image/webp"
    pub media_type: String,
    /// Raw base64 string (no data URI prefix)
    pub data:       String,
}

// =============================================================================
// Format conversion: OpenAI-compatible ↔ Anthropic ↔ Gemini
// =============================================================================

/// Convert a multimodal message into the Anthropic wire format.
/// Anthropic uses: { role, content: [ { type, text|source } ] }
pub fn to_anthropic_content(msg: &MultimodalMessage) -> serde_json::Value {
    match &msg.content {
        MessageContent::Text(t) => serde_json::json!({
            "role": msg.role,
            "content": t
        }),
        MessageContent::Parts(parts) => {
            let content: Vec<serde_json::Value> = parts.iter().map(|p| {
                match &p.kind {
                    PartKind::Text { text } => serde_json::json!({
                        "type": "text",
                        "text": text
                    }),
                    PartKind::ImageUrl { image_url } => serde_json::json!({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image_url.url
                        }
                    }),
                    PartKind::ImageBase64 { image_data } => serde_json::json!({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_data.media_type,
                            "data": image_data.data
                        }
                    }),
                }
            }).collect();

            serde_json::json!({ "role": msg.role, "content": content })
        }
    }
}

/// Convert a multimodal message into the Gemini wire format.
/// Gemini uses: { role, parts: [ { text } | { inlineData } ] }
pub fn to_gemini_content(msg: &MultimodalMessage) -> serde_json::Value {
    let role = if msg.role == "assistant" { "model" } else { "user" };

    match &msg.content {
        MessageContent::Text(t) => serde_json::json!({
            "role": role,
            "parts": [{ "text": t }]
        }),
        MessageContent::Parts(parts) => {
            let gemini_parts: Vec<serde_json::Value> = parts.iter().map(|p| {
                match &p.kind {
                    PartKind::Text { text } => serde_json::json!({ "text": text }),
                    PartKind::ImageUrl { image_url } => serde_json::json!({
                        // Gemini accepts file URIs or GCS URLs
                        "fileData": {
                            "mimeType": "image/jpeg",
                            "fileUri": image_url.url
                        }
                    }),
                    PartKind::ImageBase64 { image_data } => serde_json::json!({
                        "inlineData": {
                            "mimeType": image_data.media_type,
                            "data":     image_data.data
                        }
                    }),
                }
            }).collect();

            serde_json::json!({ "role": role, "parts": gemini_parts })
        }
    }
}

/// Convert a multimodal message into the OpenAI-compatible wire format used
/// by OpenAI, xAI/Grok, Qwen, Kimi/Moonshot, Meta Llama, Mistral, and
/// OpenRouter — all of which accept the same
/// `content: [{type: "text"|"image_url", ...}]` array shape.
pub fn to_openai_compatible_content(msg: &MultimodalMessage) -> serde_json::Value {
    match &msg.content {
        MessageContent::Text(t) => serde_json::json!({
            "role": msg.role,
            "content": t
        }),
        MessageContent::Parts(parts) => {
            let content: Vec<serde_json::Value> = parts.iter().map(|p| {
                match &p.kind {
                    PartKind::Text { text } => serde_json::json!({
                        "type": "text",
                        "text": text
                    }),
                    PartKind::ImageUrl { image_url } => serde_json::json!({
                        "type": "image_url",
                        "image_url": {
                            "url": image_url.url,
                            "detail": image_url.detail.clone().unwrap_or_else(|| "auto".into())
                        }
                    }),
                    PartKind::ImageBase64 { image_data } => serde_json::json!({
                        "type": "image_url",
                        "image_url": {
                            "url": format!("data:{};base64,{}", image_data.media_type, image_data.data)
                        }
                    }),
                }
            }).collect();

            serde_json::json!({ "role": msg.role, "content": content })
        }
    }
}

// =============================================================================
// Vision-aware routing helper
// =============================================================================

use crate::route_engine::{RouteEngine, RoutingPriority};

/// Select the best vision-capable model.
/// Only considers models that support image input (per the registry's
/// `supports_vision` flag — see route_engine.rs).
/// Falls back to quality priority if no preference given.
pub fn select_vision_model(
    engine:       &RouteEngine,
    input_tokens: u32,
    priority:     RoutingPriority,
) -> anyhow::Result<crate::route_engine::RoutingDecision> {
    let decision = engine.select(input_tokens, 1024, priority)?;

    if decision.model.supports_vision {
        return Ok(decision);
    }

    // Fallback: try quality priority (flagship models are usually vision-capable)
    let fallback = engine.select(input_tokens, 1024, RoutingPriority::Quality)?;
    if fallback.model.supports_vision {
        return Ok(fallback);
    }

    // Last resort: pick the best-scoring vision-capable model directly,
    // ignoring anything without the flag.
    let vision_models = engine.list_vision_capable();
    vision_models
        .into_iter()
        .filter(|m| input_tokens < m.context_window)
        .max_by(|a, b| a.quality_score.partial_cmp(&b.quality_score).unwrap())
        .map(|model| crate::route_engine::RoutingDecision {
            reason: format!("{} chosen as best available vision-capable model", model.display_name),
            score: model.quality_score as f64,
            model,
        })
        .ok_or_else(|| anyhow::anyhow!("No vision-capable model available in registry"))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_message_has_no_image() {
        let msg = MultimodalMessage {
            role:    "user".into(),
            content: MessageContent::Text("hello".into()),
        };
        assert!(!msg.content.has_image());
    }

    #[test]
    fn parts_message_detects_image() {
        let msg = MultimodalMessage {
            role: "user".into(),
            content: MessageContent::Parts(vec![
                ContentPart {
                    part_type: "text".into(),
                    kind: PartKind::Text { text: "What is in this image?".into() },
                },
                ContentPart {
                    part_type: "image_url".into(),
                    kind: PartKind::ImageUrl {
                        image_url: ImageUrl {
                            url:    "https://example.com/food.jpg".into(),
                            detail: Some("high".into()),
                        },
                    },
                },
            ]),
        };
        assert!(msg.content.has_image());
        assert_eq!(msg.content.as_text(), "What is in this image?");
    }

    #[test]
    fn registry_flags_flagship_models_as_vision_capable() {
        let engine = RouteEngine::new();
        assert!(engine.is_vision_capable("claude-opus-4-8"));
        assert!(engine.is_vision_capable("gpt-5.6-sol"));
        assert!(engine.is_vision_capable("gemini-3.1-pro"));
        assert!(!engine.is_vision_capable("deepseek-v4-flash"));
    }

    #[test]
    fn anthropic_conversion_base64() {
        let msg = MultimodalMessage {
            role: "user".into(),
            content: MessageContent::Parts(vec![
                ContentPart {
                    part_type: "image".into(),
                    kind: PartKind::ImageBase64 {
                        image_data: ImageBase64 {
                            media_type: "image/png".into(),
                            data:       "abc123".into(),
                        },
                    },
                },
            ]),
        };
        let v = to_anthropic_content(&msg);
        let source = &v["content"][0]["source"];
        assert_eq!(source["type"], "base64");
        assert_eq!(source["media_type"], "image/png");
    }

    #[test]
    fn openai_compatible_conversion_base64() {
        let msg = MultimodalMessage {
            role: "user".into(),
            content: MessageContent::Parts(vec![
                ContentPart {
                    part_type: "image_url".into(),
                    kind: PartKind::ImageBase64 {
                        image_data: ImageBase64 {
                            media_type: "image/jpeg".into(),
                            data:       "xyz789".into(),
                        },
                    },
                },
            ]),
        };
        let v = to_openai_compatible_content(&msg);
        let url = v["content"][0]["image_url"]["url"].as_str().unwrap();
        assert!(url.starts_with("data:image/jpeg;base64,"));
    }
}
