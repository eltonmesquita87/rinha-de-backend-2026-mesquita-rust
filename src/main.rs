use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use rinha_fraude_vetorial::{FraudEngine, FraudRequest, ResourcePaths};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let paths = ResourcePaths::from_env();
    let engine: &'static FraudEngine = Box::leak(Box::new(FraudEngine::load(&paths)?));

    eprintln!(
        "loaded {} references using {} bytes",
        engine.reference_count(),
        engine.reference_memory_bytes()
    );

    let app = Router::new()
        .route("/ready", get(ready))
        .route("/fraud-score", post(fraud_score))
        .with_state(engine);

    let port = std::env::var("PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(8080);
    let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), port);
    let listener = tokio::net::TcpListener::bind(address).await?;

    axum::serve(listener, app).await?;
    Ok(())
}

async fn ready() -> StatusCode {
    StatusCode::NO_CONTENT
}

async fn fraud_score(
    State(engine): State<&'static FraudEngine>,
    Json(request): Json<FraudRequest>,
) -> impl IntoResponse {
    match tokio::task::spawn_blocking(move || engine.score(&request)).await {
        Ok(Ok(response)) => (StatusCode::OK, Json(response)).into_response(),
        Ok(Err(_)) => StatusCode::BAD_REQUEST.into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}
