use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use rinha_fraude_vetorial::{FraudEngine, FraudRequest, ResourcePaths};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[tokio::main]
async fn main() -> Result<(), BoxError> {
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

    if let Some(socket_path) = std::env::var_os("SOCKET_PATH") {
        serve_unix(app, std::path::PathBuf::from(socket_path)).await
    } else {
        serve_tcp(app).await
    }
}

async fn serve_tcp(app: Router) -> Result<(), BoxError> {
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    let port = std::env::var("PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(8080);
    let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), port);
    let listener = tokio::net::TcpListener::bind(address).await?;
    eprintln!("listening on tcp {}", address);
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(unix)]
async fn serve_unix(app: Router, path: std::path::PathBuf) -> Result<(), BoxError> {
    use hyper::body::Incoming;
    use hyper::service::service_fn;
    use hyper_util::rt::{TokioExecutor, TokioIo};
    use hyper_util::server::conn::auto;
    use std::convert::Infallible;
    use std::os::unix::fs::PermissionsExt;
    use tokio::net::UnixListener;
    use tower::Service;

    // Limpa socket residual de execucoes anteriores no tmpfs compartilhado.
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path)?;

    // HAProxy roda em outro UID; abre permissoes para ele se conectar.
    let mut perms = std::fs::metadata(&path)?.permissions();
    perms.set_mode(0o666);
    std::fs::set_permissions(&path, perms)?;

    eprintln!("listening on unix socket {}", path.display());

    let mut make_service = app.into_make_service();

    loop {
        let (socket, _addr) = listener.accept().await?;

        // IntoMakeService is always ready; descarta o Infallible da resposta.
        let tower_service = match make_service.call(()).await {
            Ok(svc) => svc,
            Err(unreachable) => match unreachable {
                _ => unreachable!(),
            },
        };

        tokio::spawn(async move {
            let io = TokioIo::new(socket);
            let svc = service_fn(move |request: hyper::Request<Incoming>| {
                let mut tower_service = tower_service.clone();
                async move {
                    let result: Result<_, Infallible> = tower_service.call(request).await;
                    result
                }
            });

            if let Err(err) = auto::Builder::new(TokioExecutor::new())
                .serve_connection(io, svc)
                .await
            {
                eprintln!("connection error: {err:#}");
            }
        });
    }
}

#[cfg(not(unix))]
async fn serve_unix(_: Router, _: std::path::PathBuf) -> Result<(), BoxError> {
    Err("SOCKET_PATH only supported on Unix targets".into())
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
