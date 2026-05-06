# syntax=docker/dockerfile:1

FROM rust:1-bookworm AS builder
WORKDIR /build

ENV RUSTFLAGS="-C target-cpu=x86-64-v3"
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked

# Pre-processa o dataset para references.bin durante o build, evitando
# parse de JSON no startup (mais rapido e com pico de RAM controlado).
COPY data ./data
RUN /build/target/release/preprocess \
        --input /build/data/references.json.gz \
        --output /build/data/references.bin \
        --index-output /build/data/references.index.bin \
        --max-per-class 0

FROM debian:bookworm-slim
RUN useradd --system --uid 10001 --create-home appuser
WORKDIR /app

COPY --from=builder /build/target/release/rinha-fraude-vetorial /usr/local/bin/rinha-fraude-vetorial
COPY --from=builder /build/data/references.bin /app/data/references.bin
COPY --from=builder /build/data/references.index.bin /app/data/references.index.bin
COPY data/mcc_risk.json /app/data/mcc_risk.json
COPY data/normalization.json /app/data/normalization.json

ENV DATA_DIR=/app/data
ENV PORT=8080
EXPOSE 8080

USER appuser
ENTRYPOINT ["rinha-fraude-vetorial"]
