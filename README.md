# Rinha Fraude Vetorial

API em Rust para deteccao de fraude por busca vetorial kNN, submetida a Rinha de Backend 2026.

## 1. Contexto do problema

A Rinha de Backend 2026 propoe um servico que recebe transacoes financeiras e devolve um `fraud_score` indicando a probabilidade de fraude, junto com a decisao de aprovacao.

A regra base do desafio e:

- Para cada requisicao em `POST /fraud-score`, o servico deve transformar a transacao em um vetor de features e localizar os 5 vizinhos mais proximos dentro de um conjunto de 3.000.000 transacoes de referencia.
- A partir desses vizinhos, calcula-se um score baseado na proporcao de fraudes vistas e na distancia, e responde-se conforme o contrato oficial.
- O servico precisa rodar dentro de um envelope severo: 1 CPU e 350 MB de memoria total, distribuidos entre proxy e duas APIs.
- A nota considera latencia (p99 e media), taxa de erro e qualidade do score, entao tanto a velocidade do kNN quanto a fidelidade do modelo importam.

O problema combina, entao, tres restricoes simultaneas: dataset grande na memoria, busca em tempo real e orcamento apertado de CPU e RAM.

## 2. Arquitetura

### Topologia de execucao

```text
client -> localhost:9999 -> haproxy -> api1 / api2
                              |          (Unix socket em tmpfs compartilhado)
                              +---- /run/sock/api1.sock
                              +---- /run/sock/api2.sock
```

- `haproxy` faz o balanceamento round-robin entre as duas instancias.
- `api1` e `api2` sao replicas identicas do binario Rust, escutando em Unix Domain Socket dentro de um volume `tmpfs` compartilhado, evitando o overhead de TCP entre containers.
- Cada API roda em modo `read_only` e expoe somente `/tmp` como tmpfs, reduzindo superficie de ataque e uso de disco.

Limites por container (configuracao oficial da Rinha):

```text
haproxy: 0.05 CPU /  20 MB
api1:    0.475 CPU / 165 MB
api2:    0.475 CPU / 165 MB
total:   1.0  CPU / 350 MB
```

### Fluxo da requisicao

1. O `main` carrega `FraudEngine` no startup, fazendo `Box::leak` para entregar uma referencia `'static` ao roteador `axum`.
2. `GET /ready` responde `204 No Content` apenas depois que a carga do dataset terminou.
3. `POST /fraud-score` desserializa o JSON, delega a pontuacao a um `spawn_blocking` (a busca e CPU-bound) e responde com o JSON do contrato oficial.

### Decisoes do motor de fraude

- **Armazenamento**: vetores de referencia ficam em `Vec<i16>` linear com `N * 14` posicoes e labels em `Vec<u8>`. A quantizacao usa escala 10000 e preserva o sentinela `-1`.
- **Busca**: brute force exata sobre o bloco linear, calculando distancia euclidiana ao quadrado e mantendo o top 5 em arrays fixos sem alocacao por requisicao.
- **Memoria**: 3.000.000 vetores ocupam ~84 MB de vetores quantizados e ~3 MB de labels por instancia, deixando margem para runtime e parser dentro de 165 MB por API.
- **Pre-processamento**: o build do Docker gera `references.bin` a partir de `references.json.gz`, eliminando o parse de JSON no startup. Em execucao local, a API usa `references.bin` se existir; caso contrario, faz fallback para o `.json.gz`.
- **Riscos conhecidos**: o p99 e sensivel a brute force em 3M vetores por request; o startup com `references.json.gz` e caro, por isso o `references.bin` e a forma recomendada.

### Layout do codigo

```text
src/
  main.rs               # Bootstrap axum, suporte a TCP e Unix socket
  lib.rs                # FraudEngine: carga, vetorizacao, kNN e score
  bin/preprocess.rs     # Converte references.json.gz em references.bin
data/
  normalization.json    # Parametros de normalizacao das features
  mcc_risk.json         # Score de risco por MCC
  references.json.gz    # Dataset bruto (entrada do preprocess)
  references.bin        # Dataset binario pronto para mmap-like load
```

## 3. Tecnologias utilizadas

- **Rust 2024 edition** com perfil release agressivo (`codegen-units = 1`, `lto = "thin"`, `strip = true`, `panic = "abort"`) para produzir um binario pequeno e rapido, sem GC no caminho critico.
- **Tokio** como runtime assincrono multi-thread para servir requisicoes.
- **axum 0.7** como framework HTTP, com features minimas (`http1`, `json`, `tokio`).
- **hyper / hyper-util** para servir conexoes em Unix Domain Socket atraves de `auto::Builder`.
- **tower** para compor servicos com o `MakeService` do axum.
- **serde / serde_json** para serializacao do contrato JSON.
- **chrono** para parse das datas em UTC e calculo de hora e dia da semana.
- **flate2** (backend Rust puro) para descompactar `references.json.gz` quando o binario pre-processado nao esta disponivel.
- **HAProxy 2.9** como load balancer Unix-socket-aware.
- **Docker / Docker Compose** para entregar a topologia exigida pela Rinha, com build multi-stage baseado em `rust:1-bookworm` e runtime em `debian:bookworm-slim`.
- **Compilacao com `RUSTFLAGS="-C target-cpu=x86-64-v3"`** para habilitar instrucoes vetoriais modernas no host alvo da Rinha.

## 4. Como executar localmente

### Pre-requisitos

- Rust stable (edition 2024 disponivel) e `cargo`.
- Docker e Docker Compose, opcionais para reproduzir a topologia da Rinha.
- Os arquivos do dataset disponiveis em `data/`:

```text
data/
  normalization.json
  mcc_risk.json
  references.json.gz
```

### Pre-processamento opcional do dataset

Para evitar o custo de descompactar e parsear o JSON a cada startup, gere o binario compacto:

```powershell
cargo run --release --bin preprocess -- --input data/references.json.gz --output data/references.bin
```

Quando `data/references.bin` existir, a API usa esse arquivo; caso contrario, ela carrega o `.json.gz`.

### Subir a API direto com cargo

```powershell
$env:DATA_DIR="data"
$env:PORT="9999"
cargo run --release
```

Validar readiness:

```powershell
curl http://localhost:9999/ready
```

Solicitar um score:

```powershell
curl -X POST http://localhost:9999/fraud-score `
  -H "Content-Type: application/json" `
  -d '{"id":"tx-3576980410","transaction":{"amount":384.88,"installments":3,"requested_at":"2026-03-11T20:23:35Z"},"customer":{"avg_amount":769.76,"tx_count_24h":3,"known_merchants":["MERC-009","MERC-001","MERC-001"]},"merchant":{"id":"MERC-001","mcc":"5912","avg_amount":298.95},"terminal":{"is_online":false,"card_present":true,"km_from_home":13.7090520965},"last_transaction":{"timestamp":"2026-03-11T14:58:35Z","km_from_current":18.8626479774}}'
```

### Subir a topologia completa via Docker Compose

```powershell
docker compose up --build
```

A API publica passa a estar disponivel em `http://localhost:9999`.

### Rodar os testes

```powershell
cargo test
```

A suite cobre clamp, hora UTC, dia da semana, `last_transaction = null`, comerciante desconhecido, MCC desconhecido, vetorizacao completa, score, regra de aprovacao e top 5.

### Benchmark rapido com `hey`

```powershell
hey -n 1000 -c 20 -m POST `
  -H "Content-Type: application/json" `
  -d '{"id":"tx-3576980410","transaction":{"amount":384.88,"installments":3,"requested_at":"2026-03-11T20:23:35Z"},"customer":{"avg_amount":769.76,"tx_count_24h":3,"known_merchants":["MERC-009","MERC-001","MERC-001"]},"merchant":{"id":"MERC-001","mcc":"5912","avg_amount":298.95},"terminal":{"is_online":false,"card_present":true,"km_from_home":13.7090520965},"last_transaction":{"timestamp":"2026-03-11T14:58:35Z","km_from_current":18.8626479774}}' `
  http://localhost:9999/fraud-score
```
