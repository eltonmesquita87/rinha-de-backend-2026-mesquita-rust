# Rinha Fraude Vetorial

API em Rust para deteccao de fraude por busca vetorial kNN.

## Decisoes tecnicas

1. Linguagem: Rust, por binario pequeno, baixo overhead e ausencia de GC no caminho critico.
2. Armazenamento: vetores em `Vec<i16>` linear com `N * 14` posicoes e labels em `Vec<u8>`. A quantizacao usa escala 10000 e preserva o sentinela `-1`.
3. Busca: brute force exata sobre o bloco linear, calculando distancia euclidiana ao quadrado e mantendo apenas o top 5 em arrays fixos.
4. Memoria: 3.000.000 vetores usam cerca de 84 MB para vetores quantizados e 3 MB para labels por instancia, deixando margem para runtime e parser dentro de 165 MB por API.
5. Riscos: p99 pode sofrer com brute force em 3M vetores por request; o startup com `references.json.gz` tambem e caro. Para producao, prefira gerar `references.bin` antes de subir o compose.
6. Plano: carregar recursos no startup, validar readiness somente apos carga, vetorizar cada request, buscar top 5, calcular `fraud_score` e responder pelo contrato oficial.

## Arquivos esperados

Crie um diretorio `data` na raiz com:

```text
data/
  normalization.json
  mcc_risk.json
  references.json.gz
```

Opcionalmente, gere o binario compacto:

```powershell
cargo run --release --bin preprocess -- --input data/references.json.gz --output data/references.bin
```

Quando `data/references.bin` existir, a API usa esse arquivo. Caso contrario, ela carrega `references.json.gz`.

## Executar localmente

```powershell
$env:DATA_DIR="data"
$env:PORT="9999"
cargo run --release
```

Readiness:

```powershell
curl http://localhost:9999/ready
```

Score:

```powershell
curl -X POST http://localhost:9999/fraud-score `
  -H "Content-Type: application/json" `
  -d '{"id":"tx-3576980410","transaction":{"amount":384.88,"installments":3,"requested_at":"2026-03-11T20:23:35Z"},"customer":{"avg_amount":769.76,"tx_count_24h":3,"known_merchants":["MERC-009","MERC-001","MERC-001"]},"merchant":{"id":"MERC-001","mcc":"5912","avg_amount":298.95},"terminal":{"is_online":false,"card_present":true,"km_from_home":13.7090520965},"last_transaction":{"timestamp":"2026-03-11T14:58:35Z","km_from_current":18.8626479774}}'
```

## Docker Compose

```powershell
docker compose up --build
```

Topologia:

```text
client -> localhost:9999 -> haproxy -> api1/api2:8080
```

Limites configurados:

```text
haproxy: 0.05 CPU / 20 MB
api1:    0.475 CPU / 165 MB
api2:    0.475 CPU / 165 MB
total:   1 CPU / 350 MB
```

## Testes

```powershell
cargo test
```

Os testes cobrem clamp, hora UTC, dia da semana, `last_transaction = null`, comerciante desconhecido, MCC desconhecido, vetorizacao completa, score, regra de aprovacao e top 5.

## Benchmark simples

Com `hey`:

```powershell
hey -n 1000 -c 20 -m POST `
  -H "Content-Type: application/json" `
  -d '{"id":"tx-3576980410","transaction":{"amount":384.88,"installments":3,"requested_at":"2026-03-11T20:23:35Z"},"customer":{"avg_amount":769.76,"tx_count_24h":3,"known_merchants":["MERC-009","MERC-001","MERC-001"]},"merchant":{"id":"MERC-001","mcc":"5912","avg_amount":298.95},"terminal":{"is_online":false,"card_present":true,"km_from_home":13.7090520965},"last_transaction":{"timestamp":"2026-03-11T14:58:35Z","km_from_current":18.8626479774}}' `
  http://localhost:9999/fraud-score
```
