# Dataset Format

CommerceMind expects two logical datasets: products and interactions.

## Raw Products

Raw product files are expected at:

```text
data/raw/<dataset_name>/products.jsonl
```

Each line should be one JSON object. The ingestion layer accepts common product ID fields such as `item_id`, `parent_asin`, or `asin`.

Example:

```json
{"item_id":"sku-running-shoes-001","title":"Running Shoes","category":"Footwear","brand":"StrideLab","description":"Lightweight shoes for road running and daily training.","price":2999.0}
```

After normalization, products become:

```text
item_id | title | category | brand | description | price
```

## Raw Interactions

Raw interaction files are expected at:

```text
data/raw/<dataset_name>/interactions.jsonl
```

Each line should describe one user action on one product.

Example:

```json
{"user_id":"user-runner","item_id":"sku-running-shoes-001","event_type":"purchase","timestamp_ms":1710000000000}
```

After normalization, interactions become:

```text
user_id | item_id | event_type | timestamp_ms
```

Recommended `event_type` values:

- `view`
- `click`
- `add_to_cart`
- `purchase`

## Normalize Files

Run the data pipeline to create processed Parquet files:

```powershell
python -c "from commercemind.data.pipeline import materialize_normalized_dataset; materialize_normalized_dataset('amazon_fashion')"
```

This writes:

```text
data/processed/amazon_fashion/products.parquet
data/processed/amazon_fashion/interactions.parquet
```

## Seed Database

Seed the database from processed Parquet files:

```powershell
python -m commercemind.storage.load_dataset --dataset amazon_fashion --database-url sqlite+pysqlite:///./work/commercemind.db
```

The same command is available after installation:

```powershell
commercemind-seed-db --dataset amazon_fashion --database-url sqlite+pysqlite:///./work/commercemind.db
```

## Synthetic Processed Dataset

For local scale testing without downloading a public dataset, generate a
deterministic processed dataset:

```powershell
python -m commercemind.data.generate_synthetic_dataset --dataset synthetic_large --products 1000 --users 100 --interactions 5000 --seed 42
```

This creates the same normalized product and interaction columns under:

```text
data/processed/synthetic_large/
```

More details are available in [synthetic_dataset.md](synthetic_dataset.md).
