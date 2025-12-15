#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE TABLE IF NOT EXISTS items (
        id BIGSERIAL PRIMARY KEY,
        title TEXT,
        description TEXT,
        microcat_id INTEGER,
        item_price FLOAT DEFAULT 0,
        real_weight FLOAT,
        real_height FLOAT,
        real_length FLOAT,
        real_width FLOAT,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
EOSQL