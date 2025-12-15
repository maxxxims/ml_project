#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- SELECT 'CREATE DATABASE $POSTGRES_DB'
    -- WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$POSTGRES_DB')\gexec
    
    -- \c $POSTGRES_DB;
    
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
    

    -- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO $POSTGRES_USER;
    -- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO $POSTGRES_USER;
    
    -- SELECT '✅ Table items created successfully' as message;
EOSQL