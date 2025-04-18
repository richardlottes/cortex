#SQLite DB schemas
chunks_schema = """
CREATE TABLE IF NOT EXISTS chunks (
    uuid TEXT PRIMARY KEY,
    doc_uuid TEXT,
    vector_id INTEGER UNIQUE,
    text TEXT,
    embedding BLOB
)"""
docs_schema = """
CREATE TABLE IF NOT EXISTS documents (
    uuid TEXT PRIMARY KEY,
    name TEXT,
    url TEXT,
    text TEXT,
    type TEXT
)"""
schemas = [chunks_schema, docs_schema]