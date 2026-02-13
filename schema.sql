-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    display_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    verified INTEGER DEFAULT 1,
    verification_code TEXT,
    verification_expires_at TIMESTAMP
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    sources TEXT,  -- JSON string
    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast history retrieval
CREATE INDEX IF NOT EXISTS idx_messages_user_session ON messages(user_id, session_id);

-- Feedback table
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    for_id TEXT NOT NULL,
    thread_id TEXT,
    user_id INTEGER,
    value INTEGER NOT NULL,
    comment TEXT,
    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
