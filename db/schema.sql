-- Main transactions table. Every transaction (scored or not) gets written
-- here, both from the CSV-replay stream and from live agent processing.
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id       TEXT PRIMARY KEY,
    sender_account        TEXT NOT NULL,
    receiver_account      TEXT NOT NULL,
    amount                 REAL NOT NULL,
    timestamp               TEXT NOT NULL,     -- ISO 8601 string
    merchant_category       TEXT,
    transaction_type         TEXT,
    spending_deviation_score REAL,
    velocity_score            REAL,
    geo_anomaly_score          REAL,
    ip_address                  TEXT,
    location                     TEXT,
    payment_channel                TEXT,
    device_hash                     TEXT,

    -- populated after scoring
    fraud_probability                 REAL,
    is_flagged                          INTEGER DEFAULT 0,   -- 0/1

    -- populated after human review (Task 6/9)
    investigation_status                  TEXT DEFAULT 'none'  -- none | pending | cleared | confirmed_fraud
);

-- Sender/receiver both need fast lookups by account + recency, since
-- get_account_history() runs this exact filter on every flagged
-- transaction and every agent tool call.
CREATE INDEX IF NOT EXISTS idx_sender_ts   ON transactions(sender_account, timestamp);
CREATE INDEX IF NOT EXISTS idx_receiver_ts ON transactions(receiver_account, timestamp);

-- Long-term memory table (Task 9/10) — human decisions persist here so
-- future investigations of the same account aren't starting blind.
CREATE TABLE IF NOT EXISTS investigator_feedback (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id    TEXT NOT NULL,
    account_id          TEXT NOT NULL,
    decision              TEXT NOT NULL,   -- approve | reject | need_more_info
    notes                   TEXT,
    decided_at                TEXT NOT NULL,
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
);

CREATE INDEX IF NOT EXISTS idx_feedback_account ON investigator_feedback(account_id);