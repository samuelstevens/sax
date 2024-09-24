CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    -- JSON
    config_json TEXT NOT NULL,
    data_json TEXT NOT NULL,

    -- For reproducibility
    -- JSON
    argv_json TEXT NOT NULL,
    git_hash TEXT NOT NULL,
    -- 0 indicate no git changes, anything else indicates changes (dirty)
    git_dirty INTEGER NOT NULL,
    posix_time INTEGER NOT NULL,
    hostname TEXT NOT NULL
) STRICT; -- https://www.sqlite.org/stricttables.html
