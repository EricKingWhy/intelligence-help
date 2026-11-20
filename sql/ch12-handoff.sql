ALTER TABLE messages ADD COLUMN handoff_context JSON NULL;

ALTER TABLE tickets ADD COLUMN handoff_context JSON NULL,
    ADD COLUMN assigned_to VARCHAR(64) NULL,
    ADD COLUMN resolution TEXT NULL;
