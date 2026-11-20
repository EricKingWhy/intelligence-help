ALTER TABLE review_queue ADD COLUMN publication_chunk_id BIGINT NULL;
ALTER TABLE qa_extraction_staging ADD COLUMN publication_chunk_id BIGINT NULL;
ALTER TABLE tickets ADD COLUMN submission_key VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NULL,
    ADD CONSTRAINT uq_ticket_submission UNIQUE (conversation_id, submission_key);
