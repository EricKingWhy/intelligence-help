-- Manual tickets and refunds are keyed by the exact conversation owner and action-prefixed key.
-- Agent confirmations and handoffs keep their conversation-bound submission keys.
ALTER TABLE tickets
    ADD COLUMN submission_owner VARBINARY(256) NULL,
    ADD CONSTRAINT uq_ticket_account_submission UNIQUE (submission_owner, submission_key);

UPDATE tickets t JOIN conversations c ON c.id = t.conversation_id
SET t.submission_owner = CAST(c.user_id AS BINARY)
WHERE t.submission_owner IS NULL
  AND (LEFT(t.submission_key, 7) IN ('ticket:', 'refund:'));
