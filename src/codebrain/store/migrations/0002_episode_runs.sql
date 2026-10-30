-- 0002: an episode unit can hold the evidence of several runs (sketch §8).
-- A near-duplicate episode is merged into the existing unit instead of inserted, and
-- its evidence is kept as another row here rather than overwriting the first run.

ALTER TABLE episodes DROP CONSTRAINT episodes_pkey;
ALTER TABLE episodes ADD COLUMN id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY;
CREATE INDEX episodes_unit_idx ON episodes (unit_id, ended_at);
