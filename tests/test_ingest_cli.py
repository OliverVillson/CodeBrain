import json
from pathlib import Path

import psycopg
import pytest

from codebrain.ingest.__main__ import main


def test_cli_ingests_into_postgres(
    database_url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "util.py").write_text("def slugify(s):\n    return s.lower()\n")
    try:
        assert main([str(tmp_path), "--repo", "cli-demo", "--database-url", database_url]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["units_written"] == 2 and report["errors"] == []
        with psycopg.connect(database_url) as conn:
            row = conn.execute(
                "SELECT count(*) FROM unit_vectors v JOIN units u ON u.id = v.unit_id "
                "WHERE u.repo = 'cli-demo'"
            ).fetchone()
            assert row is not None and row[0] == 4
    finally:
        with psycopg.connect(database_url) as conn:
            conn.execute("DELETE FROM units WHERE repo = 'cli-demo'")
