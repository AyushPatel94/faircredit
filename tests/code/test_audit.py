from modelgate import audit


def test_audit_append_and_read(tmp_path):
    path = tmp_path / "decisions.jsonl"
    audit.append({"action": "promote", "week": 1, "version": 2}, path=path)
    audit.append({"action": "skip", "week": 2, "version": 3, "reasons": ["x"]}, path=path)
    entries = audit.read_all(path)
    assert len(entries) == 2
    assert entries[0]["action"] == "promote"
    assert entries[1]["action"] == "skip"
    assert "recorded_at" in entries[0]


def test_audit_latest_returns_tail(tmp_path):
    path = tmp_path / "decisions.jsonl"
    for i in range(7):
        audit.append({"action": "skip", "week": i}, path=path)
    tail = audit.latest(3, path=path)
    assert len(tail) == 3
    assert tail[-1]["week"] == 6
