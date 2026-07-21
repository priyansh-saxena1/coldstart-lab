from coldstart.runner import _last_json_line


def test_picks_json_out_of_library_noise():
    # this is the real hazard: transformers/torch dump progress bars, load reports
    # and warnings to stdout too sometimes. the worker's json is the last {..} line.
    noisy = (
        "Fetching 8 files: 100%|####| 8/8\n"
        "GPT2 LOAD REPORT\n"
        "some key | UNEXPECTED\n"
        '{"phases": {"load": 1.23}, "meta": {"model_id": "x"}}\n'
    )
    got = _last_json_line(noisy)
    assert got is not None
    assert got["phases"]["load"] == 1.23


def test_returns_none_when_no_json():
    assert _last_json_line("just warnings\nno json here\n") is None


def test_ignores_non_json_braced_lines():
    # a stray "{something}" that isn't valid json shouldn't be mistaken for output
    text = '{not valid json}\n{"ok": true}\n'
    assert _last_json_line(text) == {"ok": True}
