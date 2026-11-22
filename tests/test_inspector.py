"""The TrIVA Inspector's server: reading logs while they grow, listing sessions, finding a session's debug log, and streaming both to the page."""

import asyncio
import json
import os
from datetime import datetime

from aiohttp.test_utils import TestClient, TestServer

from tools import inspector
from tools.inspector import Tail, app, debug_path, llm_record, sessions


def _line(record: dict) -> bytes:
    """Encode a record as one log line."""
    return (json.dumps(record) + "\n").encode()


def test_tail_reads_only_new_complete_lines(tmp_path):
    """A line still being written is held back until its newline arrives, and nothing is read twice."""
    path = tmp_path / "log.jsonl"
    tail = Tail(path)
    assert tail.read() == []

    path.write_bytes(_line({"n": 1}) + b'{"n": ')
    assert tail.read() == [{"n": 1}]
    with path.open("ab") as file:
        file.write(b"2}\n")
    assert tail.read() == [{"n": 2}]
    assert tail.read() == []


def test_tail_skips_lines_that_are_no_json_object(tmp_path):
    """A line cut off by a crash or holding no object is left out, and the lines after it still come through."""
    path = tmp_path / "log.jsonl"
    path.write_bytes(b'{"n": 1\n[1, 2]\n\n' + _line({"n": 3}))
    assert Tail(path).read() == [{"n": 3}]


def test_sessions_lists_newest_first_with_settings(tmp_path):
    """Each study log is listed with its session_start, newest first; a log not opening with one gets no settings."""
    old = tmp_path / "dev-1.jsonl"
    old.write_bytes(_line({"event": "session_start", "session": "dev-1", "llm": "mistral-small"}))
    new = tmp_path / "dev-2.jsonl"
    new.write_bytes(_line({"event": "turn_started"}))
    empty = tmp_path / "dev-3.jsonl"
    empty.write_bytes(b"")
    for age, path in enumerate([empty, new, old]):
        stamp = 1_800_000_000 - age * 60
        os.utime(path, (stamp, stamp))

    listed = sessions(tmp_path)
    assert [entry["session"] for entry in listed] == ["dev-3", "dev-2", "dev-1"]
    assert listed[2]["start"]["llm"] == "mistral-small"
    assert listed[0]["start"] == {} and listed[1]["start"] == {}


def test_debug_path_matches_the_session_exactly(tmp_path):
    """A session's debug log is found by its name plus a start time, never another session's that shares a prefix."""
    (tmp_path / "dev-60_2026-11-20_17-26-46.jsonl").touch()
    (tmp_path / "dev-6_2026-10-30_11-44-52.jsonl").touch()
    (tmp_path / "dev-6_2026-10-30_12-00-00.jsonl").touch()

    assert debug_path("dev-6", tmp_path).name == "dev-6_2026-10-30_12-00-00.jsonl"
    assert debug_path("dev-60", tmp_path).name == "dev-60_2026-11-20_17-26-46.jsonl"
    assert debug_path("dev-7", tmp_path) is None


def test_llm_record_keeps_only_llm_records():
    """An LLM record keeps its content and gains epoch seconds; every other debug record is dropped."""
    record = {"time": "2026-11-20T17:26:46.248", "level": "DEBUG", "source": "x:y:1", "message": "LLM output", "llm": True, "text": "Hi"}
    reduced = llm_record(record)
    assert reduced == {"ts": datetime(2026, 11, 20, 17, 26, 46, 248000).timestamp(), "kind": "LLM output", "text": "Hi"}
    assert llm_record({"time": "2026-11-20T17:26:46.248", "message": "frame", "timeline": True}) is None


async def _next_batch(response) -> dict:
    """Read the next data event of a server-sent event stream."""
    while True:
        event = await asyncio.wait_for(response.content.readuntil(b"\n\n"), timeout=5)
        if event.startswith(b"data: "):
            return json.loads(event[len(b"data: "):])


async def test_events_streams_the_log_and_then_each_new_line(tmp_path, monkeypatch):
    """The stream sends the whole study log and the LLM records first, then new lines as they are appended, including a debug log that appears late."""
    monkeypatch.setattr(inspector, "POLL_S", 0.01)
    (tmp_path / "study").mkdir()
    (tmp_path / "debug").mkdir()
    study = tmp_path / "study" / "dev-1.jsonl"
    study.write_bytes(_line({"event": "session_start", "turn": 0}) + _line({"event": "turn_started", "turn": 1}))

    async with TestClient(TestServer(app(tmp_path))) as client:
        response = await client.get("/events", params={"session": "dev-1"})
        assert response.status == 200
        first = await _next_batch(response)
        assert [record["event"] for record in first["study"]] == ["session_start", "turn_started"]
        assert first["debug"] == []

        with study.open("ab") as file:
            file.write(_line({"event": "user_transcript", "turn": 1, "text": "Hello"}))
        assert (await _next_batch(response))["study"][0]["text"] == "Hello"

        debug = tmp_path / "debug" / "dev-1_2026-11-20_17-26-46.jsonl"
        debug.write_bytes(
            _line({"time": "2026-11-20T17:26:47.000", "message": "frame", "timeline": True})
            + _line({"time": "2026-11-20T17:26:48.000", "message": "LLM output", "llm": True, "text": "Hi"})
        )
        batch = await _next_batch(response)
        assert [record["kind"] for record in batch["debug"]] == ["LLM output"]
        response.close()


async def test_events_refuses_unknown_and_unsafe_sessions(tmp_path):
    """A session without a study log is not found, and a name reaching outside the log folder is refused."""
    (tmp_path / "study").mkdir()
    async with TestClient(TestServer(app(tmp_path))) as client:
        assert (await client.get("/events", params={"session": "nobody"})).status == 404
        assert (await client.get("/events", params={"session": "../secret"})).status == 400
        assert (await client.get("/events")).status == 400


async def test_page_and_session_list_are_served(tmp_path):
    """The page is served at the root and the session list as JSON."""
    (tmp_path / "study").mkdir()
    (tmp_path / "study" / "dev-1.jsonl").write_bytes(_line({"event": "session_start", "session": "dev-1"}))
    async with TestClient(TestServer(app(tmp_path))) as client:
        page = await client.get("/")
        assert page.status == 200 and "<title>TrIVA Inspector</title>" in await page.text()
        listed = await (await client.get("/sessions")).json()
        assert [entry["session"] for entry in listed] == ["dev-1"]


async def test_study_answers_a_whole_log_and_overview_serves_the_page(tmp_path):
    """A session's study log comes back as one list of its records, and the overview is served by the same page."""
    (tmp_path / "study").mkdir()
    (tmp_path / "study" / "dev-1.jsonl").write_bytes(_line({"event": "session_start"}) + _line({"event": "turn_started", "turn": 1}))
    async with TestClient(TestServer(app(tmp_path))) as client:
        records = await (await client.get("/study", params={"session": "dev-1"})).json()
        assert [record["event"] for record in records] == ["session_start", "turn_started"]
        assert (await client.get("/study", params={"session": "nobody"})).status == 404
        assert (await client.get("/study", params={"session": "../x"})).status == 400
        overview = await client.get("/overview")
        assert overview.status == 200 and "<title>TrIVA Inspector</title>" in await overview.text()
