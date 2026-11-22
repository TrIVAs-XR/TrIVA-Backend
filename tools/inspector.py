"""TrIVA Inspector: a local web page that follows one session's study log and the LLM records of its debug log as they are written, for looking at an agent turn by turn, and compares all sessions in an overview, while developing and testing it.

It only reads the logs, so it runs beside an agent or on its own, and shows past sessions as well as the one running.
"""

import argparse
import asyncio
import json
import re
import webbrowser
from datetime import datetime
from pathlib import Path

from aiohttp import web
from loguru import logger

PORT = 8081
PAGE = Path(__file__).with_suffix(".html")
LOGS = Path(__file__).resolve().parent.parent / "logs"

# Where the app finds the log folder it reads.
_LOGS_KEY = web.AppKey("logs", Path)

# How often an open stream looks for new lines.
POLL_S = 0.25
# How often an idle stream sends a comment, so a closed page is noticed.
KEEPALIVE_S = 15

# A debug log is named after its session plus the start time, as debug_log.start names it.
_DEBUG_NAME = r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.jsonl"

# Fields of a debug record that say where it was logged rather than what the LLM did.
_DEBUG_ENVELOPE = {"time", "level", "source", "message", "llm"}


class Tail:
    """Reads the JSON lines appended to a file since the last read, holding back a line still being written."""

    def __init__(self, path: Path, marker: bytes = b""):
        """Start at the beginning of the file, which need not exist yet; with a marker, only lines containing it are parsed."""
        self.path = path
        self._marker = marker
        self._offset = 0
        self._partial = b""

    def read(self) -> list[dict]:
        """Return the records of every complete line added since the last call; a line that is not a JSON object is skipped."""
        try:
            with self.path.open("rb") as file:
                file.seek(self._offset)
                chunk = file.read()
        except FileNotFoundError:
            return []
        self._offset += len(chunk)
        *lines, self._partial = (self._partial + chunk).split(b"\n")
        records = []
        for line in lines:
            if not line.strip() or self._marker not in line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipped a line of {self.path.name} that is not JSON")
                continue
            if isinstance(record, dict):
                records.append(record)
        return records


def sessions(study: Path) -> list[dict]:
    """List the sessions that have a study log, newest first, each with its file's modification time and its session_start settings."""
    listed = []
    for path in study.glob("*.jsonl"):
        try:
            with path.open(encoding="utf-8") as file:
                first = file.readline()
            start = json.loads(first) if first.strip() else {}
            modified = path.stat().st_mtime
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(start, dict) or start.get("event") != "session_start":
            start = {}
        listed.append({"session": path.stem, "modified": modified, "start": start})
    return sorted(listed, key=lambda entry: entry["modified"], reverse=True)


def debug_path(session: str, debug: Path) -> Path | None:
    """Return the debug log of a session, the newest if a name was ever reused, or None while it has none."""
    name = re.compile(re.escape(session) + _DEBUG_NAME)
    matches = [path for path in debug.glob("*.jsonl") if name.fullmatch(path.name)]
    return max(matches, key=lambda path: path.name, default=None)


def llm_record(record: dict) -> dict | None:
    """Reduce a debug record to what an LLM call took or gave, stamped with epoch seconds like the study log; None for any other record."""
    if not record.get("llm") or "time" not in record:
        return None
    try:
        ts = datetime.fromisoformat(record["time"]).timestamp()
    except (TypeError, ValueError):
        return None
    fields = {key: value for key, value in record.items() if key not in _DEBUG_ENVELOPE}
    return {"ts": ts, "kind": record.get("message"), **fields}


def _session_name(request: web.Request) -> str:
    """Return the session a request names, refusing one that could leave the log folder."""
    session = request.query.get("session", "")
    if not session or "\x00" in session or Path(session).name != session:
        raise web.HTTPBadRequest(text="Name a session.")
    return session


async def _page(request: web.Request) -> web.FileResponse:
    """Serve the Inspector page, which shows one session at the root and all sessions at /overview."""
    return web.FileResponse(PAGE, headers={"Cache-Control": "no-cache"})


async def _sessions(request: web.Request) -> web.Response:
    """Answer with the session list, for the page's picker and for following the newest session."""
    return web.json_response(sessions(request.app[_LOGS_KEY] / "study"))


async def _study(request: web.Request) -> web.Response:
    """Answer with a session's study log as it stands, for the overview's statistics across sessions."""
    path = request.app[_LOGS_KEY] / "study" / f"{_session_name(request)}.jsonl"
    if not path.exists():
        raise web.HTTPNotFound(text=f"No study log for {path.stem}.")
    return web.json_response(await asyncio.to_thread(Tail(path).read))


async def _events(request: web.Request) -> web.StreamResponse:
    """Stream a session as server-sent events: its whole study log and LLM records first, then each batch of new lines as it is written."""
    logs: Path = request.app[_LOGS_KEY]
    session = _session_name(request)
    study = Tail(logs / "study" / f"{session}.jsonl")
    if not study.path.exists():
        raise web.HTTPNotFound(text=f"No study log for {session}.")

    response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
    await response.prepare(request)
    debug: Tail | None = None
    idle = 0.0
    try:
        while True:
            if debug is None and (path := debug_path(session, logs / "debug")) is not None:
                debug = Tail(path, marker=b'"llm"')
            batch = {
                "study": await asyncio.to_thread(study.read),
                "debug": [llm for record in await asyncio.to_thread(debug.read) if (llm := llm_record(record))] if debug else [],
            }
            if batch["study"] or batch["debug"]:
                await response.write(f"data: {json.dumps(batch, ensure_ascii=False)}\n\n".encode())
                idle = 0.0
            elif idle >= KEEPALIVE_S:
                await response.write(b": keepalive\n\n")
                idle = 0.0
            await asyncio.sleep(POLL_S)
            idle += POLL_S
    except ConnectionResetError:
        pass
    return response


def app(logs: Path = LOGS) -> web.Application:
    """Build the Inspector's web app on a log folder holding study/ and debug/."""
    application = web.Application()
    application[_LOGS_KEY] = logs
    application.router.add_get("/", _page)
    application.router.add_get("/overview", _page)
    application.router.add_get("/sessions", _sessions)
    application.router.add_get("/study", _study)
    application.router.add_get("/events", _events)
    return application


def main():
    """Serve the Inspector on localhost and open it in the browser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", type=Path, default=LOGS, help="log folder holding study/ and debug/ (default: the agent's logs/)")
    parser.add_argument("--port", type=int, default=PORT, help=f"port to serve on (default: {PORT})")
    parser.add_argument("--no-browser", action="store_true", help="do not open the page in the browser")
    arguments = parser.parse_args()

    url = f"http://localhost:{arguments.port}"
    logger.info(f"TrIVA Inspector on {url}, reading {arguments.logs}")
    if not arguments.no_browser:
        webbrowser.open(url)
    web.run_app(app(arguments.logs), host="localhost", port=arguments.port, print=None)


if __name__ == "__main__":
    main()
