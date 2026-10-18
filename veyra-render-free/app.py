"""Render entrypoint: public, temporary, session-isolated version of Veyra."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from collections import OrderedDict, deque
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np  # noqa: F401 - verifies the small public inference dependency at startup

import veyra as core
from public_engine import generate


HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "7860"))
SESSION_RE = re.compile(r"^[a-f0-9]{32}$")
SESSION_TTL = 60 * 60
MAX_SESSIONS = 32
MAX_DOCS_PER_SESSION = 3
MAX_PUBLIC_UPLOAD = 3 * 1024 * 1024

_pool: OrderedDict[str, dict] = OrderedDict()
_pool_guard = threading.Lock()
_local = threading.local()


class _SessionDB(sqlite3.Connection):
    # Core handlers call close() after each request. Preserve this in-memory DB
    # until the browser session expires or the server restarts.
    def close(self) -> None:
        return

    def dispose(self) -> None:
        super().close()


def _new_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False, factory=_SessionDB, timeout=4)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE documents (
        id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL,
        pages_json TEXT NOT NULL, char_count INTEGER NOT NULL, added_at TEXT NOT NULL
    )""")
    conn.execute("""CREATE TABLE messages (
        id INTEGER PRIMARY KEY, role TEXT NOT NULL, content TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    conn.commit()
    return conn


def _discard_session(session_id: str, entry: dict) -> None:
    try:
        rows = entry["db"].execute("SELECT id FROM documents").fetchall()
        for row in rows:
            (core.UPLOAD_DIR / row["id"]).unlink(missing_ok=True)
    except Exception:
        pass
    try:
        entry["db"].dispose()
    except Exception:
        pass
    _pool.pop(session_id, None)


def _session_entry(session_id: str) -> dict:
    now = time.monotonic()
    with _pool_guard:
        for old_id, old in list(_pool.items()):
            if now - old["last"] > SESSION_TTL and old["lock"].acquire(blocking=False):
                try:
                    _discard_session(old_id, old)
                finally:
                    old["lock"].release()
        entry = _pool.get(session_id)
        if entry is None:
            if len(_pool) >= MAX_SESSIONS:
                for old_id, old in list(_pool.items()):
                    if old["lock"].acquire(blocking=False):
                        try:
                            _discard_session(old_id, old)
                        finally:
                            old["lock"].release()
                        break
            if len(_pool) >= MAX_SESSIONS:
                raise RuntimeError("Veyra está ocupada. Prueba de nuevo en un momento.")
            entry = {"db": _new_db(), "lock": threading.Lock(), "last": now,
                     "requests": deque(), "searches": deque()}
            _pool[session_id] = entry
        entry["last"] = now
        entry["lock"].acquire()
        _pool.move_to_end(session_id)
        return entry


def _session_db() -> sqlite3.Connection:
    session_id = getattr(_local, "session_id", None)
    if not session_id:
        raise RuntimeError("Falta una sesión de navegador.")
    return _pool[session_id]["db"]


def _public_model_status() -> dict:
    info = core.DATA_DIR / "modelo_generativo_veyra.json"
    try:
        gen = json.loads(info.read_text(encoding="utf-8"))
        gen_ok = gen.get("format") == "veyra-tiny-gru-v1" and gen.get("trained_from_scratch")
    except Exception:
        gen, gen_ok = {}, False
    return {"classifier": (core.DATA_DIR / "modelo_veyra_propio.json").is_file(),
            "generative": bool(gen_ok and (core.DATA_DIR / "modelo_generativo_publico.json").is_file()),
            "generation": {k: gen.get(k) for k in ("tokens", "epochs", "training_seconds", "device")}}


core._db = _session_db
core._own_model_status = _public_model_status
core._generate_own = generate
core.MAX_UPLOAD = MAX_PUBLIC_UPLOAD
core.PAGE = core.PAGE.replace(
    "async function api(path,opts={}){const r=await fetch(path,{...opts,headers:{",
    "const veyraSession=(()=>{let id=sessionStorage.getItem('veyra_session');if(!/^[a-f0-9]{32}$/.test(id||'')){id=(crypto.randomUUID?crypto.randomUUID().replaceAll('-',''):Array.from(crypto.getRandomValues(new Uint8Array(16)),x=>x.toString(16).padStart(2,'0')).join(''));sessionStorage.setItem('veyra_session',id)}return id})();async function api(path,opts={}){const r=await fetch(path,{...opts,headers:{'X-Veyra-Session':veyraSession,"
)


class PublicHandler(core.Handler):
    def _send_text(self, status: int, content_type: str, text: str) -> None:
        payload = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/health":
            self._send_text(200, "text/plain; charset=utf-8", "ok")
            return
        if path == "/robots.txt":
            self._send_text(200, "text/plain; charset=utf-8", "User-agent: *\nAllow: /\n")
            return
        if path == "/sitemap.xml":
            public_url = os.environ.get("VEYRA_PUBLIC_URL", "").strip().rstrip("/")
            if public_url:
                xml = f'<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>{public_url}/</loc></url></urlset>'
                self._send_text(200, "application/xml; charset=utf-8", xml)
            else:
                self._send_text(503, "text/plain; charset=utf-8", "VEYRA_PUBLIC_URL is not configured")
            return
        if path.startswith("/api/"):
            self._with_session(super().do_GET)
            return
        super().do_GET()

    def do_POST(self) -> None:
        self._with_session(super().do_POST)

    def _with_session(self, callback) -> None:
        session_id = self.headers.get("X-Veyra-Session", "")
        if not SESSION_RE.fullmatch(session_id):
            core._json(self, 400, {"error": "Esta pestaña no tiene una sesión local válida. Recárgala e inténtalo de nuevo."})
            return
        try:
            entry = _session_entry(session_id)
        except RuntimeError as exc:
            core._json(self, 503, {"error": str(exc)})
            return
        _local.session_id = session_id
        try:
            now = time.monotonic()
            while entry["requests"] and now - entry["requests"][0] > 600:
                entry["requests"].popleft()
            if len(entry["requests"]) >= 120:
                core._json(self, 429, {"error": "Esta sesión alcanzó el límite temporal de solicitudes. Espera unos minutos."})
                return
            entry["requests"].append(now)
            path = urlsplit(self.path).path
            if path == "/api/upload":
                count = entry["db"].execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                if count >= MAX_DOCS_PER_SESSION:
                    core._json(self, 413, {"error": f"El modo público permite hasta {MAX_DOCS_PER_SESSION} documentos temporales por sesión."})
                    return
            callback()
        finally:
            try:
                del _local.session_id
            except AttributeError:
                pass
            entry["last"] = time.monotonic()
            entry["lock"].release()

    def log_message(self, fmt: str, *args: object) -> None:
        # Do not log chat text, document content, session IDs or query strings.
        print("[Veyra public] " + fmt.split("?", 1)[0] % args)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), PublicHandler)
    server.daemon_threads = True
    print(f"Veyra public listening on {HOST}:{PORT}; no user data is persisted to disk.")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        with _pool_guard:
            for sid, entry in list(_pool.items()):
                if entry["lock"].acquire(blocking=False):
                    try:
                        _discard_session(sid, entry)
                    finally:
                        entry["lock"].release()


if __name__ == "__main__":
    main()
