"""Shared loopback HTTP transport and serialized model-worker lifecycle."""

from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler
import json
import threading
from urllib.parse import urlsplit
import uuid


class Workbench:
    def __init__(self, engine_factory, thread_name="inference"):
        self.engine_factory = engine_factory
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name)
        self.engine, self.error = None, None
        self.progress, self.job = "Chargement des modèles…", None
        self.executor.submit(self.initialize)

    def update(self, value):
        with self.lock:
            self.progress = value

    def initialize(self):
        try:
            self.engine = self.engine_factory(self.update)
            self.update("Prêt")
        except Exception as error:
            self.error = str(error)

    def status(self):
        with self.lock:
            return {"ready": self.engine is not None, "error": self.error,
                    "progress": self.progress, "busy": bool(self.job and self.job["state"] == "running"),
                    **(self.engine.status() if self.engine else {})}

    def start(self, kind, message):
        with self.lock:
            if not self.engine:
                raise ValueError("Modèles en cours de chargement")
            if self.job and self.job["state"] == "running":
                raise ValueError("Une mesure est déjà en cours")
            self.job = {"id": uuid.uuid4().hex, "state": "running", "kind": kind}
            identifier = self.job["id"]
        self.executor.submit(self.run, kind, message)
        return {"id": identifier}

    def run(self, kind, message):
        try:
            if kind == "reconnect":
                value = self.engine.reconnect()
            else:
                value = self.engine.compare(message, self.update)
            with self.lock:
                self.job.update(state="complete", result=value)
                self.progress = "Prêt"
        except Exception as error:
            with self.lock:
                self.job.update(state="error", error=str(error))
                self.progress = "Prêt"

    def close(self):
        def release():
            if self.engine:
                self.engine.close()
        self.executor.submit(release).result()
        self.executor.shutdown(wait=True)


def handler(workbench, port, web_root):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Never persist messages or request bodies.

        def send(self, status, value, content_type="application/json; charset=utf-8"):
            payload = json.dumps(value, allow_nan=False).encode() if isinstance(value, dict) else value
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def local(self):
            allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host") not in allowed:
                self.send(403, {"error": "Hôte local requis"})
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + host for host in allowed}:
                self.send(403, {"error": "Origine locale requise"})
                return False
            return True

        def do_GET(self):
            if not self.local():
                return
            path = urlsplit(self.path).path
            if path == "/api/status":
                return self.send(200, workbench.status())
            if path.startswith("/api/jobs/"):
                with workbench.lock:
                    job = dict(workbench.job) if workbench.job else {}
                    job["progress"] = workbench.progress
                return self.send(200 if job.get("id") == path.rsplit("/", 1)[1] else 404, job)
            assets = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path not in assets:
                return self.send(404, {"error": "Page inconnue"})
            name, mime = assets[path]
            self.send(200, (web_root / name).read_bytes(), mime)

        def do_POST(self):
            if not self.local():
                return
            if urlsplit(self.path).path != "/api/run":
                return self.send(404, {"error": "Route inconnue"})
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.send(415, {"error": "JSON requis"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384:
                    raise ValueError("Requête trop grande ou vide")
                payload = json.loads(self.rfile.read(size))
                kind = payload.get("kind", "compare")
                if kind not in {"compare", "reconnect"}:
                    raise ValueError("Action inconnue")
                result = workbench.start(kind, payload.get("message", ""))
                self.send(202, result)
            except (ValueError, TypeError, AttributeError) as error:
                self.send(400, {"error": str(error)})
    return Handler
