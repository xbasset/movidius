"""Local camera / dataset demo. No image recording and no cloud inference."""

import argparse
from http.server import ThreadingHTTPServer
import json
from urllib.parse import urlsplit

from movidius_lab.dino import DATA_DIR, ROOT
from movidius_lab.dino_live import EXAMPLES, LiveDino
from scripts.serve_crivo import Workbench, handler


class DinoWorkbench(Workbench):
    def initialize(self):
        try:
            self.engine = LiveDino(self.update)
            self.update("Prêt")
        except Exception as error:
            self.error = str(error)

    def run(self, kind, payload):
        try:
            operation = self.engine.calibrate if kind == "reference" else self.engine.compare
            result = operation(payload, self.update)
            with self.lock:
                self.job.update(state="complete", result=result)
                self.progress = "Prêt"
        except Exception as error:
            with self.lock:
                self.job.update(state="error", error=str(error))
                self.progress = "Prêt"


def dino_handler(workbench, port):
    class Handler(handler(workbench, port)):
        def send(self, status, value, content_type="application/json; charset=utf-8"):
            payload = json.dumps(value, allow_nan=False).encode() if isinstance(value, dict) else value
            self.send_response(status)
            for key, value in [("Content-Type", content_type), ("Content-Length", str(len(payload))),
                               ("Cache-Control", "no-store"), ("X-Content-Type-Options", "nosniff"),
                               ("Permissions-Policy", "camera=(self), microphone=()"),
                               ("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")]:
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if not self.local():
                return
            path = urlsplit(self.path).path
            if path == "/api/status" or path.startswith("/api/jobs/"):
                return super().do_GET()
            if path == "/api/reference":
                image = workbench.engine.reference_png if workbench.engine else None
                return self.send(200, image, "image/png") if image else self.send(404, {"error": "Référence absente"})
            if path.startswith("/example/"):
                key = path.rsplit("/", 1)[1]
                if key in EXAMPLES:
                    return self.send(200, (DATA_DIR / "bottle" / EXAMPLES[key]).read_bytes(), "image/png")
            assets = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path not in assets:
                return self.send(404, {"error": "Page inconnue"})
            name, mime = assets[path]
            return self.send(200, (ROOT / "web/dino" / name).read_bytes(), mime)

        def do_POST(self):
            if not self.local():
                return
            if urlsplit(self.path).path != "/api/run":
                return self.send(404, {"error": "Route inconnue"})
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.send(415, {"error": "JSON requis"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1_500_000:
                    raise ValueError("Image trop grande ou requête vide")
                payload = json.loads(self.rfile.read(size))
                kind = payload.get("kind", "compare")
                if kind not in ["compare", "reference"]:
                    raise ValueError("Action inconnue")
                return self.send(202, workbench.start(kind, payload))
            except (ValueError, TypeError, AttributeError) as error:
                return self.send(400, {"error": str(error)})
    return Handler


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    workbench, server = DinoWorkbench(), None
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), dino_handler(workbench, args.port))
        print(f"DINOv2 local : http://127.0.0.1:{args.port}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.server_close()
        workbench.close()


if __name__ == "__main__":
    main()
