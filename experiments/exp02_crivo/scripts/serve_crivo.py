"""Local-only Crivo workbench; run as a module from the repository root."""

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path

from experiments.exp02_crivo.live import LiveCrivo
from movidius_lab.workbench import Workbench, handler

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    workbench = Workbench(LiveCrivo, thread_name="crivo")
    server = None
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(workbench, args.port, WEB_DIR))
        print(f"Crivo local : http://127.0.0.1:{args.port}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.server_close()
        workbench.close()


if __name__ == "__main__":
    main()
