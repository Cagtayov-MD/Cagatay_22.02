from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
DATA_FILE = ROOT / "data" / "control_tower_state.json"
HOST = "127.0.0.1"
PORT = 8011


class ControlTowerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/health":
            self._send_json({"ok": True, "service": "control_tower_web"})
            return

        if parsed.path == "/api/state":
            payload = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            self._send_json(payload)
            return

        if parsed.path in {"/", "/index.html"}:
            self.path = "/index.html"

        return super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format, *args):
        print("[control_tower_web]", format % args)

    def _send_json(self, payload: dict):
        raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main():
    server = ThreadingHTTPServer((HOST, PORT), ControlTowerHandler)
    print(f"Control Tower Web hazir: http://{HOST}:{PORT}")
    print("Kaynak modu: read-only mock")
    print("Aktif Project/ agacina dokunulmaz.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDurduruluyor...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
