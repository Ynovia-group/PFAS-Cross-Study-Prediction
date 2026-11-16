#!/usr/bin/env python3
"""Loopback-only, offline scientific web application. Uploads stay in memory."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import diagnostics

ROOT = Path(__file__).resolve().parent


class Handler(BaseHTTPRequestHandler):
    def send(self, status, body, content_type="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        routes = {"/": ("static/index.html", "text/html; charset=utf-8"),
                  "/app.js": ("static/app.js", "text/javascript; charset=utf-8"),
                  "/style.css": ("static/style.css", "text/css; charset=utf-8"),
                  "/template.csv": ("examples/template.csv", "text/csv; charset=utf-8"),
                  "/example.csv": ("examples/paper_example.csv", "text/csv; charset=utf-8"),
                  "/example_provenance.json": ("examples/provenance.json", "application/json; charset=utf-8")}
        if path == "/api/health":
            return self.send(200, {"status": "ok", "version": diagnostics.VERSION})
        if path not in routes:
            return self.send(404, {"error": "Page not found."})
        relative, kind = routes[path]
        file = ROOT / relative
        if not file.exists():
            return self.send(404, {"error": "The example file is not available."})
        return self.send(200, file.read_bytes(), kind)

    def do_POST(self):
        expected = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
        if self.headers.get("Origin") not in expected | {None}:
            return self.send(403, {"error": "Requests are accepted only from the local application."})
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length < 12_000_000:
                raise diagnostics.InputError("The request size is invalid or exceeds the limit.")
            payload = json.loads(self.rfile.read(length))
            if self.path == "/api/inspect":
                result = diagnostics.inspect(payload)
            elif self.path == "/api/analyze":
                result = diagnostics.analyze(payload)
            else:
                return self.send(404, {"error": "Endpoint not found."})
            self.send(200, result)
        except (diagnostics.InputError, ValueError, KeyError, TypeError) as exc:
            self.send(400, {"error": str(exc)})
        except Exception:
            import traceback
            traceback.print_exc()
            self.send(500, {"error": "Analysis did not complete. Check the local server log for details."})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"PFAS Cross-Study Prediction Diagnostics: http://127.0.0.1:{args.port}", flush=True)
    print("Files are processed in local memory only. Press Ctrl+C to stop the server.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
