"""HTTP API and static web UI server for ROCmScout Web Retrieval Agent."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.retriever import RetrievalConfig, WebRetrievalAgent, detect_compute_backend


class RetrievalHTTPHandler(SimpleHTTPRequestHandler):
    """Serves the ROCmScout UI and JSON retrieval endpoints."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format: str, *args) -> None:
        return

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._send_json(
                {
                    "status": "ok",
                    "service": "ROCmScout Intelligent Web Retrieval Agent",
                    "compute": detect_compute_backend(),
                }
            )
            return
        if self.path == "/" or self.path == "":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/retrieve":
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"error": "Invalid Content-Length"}, status=HTTPStatus.BAD_REQUEST)
            return

        raw = self.rfile.read(min(length, 256_000))
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST)
            return

        query = str(data.get("query", "")).strip()
        raw_urls = data.get("seedUrls", [])
        if isinstance(raw_urls, str):
            seed_urls = [line.strip() for line in raw_urls.splitlines() if line.strip()]
        elif isinstance(raw_urls, list):
            seed_urls = [str(u).strip() for u in raw_urls if str(u).strip()]
        else:
            seed_urls = []

        if not query or not seed_urls:
            self._send_json(
                {"error": "Both 'query' and at least one 'seedUrls' item are required."},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        config = RetrievalConfig(
            max_pages=max(1, min(int(data.get("maxPages", 6)), 15)),
            max_depth=max(0, min(int(data.get("maxDepth", 2)), 3)),
            same_domain_only=bool(data.get("sameDomainOnly", True)),
            top_k_evidence=max(1, min(int(data.get("topK", 5)), 10)),
        )
        agent = WebRetrievalAgent(config=config)
        report = agent.retrieve(query=query, seed_urls=seed_urls)
        self._send_json(asdict(report))


def create_server(host: str = "127.0.0.1", port: int = 8089) -> ThreadingHTTPServer:
    """Create and return the configured retrieval HTTP server."""
    return ThreadingHTTPServer((host, port), RetrievalHTTPHandler)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8089"))
    srv = create_server(port=port)
    print(f"ROCmScout Retrieval Agent running on http://127.0.0.1:{port}", flush=True)
    srv.serve_forever()
