"""Unit and end-to-end tests for the ROCmScout Web Retrieval Agent."""

from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.retriever import RetrievalConfig, WebRetrievalAgent, parse_html_document
from backend.server import create_server

MOCK_SITE = {
    "/": """
    <html><head><title>ROCm Developer Portal</title>
    <meta name="description" content="Official AMD ROCm hub for GPU computing and AI workloads.">
    </head><body>
    <h1>Welcome to ROCm Hub</h1>
    <p>Explore documentation for HIP kernels, PyTorch acceleration, and MI300X deployment.</p>
    <nav>
      <a href="/blog/news">General Community Blog</a>
      <a href="/docs/pytorch-rocm-install">PyTorch ROCm Installation and GPU Requirements</a>
      <a href="/pricing">Cloud Credits Overview</a>
    </nav>
    </body></html>
    """,
    "/blog/news": """
    <html><head><title>Community News</title></head><body>
    <h1>Community Events</h1>
    <p>Monthly developer meetups and hackathon winners announcement.</p>
    </body></html>
    """,
    "/docs/pytorch-rocm-install": """
    <html><head><title>PyTorch ROCm Setup Guide</title></head><body>
    <h1>Installing PyTorch with AMD ROCm 6.2</h1>
    <p>To run PyTorch models on AMD Instinct GPUs, install torch with the ROCm 6.2 index URL and verify torch.cuda.is_available returns True.</p>
    <h2>Persistent Storage Limit</h2>
    <p>Hackathon notebooks provide 25 GB of persistent workspace storage and 3 hours per day of active GPU runtime.</p>
    <a href="/docs/hip-kernels">Deep Dive: Custom HIP Kernels</a>
    </body></html>
    """,
    "/docs/hip-kernels": """
    <html><head><title>Custom HIP Kernels</title></head><body>
    <h1>Writing HIP Kernels</h1>
    <p>HIP allows portable C++ GPU kernels across AMD Instinct and Radeon hardware.</p>
    </body></html>
    """,
    "/pricing": """
    <html><head><title>Credits</title></head><body>
    <h1>Credits</h1>
    <p>Top 50 valid participants earn 100 USD in GPU credits.</p>
    </body></html>
    """,
}


class _MockSiteHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        html = MOCK_SITE.get(self.path)
        if html is None:
            self.send_response(404)
            self.end_headers()
            return
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class TestWebRetrievalAgent(unittest.TestCase):
    """Verifies HTML parsing, anchor-guided multi-hop navigation, ranking, and API."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mock_site = HTTPServer(("127.0.0.1", 0), _MockSiteHandler)
        cls.site_port = cls.mock_site.server_address[1]
        threading.Thread(target=cls.mock_site.serve_forever, daemon=True).start()

        cls.api_server = create_server(port=0)
        cls.api_port = cls.api_server.server_address[1]
        threading.Thread(target=cls.api_server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.mock_site.shutdown()
        cls.mock_site.server_close()
        cls.api_server.shutdown()
        cls.api_server.server_close()

    def test_parse_html_extracts_headings_and_links(self) -> None:
        parsed = parse_html_document(MOCK_SITE["/"], "http://example.com/")
        self.assertEqual(parsed.title, "ROCm Developer Portal")
        self.assertGreaterEqual(len(parsed.sections), 2)
        urls = {link.url for link in parsed.links}
        self.assertIn("http://example.com/docs/pytorch-rocm-install", urls)

    def test_multi_hop_navigation_finds_target_fact(self) -> None:
        agent = WebRetrievalAgent(RetrievalConfig(max_pages=4, max_depth=2, top_k_evidence=3))
        report = agent.retrieve(
            query="What is the persistent storage limit and PyTorch ROCm 6.2 installation setup?",
            seed_urls=[f"http://127.0.0.1:{self.site_port}/"],
        )
        self.assertGreater(report.overall_confidence, 0.4)
        self.assertTrue(len(report.evidences) >= 1)
        combined = " ".join(ev.snippet for ev in report.evidences)
        self.assertIn("25 GB", combined)
        self.assertEqual(report.navigation_trace[0].top_next_link, f"http://127.0.0.1:{self.site_port}/docs/pytorch-rocm-install")

    def test_empty_seed_urls_returns_clean_error(self) -> None:
        agent = WebRetrievalAgent()
        report = agent.retrieve(query="test", seed_urls=[])
        self.assertEqual(report.pages_visited, 0)
        self.assertTrue(len(report.errors) > 0)

    def test_http_api_retrieve_end_to_end(self) -> None:
        payload = json.dumps(
            {
                "query": "How much GPU runtime per day and storage limit do hackathon notebooks provide?",
                "seedUrls": [f"http://127.0.0.1:{self.site_port}/"],
                "maxPages": 4,
                "maxDepth": 2,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.api_port}/api/retrieve",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
        self.assertIn("25 GB", data["answer_summary"])
        self.assertGreaterEqual(data["pages_visited"], 2)


if __name__ == "__main__":
    unittest.main()
