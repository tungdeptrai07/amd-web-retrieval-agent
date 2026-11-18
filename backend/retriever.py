"""Intelligent multi-hop web information retrieval agent for AMD AI League Match 4.

Supports natural-language intent decomposition, multi-hop anchor-guided website
navigation, HTML structure extraction, and AMD ROCm / PyTorch-adaptive scoring
with a deterministic pure-Python BM25 + structural boost fallback.
"""

from __future__ import annotations

import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Iterable

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "me", "of", "on", "or",
    "that", "the", "this", "to", "what", "when", "where", "which", "who",
    "why", "with", "find", "get", "tell", "give", "show", "need", "needs",
    "specific", "information", "about", "into", "their", "they", "you", "your",
}

BLOCKED_LINK_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".mp4", ".mp3", ".css", ".js", ".woff2",
)


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms excluding stopwords."""
    raw = re.findall(r"[a-zA-Z0-9_.-]+", text.lower())
    return [tok.strip("._-") for tok in raw if len(tok.strip("._-")) >= 2 and tok.strip("._-") not in STOPWORDS]


def detect_compute_backend() -> dict[str, str]:
    """Detect whether PyTorch / AMD ROCm GPU acceleration is available."""
    try:
        import torch  # type: ignore

        hip_ver = getattr(getattr(torch, "version", None), "hip", None)
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            backend = "AMD ROCm (HIP)" if hip_ver else "PyTorch GPU"
            return {"backend": backend, "device": dev_name, "hip": str(hip_ver or "n/a")}
        return {"backend": "PyTorch CPU", "device": "CPU", "hip": str(hip_ver or "n/a")}
    except Exception:
        return {"backend": "Pure-Python BM25 + Intent Ranker", "device": "CPU", "hip": "n/a"}


@dataclass
class ExtractedLink:
    """Represents a normalized outbound hyperlink and its anchor text."""

    url: str
    anchor_text: str


@dataclass
class ExtractedSection:
    """A coherent semantic block extracted from an HTML document."""

    heading: str
    text: str
    url: str
    depth: int


@dataclass
class ParsedPage:
    """Structured content parsed from a fetched HTML page."""

    url: str
    title: str
    description: str
    sections: list[ExtractedSection]
    links: list[ExtractedLink]
    status_code: int = 200


class _StructuredHTMLParser(HTMLParser):
    """Lightweight HTML parser extracting headings, paragraphs, list items, and links."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self._skip_depth = 0
        self._in_title = False
        self._active_heading_tag: str | None = None
        self._active_block_tag: str | None = None
        self._active_href: str | None = None

        self.title_parts: list[str] = []
        self.meta_description: str = ""
        self.current_heading = "Overview"
        self._heading_buf: list[str] = []
        self._block_buf: list[str] = []
        self._link_buf: list[str] = []

        self.raw_blocks: list[tuple[str, str]] = []
        self.links: list[ExtractedLink] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attr_map = {k.lower(): (v or "") for k, v in attrs}

        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return

        if tag == "title":
            self._in_title = True
        elif tag == "meta" and attr_map.get("name", "").lower() == "description":
            self.meta_description = attr_map.get("content", "").strip()
        elif tag in {"h1", "h2", "h3", "h4"}:
            self._flush_block()
            self._active_heading_tag = tag
            self._heading_buf = []
        elif tag in {"p", "li", "tr", "td", "th", "blockquote", "pre", "code", "dt", "dd"}:
            if self._active_block_tag is None:
                self._active_block_tag = tag
                self._block_buf = []
        elif tag == "a":
            href = attr_map.get("href", "").strip()
            if href and not href.startswith(("#", "javascript:", "mailto:", "tel:")):
                resolved = urllib.parse.urljoin(self.base_url, href)
                parsed = urllib.parse.urlparse(resolved)
                if parsed.scheme in {"http", "https"} and not parsed.path.lower().endswith(BLOCKED_LINK_EXTENSIONS):
                    clean_url = urllib.parse.urldefrag(resolved)[0]
                    self._active_href = clean_url
                    self._link_buf = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            return

        if tag == "title":
            self._in_title = False
        elif tag == self._active_heading_tag:
            heading_text = _normalize_ws(" ".join(self._heading_buf))
            if heading_text:
                self.current_heading = heading_text
            self._active_heading_tag = None
            self._heading_buf = []
        elif tag == self._active_block_tag:
            self._flush_block()
        elif tag == "a" and self._active_href is not None:
            anchor = _normalize_ws(" ".join(self._link_buf))
            if anchor or self._active_href:
                self.links.append(ExtractedLink(url=self._active_href, anchor_text=anchor))
            self._active_href = None
            self._link_buf = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title:
            self.title_parts.append(data)
        if self._active_heading_tag is not None:
            self._heading_buf.append(data)
        if self._active_block_tag is not None:
            self._block_buf.append(data)
        if self._active_href is not None:
            self._link_buf.append(data)

    def _flush_block(self) -> None:
        if self._active_block_tag is None:
            return
        text = _normalize_ws(" ".join(self._block_buf))
        if len(text) >= 18:
            self.raw_blocks.append((self.current_heading, text))
        self._active_block_tag = None
        self._block_buf = []


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_html_document(html: str, url: str, depth: int = 0) -> ParsedPage:
    """Parse raw HTML into titled semantic sections and clean outbound links."""
    parser = _StructuredHTMLParser(url)
    parser.feed(html)
    parser.close()

    title = _normalize_ws(" ".join(parser.title_parts)) or url
    sections: list[ExtractedSection] = []
    seen_texts: set[str] = set()

    if parser.meta_description:
        sections.append(
            ExtractedSection(
                heading="Meta Summary",
                text=parser.meta_description,
                url=url,
                depth=depth,
            )
        )
        seen_texts.add(parser.meta_description)

    for heading, body in parser.raw_blocks:
        if body in seen_texts:
            continue
        seen_texts.add(body)
        sections.append(ExtractedSection(heading=heading, text=body, url=url, depth=depth))

    # Deduplicate links while preserving the richest anchor text.
    link_map: dict[str, str] = {}
    for item in parser.links:
        prev = link_map.get(item.url, "")
        if len(item.anchor_text) > len(prev):
            link_map[item.url] = item.anchor_text
    deduped_links = [ExtractedLink(url=u, anchor_text=a) for u, a in link_map.items()]

    return ParsedPage(
        url=url,
        title=title,
        description=parser.meta_description,
        sections=sections,
        links=deduped_links,
    )


def fetch_url(url: str, timeout_s: float = 8.0) -> tuple[str, int]:
    """Fetch a single HTTP/HTTPS URL and decode text safely."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ROCmScout-RetrievalAgent/1.0 (+https://lablab.build)",
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        status = getattr(resp, "status", 200)
        raw = resp.read(1_500_000)
        charset = resp.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="replace"), status


@dataclass
class RetrievalConfig:
    """Controls bounded multi-hop exploration and scoring."""

    max_pages: int = 8
    max_depth: int = 2
    same_domain_only: bool = True
    top_k_evidence: int = 5
    timeout_s: float = 8.0


@dataclass
class EvidenceMatch:
    """Ranked retrieved passage with provenance and confidence."""

    score: float
    confidence: float
    url: str
    page_title: str
    heading: str
    snippet: str
    matched_terms: list[str]
    depth: int


@dataclass
class HopTrace:
    """Audit trail entry for a single page visited by the navigating agent."""

    step: int
    url: str
    depth: int
    status: str
    title: str
    links_discovered: int
    top_next_link: str | None = None
    top_link_score: float = 0.0
    elapsed_ms: float = 0.0


@dataclass
class RetrievalReport:
    """Final structured retrieval output returned by the agent."""

    query: str
    query_terms: list[str]
    answer_summary: str
    overall_confidence: float
    evidences: list[EvidenceMatch]
    navigation_trace: list[HopTrace]
    compute_backend: dict[str, str]
    pages_visited: int
    sections_indexed: int
    errors: list[str] = field(default_factory=list)


def score_link_relevance(link: ExtractedLink, query_terms: list[str], raw_query: str) -> float:
    """Score an outbound link to decide which path the agent should navigate next."""
    combined = f"{link.anchor_text} {urllib.parse.urlparse(link.url).path.replace('/', ' ')}".lower()
    tokens = set(tokenize(combined))
    if not query_terms:
        return 0.1

    overlap = sum(1 for term in query_terms if term in tokens)
    partial = sum(0.45 for term in query_terms if term not in tokens and term in combined)
    phrase_bonus = 1.5 if any(len(w) > 4 and w in combined for w in raw_query.lower().split()) else 0.0
    return round(overlap * 2.0 + partial + phrase_bonus, 4)


def rank_sections_bm25(
    sections: list[tuple[ExtractedSection, str]],
    query_terms: list[str],
    raw_query: str,
) -> list[EvidenceMatch]:
    """Rank extracted page sections using BM25 + heading & exact-phrase boosts."""
    if not sections or not query_terms:
        return []

    docs_tokens: list[list[str]] = []
    heading_tokens: list[set[str]] = []
    for sec, title in sections:
        docs_tokens.append(tokenize(f"{title} {sec.heading} {sec.text}"))
        heading_tokens.append(set(tokenize(f"{title} {sec.heading}")))

    num_docs = len(docs_tokens)
    avg_len = sum(len(d) for d in docs_tokens) / max(num_docs, 1)
    df: Counter[str] = Counter()
    for d in docs_tokens:
        for term in set(d):
            df[term] += 1

    k1 = 1.5
    b = 0.75
    query_lower = raw_query.lower()
    raw_matches: list[tuple[float, ExtractedSection, str, list[str]]] = []

    for idx, (sec, title) in enumerate(sections):
        d = docs_tokens[idx]
        if not d:
            continue
        tf = Counter(d)
        doc_len = len(d)
        bm25 = 0.0
        matched: list[str] = []

        for q in set(query_terms):
            freq = tf.get(q, 0)
            if freq == 0:
                continue
            matched.append(q)
            idf = math.log(1.0 + (num_docs - df[q] + 0.5) / (df[q] + 0.5))
            denom = freq + k1 * (1.0 - b + b * (doc_len / max(avg_len, 1.0)))
            bm25 += idf * ((freq * (k1 + 1.0)) / max(denom, 1e-6))

        if not matched:
            continue

        # Structural heading + multi-term coverage bonuses.
        heading_hits = sum(1 for q in matched if q in heading_tokens[idx])
        coverage = len(matched) / max(len(set(query_terms)), 1)
        exact_boost = 0.75 if len(matched) >= 2 and any(
            f"{a} {b_tok}" in sec.text.lower()
            for a, b_tok in zip(query_terms, query_terms[1:])
        ) else 0.0

        total_score = bm25 * (1.0 + 0.35 * coverage) + 0.45 * heading_hits + exact_boost
        if query_lower in sec.text.lower():
            total_score += 1.0

        raw_matches.append((total_score, sec, title, sorted(matched)))

    if not raw_matches:
        return []

    raw_matches.sort(key=lambda item: item[0], reverse=True)
    top_score = raw_matches[0][0]
    unique_query_cnt = max(len(set(query_terms)), 1)

    results: list[EvidenceMatch] = []
    for score, sec, title, matched in raw_matches:
        coverage_ratio = len(matched) / unique_query_cnt
        rel_ratio = score / max(top_score, 1e-6)
        confidence = round(min(0.99, 0.55 * coverage_ratio + 0.45 * rel_ratio), 3)
        results.append(
            EvidenceMatch(
                score=round(score, 4),
                confidence=confidence,
                url=sec.url,
                page_title=title,
                heading=sec.heading,
                snippet=sec.text,
                matched_terms=matched,
                depth=sec.depth,
            )
        )
    return results


class WebRetrievalAgent:
    """Autonomous multi-hop web navigation and information retrieval agent."""

    def __init__(self, config: RetrievalConfig | None = None, fetcher=fetch_url) -> None:
        self.config = config or RetrievalConfig()
        self.fetcher = fetcher

    def retrieve(self, query: str, seed_urls: Iterable[str]) -> RetrievalReport:
        """Navigate from seed URLs, index pages, and extract grounded answers."""
        clean_seeds = [u.strip() for u in seed_urls if u and u.strip()]
        query_terms = tokenize(query)
        compute_info = detect_compute_backend()

        if not clean_seeds:
            return RetrievalReport(
                query=query,
                query_terms=query_terms,
                answer_summary="No seed URLs provided.",
                overall_confidence=0.0,
                evidences=[],
                navigation_trace=[],
                compute_backend=compute_info,
                pages_visited=0,
                sections_indexed=0,
                errors=["At least one valid HTTP/HTTPS seed URL is required."],
            )

        allowed_hosts = {urllib.parse.urlparse(u).netloc.lower() for u in clean_seeds if urllib.parse.urlparse(u).netloc}
        # Priority queue stored as (priority_score, depth, url) — popped by highest priority first.
        frontier: list[tuple[float, int, str]] = [(100.0, 0, u) for u in clean_seeds]
        visited: set[str] = set()
        indexed_sections: list[tuple[ExtractedSection, str]] = []
        trace: list[HopTrace] = []
        errors: list[str] = []

        while frontier and len(visited) < self.config.max_pages:
            frontier.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            _, depth, url = frontier.pop(0)
            canonical = urllib.parse.urldefrag(url)[0]
            if canonical in visited:
                continue
            visited.add(canonical)

            t0 = time.perf_counter()
            try:
                html, status = self.fetcher(canonical, self.config.timeout_s)
                if status >= 400:
                    raise RuntimeError(f"HTTP {status}")
            except Exception as exc:
                elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
                err_msg = f"{canonical}: {exc}"
                errors.append(err_msg)
                trace.append(
                    HopTrace(
                        step=len(trace) + 1,
                        url=canonical,
                        depth=depth,
                        status=f"ERROR ({exc})",
                        title="",
                        links_discovered=0,
                        elapsed_ms=elapsed,
                    )
                )
                continue

            parsed = parse_html_document(html, canonical, depth=depth)
            for sec in parsed.sections:
                indexed_sections.append((sec, parsed.title))

            best_link_url: str | None = None
            best_link_score = 0.0
            discovered_count = 0

            if depth < self.config.max_depth:
                scored_candidates: list[tuple[float, str]] = []
                for link in parsed.links:
                    target_host = urllib.parse.urlparse(link.url).netloc.lower()
                    if self.config.same_domain_only and target_host not in allowed_hosts:
                        continue
                    if link.url in visited:
                        continue
                    l_score = score_link_relevance(link, query_terms, query)
                    scored_candidates.append((l_score, link.url))

                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                discovered_count = len(scored_candidates)
                if scored_candidates:
                    best_link_score, best_link_url = scored_candidates[0]
                for l_score, l_url in scored_candidates[:12]:
                    frontier.append((l_score, depth + 1, l_url))

            elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
            trace.append(
                HopTrace(
                    step=len(trace) + 1,
                    url=canonical,
                    depth=depth,
                    status="OK",
                    title=parsed.title,
                    links_discovered=discovered_count,
                    top_next_link=best_link_url,
                    top_link_score=best_link_score,
                    elapsed_ms=elapsed,
                )
            )

        ranked = rank_sections_bm25(indexed_sections, query_terms, query)[: self.config.top_k_evidence]
        if ranked:
            top_bullets = [
                f"[{idx + 1}] ({item.heading} — {item.url}) {item.snippet}"
                for idx, item in enumerate(ranked[:3])
            ]
            answer_summary = "\n".join(top_bullets)
            overall_conf = ranked[0].confidence
        else:
            answer_summary = "No relevant sections matched the query across the visited pages."
            overall_conf = 0.0

        return RetrievalReport(
            query=query,
            query_terms=query_terms,
            answer_summary=answer_summary,
            overall_confidence=overall_conf,
            evidences=ranked,
            navigation_trace=trace,
            compute_backend=compute_info,
            pages_visited=len(visited),
            sections_indexed=len(indexed_sections),
            errors=errors,
        )
