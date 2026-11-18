# ROCmScout — Intelligent Web Information Retrieval Agent
**AMD AI League · Match 4 Submission Deck**

---

## Slide 1: Problem
- Standard search engines return static links; single-page LLM wrappers miss crucial information buried 1–2 clicks deep inside documentation, pricing tables, or technical hubs.
- Users asking natural-language questions need an autonomous agent that **navigates hyperlinks intelligently**, verifies facts across hops, and cites exact provenance.

## Slide 2: Solution — ROCmScout
- **Autonomous Multi-Hop Navigator**: Uses anchor-text and URL path intent scoring (`score_link_relevance`) to prioritize the most promising outbound links at each hop.
- **Structured HTML Knowledge Extractor**: Extracts titled semantic blocks (`h1`–`h4`, `p`, `li`, `tr`, `code`, meta tags) while stripping script/style noise.
- **ROCm-Adaptive Hybrid Scorer**: Detects AMD ROCm / HIP GPUs via PyTorch (`detect_compute_backend`) and pairs with deterministic BM25 + structural heading + exact-phrase ranking.

## Slide 3: Verification & Live Demo
- **4/4 Automated Tests (`test_retriever.py`)**: Verifies multi-hop discovery across interconnected pages, invalid-input safety, and end-to-end REST retrieval.
- **Live Verification**: Tested against `https://docs.trnch.fun/getting-started` — autonomously navigated 3–6 pages (depth 0–2), indexed 179 sections, and retrieved exact `chainId 4663`, gas token (`ETH`), and stablecoin (`USDG`) details with 72.5%–79% confidence.
- **Interactive UI**: Real-time navigation trace and ranked citation explorer running on `http://127.0.0.1:8089`.
