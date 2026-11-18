const PRESETS = {
  trnch: {
    query: "What chainId does Robinhood Chain use, what are the swap fees, and which routers are aggregated?",
    seeds: "https://docs.trnch.fun/getting-started",
  },
  rocm: {
    query: "What is AMD ROCm, which programming models like HIP does it support, and what GPUs run it?",
    seeds: "https://en.wikipedia.org/wiki/ROCm",
  },
  python: {
    query: "How do coroutines and asyncio.run execute asynchronous tasks in Python?",
    seeds: "https://docs.python.org/3/library/asyncio-task.html",
  },
};

const queryInput = document.getElementById("queryInput");
const seedsInput = document.getElementById("seedsInput");
const maxPages = document.getElementById("maxPages");
const maxDepth = document.getElementById("maxDepth");
const sameDomain = document.getElementById("sameDomain");
const runBtn = document.getElementById("runBtn");
const statusBar = document.getElementById("statusBar");
const summaryBox = document.getElementById("summaryBox");
const evidenceList = document.getElementById("evidenceList");
const traceList = document.getElementById("traceList");
const confidencePill = document.getElementById("confidencePill");
const metricsPill = document.getElementById("metricsPill");
const computeBadge = document.getElementById("computeBadge");

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text ?? "";
  return div.innerHTML;
}

async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error("Static host");
    const contentType = res.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
      throw new Error("Static GitHub Pages host cannot run Python live crawling — run `python backend/server.py` locally for arbitrary live URLs.");
    }
    const data = await res.json();
    const c = data.compute || {};
    computeBadge.textContent = `${c.backend || "Ready"} (${c.device || "CPU"})`;
  } catch {
    computeBadge.textContent = "Static Preview Mode (run backend/server.py for live crawling)";
  }
}

document.querySelectorAll(".preset-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const p = PRESETS[btn.dataset.preset];
    if (!p) return;
    queryInput.value = p.query;
    seedsInput.value = p.seeds;
  });
});

// Load default preset on initial open.
queryInput.value = PRESETS.trnch.query;
seedsInput.value = PRESETS.trnch.seeds;
loadHealth();

runBtn.addEventListener("click", async () => {
  const query = queryInput.value.trim();
  const seedUrls = seedsInput.value.split("\n").map((s) => s.trim()).filter(Boolean);
  if (!query || seedUrls.length === 0) {
    statusBar.hidden = false;
    statusBar.textContent = "Please enter both a query and at least one seed URL.";
    return;
  }

  runBtn.disabled = true;
  statusBar.hidden = false;
  statusBar.textContent = "Navigating pages, parsing HTML sections, and ranking evidence...";

  try {
    const res = await fetch("/api/retrieve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        seedUrls,
        maxPages: Number(maxPages.value) || 6,
        maxDepth: Number(maxDepth.value) || 2,
        sameDomainOnly: sameDomain.checked,
      }),
    });
    const contentType = res.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
      throw new Error("Static GitHub Pages host cannot run Python live crawling — run `python backend/server.py` locally for arbitrary live URLs.");
    }
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }

    statusBar.hidden = true;
    confidencePill.textContent = `Confidence: ${Math.round((data.overall_confidence || 0) * 100)}%`;
    metricsPill.textContent = `${data.pages_visited} pages · ${data.sections_indexed} sections`;
    summaryBox.textContent = data.answer_summary || "No answer extracted.";

    evidenceList.innerHTML = (data.evidences || []).map((ev, i) => `
      <div class="item-box">
        <div class="item-meta">
          <span>#${i + 1} · ${escapeHtml(ev.heading)} (depth ${ev.depth})</span>
          <span class="score-tag">Score ${ev.score} · ${Math.round(ev.confidence * 100)}%</span>
        </div>
        <div>${escapeHtml(ev.snippet)}</div>
        <div class="item-meta" style="margin-top:8px;margin-bottom:0">
          <a href="${escapeHtml(ev.url)}" target="_blank" rel="noopener">${escapeHtml(ev.url)}</a>
          <span>Terms: ${escapeHtml((ev.matched_terms || []).join(", "))}</span>
        </div>
      </div>
    `).join("") || `<div class="item-box">No matching evidence passages found.</div>`;

    traceList.innerHTML = (data.navigation_trace || []).map((hop) => `
      <div class="item-box">
        <div class="item-meta">
          <span>Step ${hop.step} · Depth ${hop.depth} · ${escapeHtml(hop.status)}</span>
          <span>${hop.elapsed_ms} ms</span>
        </div>
        <div><strong>${escapeHtml(hop.title || "(no title)")}</strong></div>
        <div class="item-meta" style="margin-top:4px">
          <a href="${escapeHtml(hop.url)}" target="_blank" rel="noopener">${escapeHtml(hop.url)}</a>
          <span>${hop.links_discovered} candidate links</span>
        </div>
        ${hop.top_next_link ? `<div class="subtitle">Next best hop (${hop.top_link_score}): ${escapeHtml(hop.top_next_link)}</div>` : ""}
      </div>
    `).join("");
  } catch (err) {
    statusBar.hidden = false;
    statusBar.textContent = `Retrieval failed: ${err.message}`;
  } finally {
    runBtn.disabled = false;
  }
});
