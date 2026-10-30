// Simple Presets
const EXAMPLES = {
  bank_phish: {
    url: "http://192.168.1.104/chase/login.php?token=92819",
    message: "URGENT: Your bank account is locked due to suspicious activity. Verify your password immediately to avoid termination."
  },
  delivery_scam: {
    url: "https://usps-tracking-support.xyz/verify?id=839102",
    message: "USPS Notification: Package could not be delivered due to an unpaid postage fee. Click to update shipping details."
  },
  work_meeting: {
    url: "https://zoom.us/j/9482019482",
    message: "Hi team, our sprint standup meeting is scheduled for Monday at 10:00 AM. Please update your tickets before the call."
  },
  safe_site: {
    url: "https://coursera.org/learn/machine-learning",
    message: "Your class registration is confirmed. Lecture videos and weekly practice exercises are now available."
  }
};

// Switch Tabs
function switchTab(name) {
  const scannerView = document.getElementById("scannerView");
  const benchmarkView = document.getElementById("benchmarkView");
  const tabScanner = document.getElementById("tabScannerBtn");
  const tabBenchmark = document.getElementById("tabBenchmarkBtn");

  if (name === "scanner") {
    scannerView.classList.remove("hidden");
    benchmarkView.classList.add("hidden");
    tabScanner.classList.add("active");
    tabBenchmark.classList.remove("active");
  } else {
    scannerView.classList.add("hidden");
    benchmarkView.classList.remove("hidden");
    tabScanner.classList.remove("active");
    tabBenchmark.classList.add("active");
    loadBenchmarkTables();
  }
}

// Load Example
function loadExample(key) {
  const item = EXAMPLES[key];
  if (!item) return;

  document.getElementById("urlInput").value = item.url;
  document.getElementById("msgInput").value = item.message;
  document.getElementById("errorMsg").innerText = "";

  onUrlChange(item.url);
}

// Clear URL only
function clearUrl() {
  document.getElementById("urlInput").value = "";
  onUrlChange("");
}

// Reset Form
function resetAll() {
  document.getElementById("urlInput").value = "";
  document.getElementById("msgInput").value = "";
  document.getElementById("errorMsg").innerText = "";
  document.getElementById("resultBox").classList.add("hidden");
  onUrlChange("");
}

// Live URL Preview Handler
let debounceTimer = null;
function onUrlChange(val) {
  const trimmed = val.trim();
  const previewBox = document.getElementById("urlPreviewBox");
  const clearBtn = document.getElementById("clearUrlBtn");

  if (!trimmed) {
    previewBox.classList.add("hidden");
    clearBtn.style.display = "none";
    document.getElementById("previewIframe").src = "about:blank";
    return;
  }

  clearBtn.style.display = "block";
  previewBox.classList.remove("hidden");

  const formattedUrl = /^[a-zA-Z]+:\/\//.test(trimmed) ? trimmed : "http://" + trimmed;
  const isHttps = formattedUrl.toLowerCase().startsWith("https://");

  document.getElementById("previewSslIcon").innerText = isHttps ? "🔒" : "⚠️";
  document.getElementById("previewUrlText").innerText = formattedUrl;
  document.getElementById("previewExtLink").href = formattedUrl;

  const iframe = document.getElementById("previewIframe");
  if (iframe.src !== formattedUrl) {
    iframe.src = formattedUrl;
  }

  // Lookup breakdown info
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(async () => {
    try {
      const res = await fetch(`/api/url-info?url=${encodeURIComponent(trimmed)}`);
      if (res.ok) {
        const info = await res.json();
        document.getElementById("metaProtocol").innerText = info.is_https ? "HTTPS (Secure)" : "HTTP (Not Secure)";
        document.getElementById("metaHost").innerText = info.host || "—";
        document.getElementById("metaHostType").innerText = info.is_ip ? "IP Address" : "Domain Name";
        document.getElementById("metaTld").innerText = info.tld ? `.${info.tld}` : "N/A";
      }
    } catch (e) {
      console.warn("URL info lookup failed:", e);
    }
  }, 200);
}

// Run Phishing Check
async function checkRisk() {
  const urlVal = document.getElementById("urlInput").value.trim();
  const msgVal = document.getElementById("msgInput").value.trim();
  const errorEl = document.getElementById("errorMsg");
  const btn = document.getElementById("checkBtn");

  if (!urlVal && !msgVal) {
    errorEl.innerText = "Please enter a website link or message text to check.";
    return;
  }
  errorEl.innerText = "";

  btn.disabled = true;
  btn.innerText = "Checking...";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: urlVal, message: msgVal })
    });

    if (!res.ok) {
      throw new Error("Server error " + res.status);
    }

    const data = await res.json();
    showResults(data);
  } catch (err) {
    errorEl.innerText = "Error: " + err.message;
  } finally {
    btn.disabled = false;
    btn.innerText = "Check for Phishing";
  }
}

// Show Results
function showResults(data) {
  const box = document.getElementById("resultBox");
  box.classList.remove("hidden");

  box.classList.remove("phishing", "suspicious", "safe");

  const vIcon = document.getElementById("verdictIcon");
  const vText = document.getElementById("verdictText");
  const rNum = document.getElementById("riskNumber");

  if (data.verdict === "PHISHING") {
    box.classList.add("phishing");
    vIcon.innerText = "🔴";
    vText.innerText = "Phishing Scam";
  } else if (data.verdict === "SUSPICIOUS") {
    box.classList.add("suspicious");
    vIcon.innerText = "🟠";
    vText.innerText = "Suspicious Link";
  } else {
    box.classList.add("safe");
    vIcon.innerText = "🟢";
    vText.innerText = "Safe Link / Message";
  }

  rNum.innerText = Math.round(data.risk);
  document.getElementById("adviceText").innerText = data.recommendation;
  document.getElementById("latencyTag").innerText = `⚡ ${data.latency_ms || 12} ms`;

  // Bars
  const ur = data.url_risk;
  const tr = data.text_risk;

  document.getElementById("urlRiskLabel").innerText = ur === null ? "Not scanned" : `${ur}%`;
  document.getElementById("textRiskLabel").innerText = tr === null ? "Not scanned" : `${tr}%`;

  const urlFill = document.getElementById("urlBarFill");
  const textFill = document.getElementById("textBarFill");

  urlFill.style.width = (ur || 0) + "%";
  urlFill.style.background = (ur || 0) > 65 ? "var(--red-text)" : (ur || 0) > 35 ? "var(--yellow-text)" : "var(--green-text)";

  textFill.style.width = (tr || 0) + "%";
  textFill.style.background = (tr || 0) > 65 ? "var(--red-text)" : (tr || 0) > 35 ? "var(--yellow-text)" : "var(--green-text)";

  if (data.models_used) {
    document.getElementById("urlModelLabel").innerText = `Model: ${data.models_used.url_model}`;
    document.getElementById("textModelLabel").innerText = `Model: ${data.models_used.text_model}`;
  }

  // Reasons
  const ul = document.getElementById("reasonsList");
  ul.innerHTML = "";
  const reasons = data.reasons && data.reasons.length ? data.reasons : ["No suspicious signals found."];

  reasons.forEach(r => {
    const li = document.createElement("li");
    li.innerText = r;
    ul.appendChild(li);
  });

  box.scrollIntoView({ behavior: "smooth", block: "center" });
}

// Load Benchmark Data
let tablesDone = false;
async function loadBenchmarkTables() {
  if (tablesDone) return;
  try {
    const res = await fetch("/api/benchmark");
    if (!res.ok) return;
    const meta = await res.json();

    const bestUrl = meta.best_url_model;
    const bestText = meta.best_text_model;

    // URL Table
    const urlTbody = document.querySelector("#urlTable tbody");
    urlTbody.innerHTML = "";
    (meta.url_models_benchmark || []).forEach(row => {
      const isBest = row.Model === bestUrl;
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${row.Model}</strong></td>
        <td>${(row.Accuracy * 100).toFixed(1)}%</td>
        <td>${(row.Precision * 100).toFixed(1)}%</td>
        <td>${(row.Recall * 100).toFixed(1)}%</td>
        <td>${(row.F1 * 100).toFixed(1)}%</td>
        <td>${(row.ROC_AUC * 100).toFixed(1)}%</td>
        <td>${isBest ? '<span class="badge-top">★ Selected Best</span>' : '<span class="badge-eval">Evaluated</span>'}</td>
      `;
      urlTbody.appendChild(tr);
    });

    // Text Table
    const textTbody = document.querySelector("#textTable tbody");
    textTbody.innerHTML = "";
    (meta.text_models_benchmark || []).forEach(row => {
      const isBest = row.Model === bestText;
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${row.Model}</strong></td>
        <td>${(row.Accuracy * 100).toFixed(1)}%</td>
        <td>${(row.Precision * 100).toFixed(1)}%</td>
        <td>${(row.Recall * 100).toFixed(1)}%</td>
        <td>${(row.F1 * 100).toFixed(1)}%</td>
        <td>${(row.ROC_AUC * 100).toFixed(1)}%</td>
        <td>${isBest ? '<span class="badge-top">★ Selected Best</span>' : '<span class="badge-eval">Evaluated</span>'}</td>
      `;
      textTbody.appendChild(tr);
    });

    tablesDone = true;
  } catch (e) {
    console.error("Benchmark load error:", e);
  }
}
