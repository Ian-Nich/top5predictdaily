const API_BASE = "https://predictionmodellally.onrender.com";

async function loadTopPicks(endpoint = "/api/top-picks", method = "GET") {
  setLoading();

  let response;
  try {
    response = await fetch(`${API_BASE}${endpoint}`, {
      method,
      headers: { "Content-Type": "application/json" }
    });
  } catch (error) {
    showFailure("Could not reach the backend.");
    return;
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status}).`;
    try {
      const errBody = await response.json();
      if (errBody && errBody.detail) detail = errBody.detail;
    } catch (_) {
      // response wasn't JSON - keep the generic message
    }
    showFailure(detail);
    return;
  }

  let data;
  try {
    data = await response.json();
  } catch (error) {
    showFailure("Backend returned an unreadable response.");
    return;
  }

  renderTopPicks(data);
}

function setLoading() {
  hideBanner();
  document.getElementById("scanMeta").textContent = "";
  document.getElementById("picksTableBody").innerHTML =
    `<tr><td colspan="7">Loading...</td></tr>`;
}

function hideBanner() {
  const banner = document.getElementById("statusBanner");
  banner.style.display = "none";
  banner.textContent = "";
}

function showFailure(message) {
  // No fallback: on any failure, show ONLY the fail message - no stale
  // picks, no partial table, nothing left over from a previous load.
  const banner = document.getElementById("statusBanner");
  banner.textContent = message;
  banner.style.display = "block";

  document.getElementById("scanMeta").textContent = "";
  document.getElementById("picksTableBody").innerHTML = "";
}

function renderTopPicks(data) {
  hideBanner();

  const generatedAt = data.generated_at ? new Date(data.generated_at).toLocaleString() : "unknown time";
  document.getElementById("scanMeta").textContent =
    `${data.candidates_scanned} candidates scanned - ${data.market_session} - ${generatedAt}`;

  const tableBody = document.getElementById("picksTableBody");
  tableBody.innerHTML = "";

  if (!data.top_picks || data.top_picks.length === 0) {
    tableBody.innerHTML = `<tr><td colspan="7">No candidates cleared the 5% / sub-$20 bar right now.</td></tr>`;
    return;
  }

  data.top_picks.forEach((pick, i) => {
    const row = document.createElement("tr");
    const dirClass = pick.predicted_direction === "UP" ? "positive" : "negative";
    row.innerHTML = `
      <td><span class="rank-badge">${i + 1}</span></td>
      <td>${pick.ticker}</td>
      <td>$${pick.price}</td>
      <td class="${dirClass}">${pick.predicted_direction}</td>
      <td>${pick.confidence_pct}%</td>
      <td>${pick.expected_move_pct}%</td>
      <td>${pick.explanation}</td>
    `;
    tableBody.appendChild(row);
  });
}

document.getElementById("refreshBtn").addEventListener("click", async () => {
  await loadTopPicks("/api/refresh", "POST");
});

loadTopPicks();
