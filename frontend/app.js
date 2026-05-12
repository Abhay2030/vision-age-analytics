let token = "";
let liveIntervalId = null;
let mediaStream = null;

const authState = document.getElementById("authState");
const output = document.getElementById("analysisOutput");
const sessionsTable = document.getElementById("sessionsTable");
const cameraView = document.getElementById("cameraView");
const captureCanvas = document.getElementById("captureCanvas");
const bucketBars = document.getElementById("bucketBars");

function setAuthState(text) {
  authState.textContent = text;
}

async function authFetch(url, options = {}) {
  if (!token) throw new Error("Please login first.");
  const headers = { ...(options.headers || {}), Authorization: `Bearer ${token}` };
  return fetch(url, { ...options, headers });
}

function renderBucketBars(distribution = {}) {
  const entries = Object.entries(distribution);
  if (!entries.length) {
    bucketBars.innerHTML = "<p class='hint'>No bucket data yet.</p>";
    return;
  }
  const maxVal = Math.max(...entries.map(([, v]) => Number(v) || 0), 1);
  bucketBars.innerHTML = entries
    .map(([k, v]) => {
      const pct = Math.round(((Number(v) || 0) / maxVal) * 100);
      return `
      <div class="bar-row">
        <span>${k}</span>
        <div class="bar"><span style="width:${pct}%"></span></div>
        <span>${v}</span>
      </div>`;
    })
    .join("");
}

function updateOverview(overview) {
  document.getElementById("kpiSessions").textContent = overview.sessions ?? "-";
  document.getElementById("kpiAge").textContent = overview.avg_estimated_age ?? "-";
  document.getElementById("kpiFaces").textContent = overview.avg_faces_per_frame ?? "-";
  document.getElementById("kpiBucket").textContent = overview.top_bucket ?? "-";
}

async function login() {
  try {
    const username = document.getElementById("username").value.trim();
    const password = document.getElementById("password").value.trim();
    if (!username || !password) throw new Error("Enter both username and password.");

    const form = new URLSearchParams();
    form.append("username", username);
    form.append("password", password);
    const res = await fetch("https://vision-age-analytics.onrender.com/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: form.toString(),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Login failed");
    token = data.access_token;
    setAuthState(`Logged in as ${username} (${data.role})`);
    output.textContent = "Login successful.";
    await refreshOverview();
  } catch (err) {
    setAuthState("Login failed");
    output.textContent = String(err);
  }
}

async function analyzeBlob(blob, sourceLabel = "web_input") {
  const form = new FormData();
  form.append("file", blob, `${sourceLabel}.jpg`);
  const res = await authFetch("https://vision-age-analytics.onrender.com/v1/analyze/image", { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) throw new Error(JSON.stringify(data));
  output.textContent = JSON.stringify(data, null, 2);
  renderBucketBars(data.summary?.bucket_distribution || {});
  return data;
}

async function analyzeImage() {
  try {
    const fileInput = document.getElementById("imageFile");
    if (!fileInput.files.length) throw new Error("Please select an image.");
    await analyzeBlob(fileInput.files[0], "upload_image");
    await refreshOverview();
  } catch (err) {
    output.textContent = String(err);
  }
}

async function startCamera() {
  if (mediaStream) return;
  mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  cameraView.srcObject = mediaStream;
}

function stopCamera() {
  if (!mediaStream) return;
  mediaStream.getTracks().forEach((t) => t.stop());
  mediaStream = null;
  cameraView.srcObject = null;
  stopLiveAnalysis();
}

async function analyzeCameraFrame() {
  if (!mediaStream) throw new Error("Start camera first.");
  const width = cameraView.videoWidth || 640;
  const height = cameraView.videoHeight || 360;
  captureCanvas.width = width;
  captureCanvas.height = height;
  captureCanvas.getContext("2d").drawImage(cameraView, 0, 0, width, height);
  const blob = await new Promise((resolve) => captureCanvas.toBlob(resolve, "image/jpeg", 0.92));
  if (!blob) throw new Error("Could not capture frame.");
  return analyzeBlob(blob, "camera_frame");
}

function startLiveAnalysis() {
  const intervalMs = Number(document.getElementById("cameraInterval").value || 1200);
  if (liveIntervalId) clearInterval(liveIntervalId);
  liveIntervalId = setInterval(async () => {
    try {
      await analyzeCameraFrame();
    } catch (err) {
      output.textContent = `Live analyze error: ${err}`;
      stopLiveAnalysis();
    }
  }, Math.max(500, intervalMs));
  document.getElementById("toggleLiveBtn").textContent = "Stop Live Analysis";
}

function stopLiveAnalysis() {
  if (liveIntervalId) clearInterval(liveIntervalId);
  liveIntervalId = null;
  document.getElementById("toggleLiveBtn").textContent = "Start Live Analysis";
}

function toggleLiveAnalysis() {
  if (liveIntervalId) stopLiveAnalysis();
  else startLiveAnalysis();
}

function renderSessions(items) {
  if (!items.length) {
    sessionsTable.innerHTML = "<p>No sessions found.</p>";
    return;
  }
  const rows = items
    .map(
      (s) => `
      <tr>
        <td>${s.id}</td>
        <td>${s.created_at}</td>
        <td>${s.source_type}</td>
        <td>${s.summary.frames_processed ?? 0}</td>
        <td>${s.summary.estimated_avg_age ?? 0}</td>
        <td>${s.summary.unique_people_tracks ?? 0}</td>
      </tr>`
    )
    .join("");
  sessionsTable.innerHTML = `
  <table>
    <thead><tr><th>ID</th><th>Created</th><th>Source</th><th>Frames</th><th>Avg Age</th><th>Tracks</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

async function loadSessions() {
  try {
    const res = await authFetch("https://vision-age-analytics.onrender.com/v1/sessions?limit=20");
    const data = await res.json();
    if (!res.ok) throw new Error(JSON.stringify(data));
    renderSessions(data.items || []);
  } catch (err) {
    sessionsTable.innerHTML = `<pre>${String(err)}</pre>`;
  }
}

async function loadSessionDetail() {
  try {
    const id = Number(document.getElementById("sessionIdInput").value || 0);
    if (!id) throw new Error("Enter a valid session ID.");
    const res = await authFetch(`/v1/sessions/${id}`);
    const data = await res.json();
    if (!res.ok) throw new Error(JSON.stringify(data));
    output.textContent = JSON.stringify(data, null, 2);
    renderBucketBars(data.item?.summary?.bucket_distribution || {});
  } catch (err) {
    output.textContent = String(err);
  }
}

async function refreshOverview() {
  try {
    const res = await authFetch("https://vision-age-analytics.onrender.com/v1/overview");
    const data = await res.json();
    if (!res.ok) throw new Error(JSON.stringify(data));
    updateOverview(data.overview || {});
  } catch (err) {
    output.textContent = `Overview error: ${err}`;
  }
}

document.getElementById("loginBtn").addEventListener("click", login);
document.getElementById("analyzeBtn").addEventListener("click", analyzeImage);
document.getElementById("loadSessionsBtn").addEventListener("click", loadSessions);
document.getElementById("loadSessionDetailBtn").addEventListener("click", loadSessionDetail);
document.getElementById("refreshOverviewBtn").addEventListener("click", refreshOverview);
document.getElementById("startCameraBtn").addEventListener("click", async () => {
  try { await startCamera(); } catch (err) { output.textContent = String(err); }
});
document.getElementById("stopCameraBtn").addEventListener("click", stopCamera);
document.getElementById("analyzeFrameBtn").addEventListener("click", async () => {
  try { await analyzeCameraFrame(); await refreshOverview(); } catch (err) { output.textContent = String(err); }
});
document.getElementById("toggleLiveBtn").addEventListener("click", toggleLiveAnalysis);
document.getElementById("password").addEventListener("keydown", (e) => {
  if (e.key === "Enter") login();
});
