const state = {
  surahs: [],
  ayahs: [],
  sessionId: null,
  mediaRecorder: null,
  mediaStream: null,
  websocket: null,
  mimeType: "",
};

const elements = {
  healthPill: document.getElementById("health-pill"),
  surahSelect: document.getElementById("surah-select"),
  ayahSelect: document.getElementById("ayah-select"),
  createSessionBtn: document.getElementById("create-session-btn"),
  startBtn: document.getElementById("start-btn"),
  stopBtn: document.getElementById("stop-btn"),
  sessionId: document.getElementById("session-id"),
  currentAyah: document.getElementById("current-ayah"),
  recorderState: document.getElementById("recorder-state"),
  expectedText: document.getElementById("expected-text"),
  recognizedText: document.getElementById("recognized-text"),
  statusText: document.getElementById("status-text"),
  similarityText: document.getElementById("similarity-text"),
  missingText: document.getElementById("missing-text"),
  incorrectText: document.getElementById("incorrect-text"),
  ayahList: document.getElementById("ayah-list"),
};

function setHealth(status, label) {
  elements.healthPill.textContent = label;
  elements.healthPill.className = `status-pill ${status}`;
}

function setRecorderState(label) {
  elements.recorderState.textContent = label;
}

function populateSelect(select, items, formatter) {
  select.innerHTML = "";
  for (const item of items) {
    const option = document.createElement("option");
    option.value = String(item.value);
    option.textContent = formatter(item);
    select.appendChild(option);
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json();
}

function renderAyahs(activeAyah = null) {
  elements.ayahList.innerHTML = "";
  for (const ayah of state.ayahs) {
    const row = document.createElement("article");
    row.className = `ayah-row${activeAyah === ayah.ayah ? " active" : ""}`;
    row.innerHTML = `
      <div class="ayah-index">Ayah ${ayah.ayah}</div>
      <div class="ayah-body">${ayah.text}</div>
    `;
    elements.ayahList.appendChild(row);
  }
}

function renderSession(payload) {
  state.sessionId = payload.session_id;
  elements.sessionId.textContent = payload.session_id;
  elements.currentAyah.textContent = `${payload.surah}:${payload.current_ayah}`;
  elements.expectedText.textContent = payload.expected_text || "-";
  elements.recognizedText.textContent = payload.recognized_text || "-";
  elements.statusText.textContent = payload.status || "-";
  elements.similarityText.textContent = `${Number(payload.similarity || 0).toFixed(1)}%`;
  elements.missingText.textContent = payload.missing_words?.join(" ") || "-";
  elements.incorrectText.textContent = payload.incorrect_pairs?.join(" | ") || "-";
  elements.startBtn.disabled = false;
  renderAyahs(payload.current_ayah);
}

async function loadHealth() {
  try {
    await fetchJson("/health");
    setHealth("ok", "API Ready");
  } catch (error) {
    setHealth("error", "API Offline");
  }
}

async function loadSurahs() {
  state.surahs = await fetchJson("/surahs");
  populateSelect(
    elements.surahSelect,
    state.surahs.map((surah) => ({ value: surah.surah, ...surah })),
    (surah) => `${surah.surah}. ${surah.name} (${surah.ayah_count} ayahs)`
  );
  await loadAyahs();
}

async function loadAyahs() {
  const surah = Number(elements.surahSelect.value);
  state.ayahs = await fetchJson(`/surahs/${surah}/ayahs`);
  populateSelect(
    elements.ayahSelect,
    state.ayahs.map((ayah) => ({ value: ayah.ayah, ...ayah })),
    (ayah) => `Ayah ${ayah.ayah}`
  );
  renderAyahs();
}

async function createSession() {
  const surah = Number(elements.surahSelect.value);
  const startAyah = Number(elements.ayahSelect.value);
  const payload = await fetchJson("/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ surah, start_ayah: startAyah }),
  });
  renderSession(payload);
}

function websocketUrl(sessionId) {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}/ws/sessions/${sessionId}`;
}

function preferredMimeType() {
  const options = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
  ];
  return options.find((mime) => window.MediaRecorder && MediaRecorder.isTypeSupported(mime)) || "";
}

async function startRecitation() {
  if (!state.sessionId) {
    throw new Error("Create a session first.");
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("This browser does not support microphone capture.");
  }

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ws = new WebSocket(websocketUrl(state.sessionId));
  ws.binaryType = "arraybuffer";

  ws.addEventListener("open", () => {
    setRecorderState("Connected");
  });

  ws.addEventListener("message", (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === "session_ready" || payload.type === "update") {
      renderSession(payload);
      if (payload.session_complete) {
        stopRecitation();
        setRecorderState("Completed");
      }
    } else if (payload.type === "error") {
      elements.statusText.textContent = payload.detail;
    }
  });

  ws.addEventListener("close", () => {
    if (state.websocket === ws) {
      state.websocket = null;
    }
  });

  const mimeType = preferredMimeType();
  const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
  recorder.addEventListener("dataavailable", async (event) => {
    if (!event.data || event.data.size === 0 || !state.websocket || state.websocket.readyState !== WebSocket.OPEN) {
      return;
    }
    const buffer = await event.data.arrayBuffer();
    state.websocket.send(buffer);
  });

  recorder.addEventListener("start", () => {
    setRecorderState("Listening");
  });

  recorder.addEventListener("stop", () => {
    setRecorderState("Stopped");
  });

  state.mediaStream = stream;
  state.websocket = ws;
  state.mediaRecorder = recorder;
  state.mimeType = mimeType;

  recorder.start(1500);
  elements.startBtn.disabled = true;
  elements.stopBtn.disabled = false;
  elements.createSessionBtn.disabled = true;
}

function stopTracks(stream) {
  if (!stream) {
    return;
  }
  for (const track of stream.getTracks()) {
    track.stop();
  }
}

function stopRecitation() {
  if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
    state.mediaRecorder.stop();
  }
  if (state.websocket && state.websocket.readyState < WebSocket.CLOSING) {
    state.websocket.close();
  }
  stopTracks(state.mediaStream);
  state.mediaRecorder = null;
  state.mediaStream = null;
  state.websocket = null;
  elements.startBtn.disabled = !state.sessionId;
  elements.stopBtn.disabled = true;
  elements.createSessionBtn.disabled = false;
}

function bindEvents() {
  elements.surahSelect.addEventListener("change", async () => {
    await loadAyahs();
  });

  elements.createSessionBtn.addEventListener("click", async () => {
    try {
      await createSession();
      setRecorderState("Session ready");
    } catch (error) {
      elements.statusText.textContent = error.message;
    }
  });

  elements.startBtn.addEventListener("click", async () => {
    try {
      await startRecitation();
    } catch (error) {
      elements.statusText.textContent = error.message;
      stopRecitation();
    }
  });

  elements.stopBtn.addEventListener("click", () => {
    stopRecitation();
  });

  window.addEventListener("beforeunload", () => {
    stopRecitation();
  });
}

async function init() {
  bindEvents();
  await loadHealth();
  await loadSurahs();
  setRecorderState("Idle");
}

init().catch((error) => {
  elements.statusText.textContent = error.message;
  setHealth("error", "Startup Failed");
});
