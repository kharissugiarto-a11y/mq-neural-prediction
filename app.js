const SENSOR_KEYS = ["mq6", "mq2", "mq135", "mq3", "mq131"];
const MAX_ROWS = 100000;

const elements = {
  connectionStatus: document.querySelector("#connection-status"),
  connect: document.querySelector("#connect-button"),
  disconnect: document.querySelector("#disconnect-button"),
  calibrate: document.querySelector("#calibrate-button"),
  browserMessage: document.querySelector("#browser-message"),
  runtimeMessage: document.querySelector("#runtime-message"),
  calibrationBadge: document.querySelector("#calibration-badge"),
  gasLabel: document.querySelector("#gas-label"),
  referencePpm: document.querySelector("#reference-ppm"),
  temperature: document.querySelector("#temperature"),
  humidity: document.querySelector("#humidity"),
  notes: document.querySelector("#notes"),
  recordTitle: document.querySelector("#record-title"),
  recordIndicator: document.querySelector("#record-indicator"),
  sampleCount: document.querySelector("#sample-count"),
  sessionName: document.querySelector("#session-name"),
  lastUpdate: document.querySelector("#last-update"),
  start: document.querySelector("#start-button"),
  stop: document.querySelector("#stop-button"),
  download: document.querySelector("#download-button"),
  clear: document.querySelector("#clear-button"),
  previewBody: document.querySelector("#preview-body"),
  modelBadge: document.querySelector("#model-badge"),
};

const state = {
  port: null,
  reader: null,
  connected: false,
  connecting: false,
  recording: false,
  stopRequested: false,
  latest: null,
  rows: [],
  sessionId: "",
  sampleIndex: 0,
  recordStartedAt: 0,
  recordMeta: null,
  downloaded: false,
};

function showError(message) {
  elements.runtimeMessage.textContent = message;
  elements.runtimeMessage.classList.toggle("hidden", !message);
}

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function optionalNumber(input) {
  if (!input.value.trim()) return null;
  return finiteNumber(input.value);
}

function formatNumber(value, decimals = 4) {
  return Number.isFinite(value) ? Number(value).toFixed(decimals) : "—";
}

function setExperimentFieldsDisabled(disabled) {
  [elements.gasLabel, elements.referencePpm, elements.temperature, elements.humidity, elements.notes]
    .forEach((element) => { element.disabled = disabled; });
}

function updateControls() {
  elements.connect.disabled = state.connecting || state.connected || !("serial" in navigator);
  elements.connect.textContent = state.connecting ? "Membuka port…" : "Hubungkan Arduino";
  elements.connect.classList.toggle("hidden", state.connected);
  elements.disconnect.classList.toggle("hidden", !state.connected);
  elements.calibrate.disabled = !state.connected || state.recording;
  elements.start.disabled = !state.connected || state.recording || !state.latest?.calibrated;
  elements.start.classList.toggle("hidden", state.recording);
  elements.stop.classList.toggle("hidden", !state.recording);
  elements.download.disabled = state.rows.length === 0;
  elements.clear.disabled = state.rows.length === 0 || state.recording;

  elements.connectionStatus.classList.toggle("online", state.connected);
  elements.connectionStatus.classList.toggle("offline", !state.connected);
  elements.connectionStatus.querySelector("span").textContent = state.connected
    ? "Terhubung ke Arduino"
    : "Belum terhubung";

  elements.calibrationBadge.textContent = state.latest?.calibrated
    ? "Kalibrasi tersedia"
    : "Belum terkalibrasi";
  elements.calibrationBadge.classList.toggle("warning", !state.latest?.calibrated);

  elements.recordIndicator.classList.toggle("active", state.recording);
  elements.recordIndicator.innerHTML = state.recording
    ? "<i></i>Merekam data"
    : "<i></i>Tidak merekam";
  elements.recordTitle.textContent = state.recording
    ? `Merekam ${state.recordMeta?.gasLabel ?? "data"}`
    : state.rows.length
      ? "Rekaman siap diunduh"
      : "Siap menunggu";
  elements.sampleCount.textContent = state.rows.length.toLocaleString("id-ID");
  elements.sessionName.textContent = state.sessionId || "—";
  setExperimentFieldsDisabled(state.recording);
}

function updateSensorCards(reading) {
  SENSOR_KEYS.forEach((key) => {
    const card = document.querySelector(`[data-sensor="${key}"]`);
    card.querySelector('[data-field="adc"]').textContent = reading?.adc?.[key] ?? "—";
    card.querySelector('[data-field="ratio"]').textContent = formatNumber(reading?.ratio?.[key]);
  });
  elements.modelBadge.textContent = `MQ131: ${reading?.mq131_model ?? "—"}`;
}

function makeSessionId(label) {
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
  return `${label}_${stamp}`;
}

function validRatioReading(reading) {
  return reading?.calibrated && SENSOR_KEYS.every((key) => {
    const value = reading.ratio?.[key];
    return Number.isFinite(value) && value > 0;
  });
}

function appendRow(reading, now) {
  if (!state.recording || !state.recordMeta || !validRatioReading(reading)) return;
  if (state.rows.length >= MAX_ROWS) {
    stopRecording();
    showError(`Perekaman dihentikan pada batas ${MAX_ROWS.toLocaleString("id-ID")} sampel. Unduh data sebelum membuat sesi baru.`);
    return;
  }

  state.sampleIndex += 1;
  const row = {
    session_id: state.sessionId,
    timestamp_iso: now.toISOString(),
    sample_index: state.sampleIndex,
    recording_elapsed_ms: now.getTime() - state.recordStartedAt,
    arduino_ms: reading.ms,
    gas_label: state.recordMeta.gasLabel,
    reference_ppm: state.recordMeta.referencePpm,
    temperature_c: state.recordMeta.temperature,
    humidity_percent: state.recordMeta.humidity,
    notes: state.recordMeta.notes,
    calibrated: reading.calibrated,
    mq131_model: reading.mq131_model,
  };

  SENSOR_KEYS.forEach((key) => {
    row[`adc_${key}`] = reading.adc[key];
    row[`r0_${key}`] = reading.r0[key];
    row[`ratio_${key}`] = reading.ratio[key];
  });

  state.rows.push(row);
  state.downloaded = false;
  renderPreview();
  updateControls();
}

function handleLine(line) {
  if (!line.trim().startsWith("{")) return;
  try {
    const reading = JSON.parse(line);
    if (!reading.adc || !reading.ratio || !reading.r0) return;
    const now = new Date();
    state.latest = reading;
    elements.lastUpdate.textContent = now.toLocaleTimeString("id-ID");
    updateSensorCards(reading);
    showError("");
    appendRow(reading, now);
    updateControls();
  } catch {
    // Baris status firmware yang bukan JSON sengaja diabaikan.
  }
}

async function readSerial(port) {
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (port.readable && !state.stopRequested) {
      const reader = port.readable.getReader();
      state.reader = reader;
      try {
        while (!state.stopRequested) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split(/\r?\n/);
          buffer = lines.pop() ?? "";
          lines.forEach(handleLine);
        }
      } finally {
        reader.releaseLock();
        state.reader = null;
      }
    }
  } catch (error) {
    if (!state.stopRequested) showError(error instanceof Error ? error.message : "Koneksi serial terputus.");
  } finally {
    if (!state.stopRequested) {
      state.connected = false;
      state.recording = false;
      updateControls();
    }
  }
}

async function connectSerial() {
  showError("");
  if (!window.isSecureContext) {
    showError("Halaman harus dibuka melalui HTTPS atau localhost agar Chrome mengizinkan Web Serial.");
    return;
  }
  if (!("serial" in navigator)) {
    showError("Web Serial tidak tersedia. Gunakan Chrome atau Edge desktop terbaru.");
    return;
  }

  state.connecting = true;
  updateControls();
  try {
    const port = await navigator.serial.requestPort();
    await port.open({ baudRate: 115200, bufferSize: 4096 });
    state.port = port;
    state.stopRequested = false;
    state.connected = true;
    state.latest = null;
    updateSensorCards(null);
    void readSerial(port);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Port serial tidak dapat dibuka.";
    if (!message.toLowerCase().includes("no port selected")) showError(message);
  } finally {
    state.connecting = false;
    updateControls();
  }
}

async function disconnectSerial() {
  state.stopRequested = true;
  state.recording = false;
  try {
    await state.reader?.cancel();
    if (state.port) await state.port.close();
  } catch {
    // Port mungkin sudah tertutup karena kabel dicabut.
  } finally {
    state.reader = null;
    state.port = null;
    state.connected = false;
    updateControls();
  }
}

async function calibrate() {
  if (!state.port?.writable) return;
  const accepted = window.confirm(
    "Pastikan semua sensor sudah stabil dan berada di udara bersih berventilasi. Mulai kalibrasi sekitar 15 detik?",
  );
  if (!accepted) return;
  try {
    const writer = state.port.writable.getWriter();
    await writer.write(new TextEncoder().encode("C\n"));
    writer.releaseLock();
    elements.calibrationBadge.textContent = "Kalibrasi berlangsung…";
    elements.calibrationBadge.classList.add("warning");
  } catch (error) {
    showError(error instanceof Error ? error.message : "Perintah kalibrasi gagal dikirim.");
  }
}

function startRecording() {
  if (!state.connected || !validRatioReading(state.latest)) {
    showError("Hubungkan dan kalibrasi Arduino sebelum mulai merekam.");
    return;
  }

  const gasLabel = elements.gasLabel.value;
  if (!gasLabel) {
    showError("Pilih label gas sebelum mulai merekam.");
    elements.gasLabel.focus();
    return;
  }

  const referencePpm = optionalNumber(elements.referencePpm);
  if (elements.referencePpm.value.trim() && (referencePpm == null || referencePpm < 0)) {
    showError("PPM referensi harus berupa angka nol atau lebih besar.");
    elements.referencePpm.focus();
    return;
  }

  const humidity = optionalNumber(elements.humidity);
  if (humidity != null && (humidity < 0 || humidity > 100)) {
    showError("Kelembapan harus berada pada rentang 0–100% RH.");
    elements.humidity.focus();
    return;
  }

  if (state.rows.length && !window.confirm("Mulai sesi baru akan menghapus data rekaman yang sekarang. Pastikan data penting sudah diunduh. Lanjutkan?")) {
    return;
  }

  state.rows = [];
  state.sampleIndex = 0;
  state.sessionId = makeSessionId(gasLabel);
  state.recordStartedAt = Date.now();
  state.recordMeta = {
    gasLabel,
    referencePpm,
    temperature: optionalNumber(elements.temperature),
    humidity,
    notes: elements.notes.value.trim(),
  };
  state.downloaded = false;
  state.recording = true;
  renderPreview();
  showError("");
  updateControls();
}

function stopRecording() {
  state.recording = false;
  updateControls();
}

function clearRows() {
  if (!state.rows.length) return;
  if (!window.confirm("Hapus seluruh data rekaman pada halaman ini?")) return;
  state.rows = [];
  state.sessionId = "";
  state.sampleIndex = 0;
  state.recordMeta = null;
  state.downloaded = false;
  renderPreview();
  updateControls();
}

function csvCell(value) {
  let text = value == null ? "" : String(value);
  if (/^[=+\-@]/.test(text)) text = `'${text}`;
  return `"${text.replaceAll('"', '""')}"`;
}

function downloadCsv() {
  if (!state.rows.length) return;
  const columns = [
    "session_id", "timestamp_iso", "sample_index", "recording_elapsed_ms", "arduino_ms",
    "gas_label", "reference_ppm", "temperature_c", "humidity_percent", "notes",
    "calibrated", "mq131_model",
    ...SENSOR_KEYS.map((key) => `adc_${key}`),
    ...SENSOR_KEYS.map((key) => `r0_${key}`),
    ...SENSOR_KEYS.map((key) => `ratio_${key}`),
  ];
  const csv = [
    columns.map(csvCell).join(","),
    ...state.rows.map((row) => columns.map((column) => csvCell(row[column])).join(",")),
  ].join("\r\n");

  const blob = new Blob(["\uFEFF", csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `mq_dataset_${state.sessionId || Date.now()}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  state.downloaded = true;
}

function renderPreview() {
  if (!state.rows.length) {
    elements.previewBody.innerHTML = '<tr class="empty-row"><td colspan="9">Belum ada sampel yang direkam.</td></tr>';
    return;
  }

  elements.previewBody.innerHTML = state.rows.slice(-8).reverse().map((row) => `
    <tr>
      <td>${row.sample_index}</td>
      <td>${new Date(row.timestamp_iso).toLocaleTimeString("id-ID")}</td>
      <td>${row.gas_label}</td>
      <td>${row.reference_ppm ?? "—"}</td>
      <td>${formatNumber(row.ratio_mq6, 3)}</td>
      <td>${formatNumber(row.ratio_mq2, 3)}</td>
      <td>${formatNumber(row.ratio_mq135, 3)}</td>
      <td>${formatNumber(row.ratio_mq3, 3)}</td>
      <td>${formatNumber(row.ratio_mq131, 3)}</td>
    </tr>
  `).join("");
}

elements.connect.addEventListener("click", connectSerial);
elements.disconnect.addEventListener("click", disconnectSerial);
elements.calibrate.addEventListener("click", calibrate);
elements.start.addEventListener("click", startRecording);
elements.stop.addEventListener("click", stopRecording);
elements.download.addEventListener("click", downloadCsv);
elements.clear.addEventListener("click", clearRows);

window.addEventListener("beforeunload", (event) => {
  if (state.rows.length && !state.downloaded) {
    event.preventDefault();
    event.returnValue = "";
  }
});

window.addEventListener("pagehide", () => {
  state.stopRequested = true;
  void state.reader?.cancel();
});

if (!("serial" in navigator)) {
  elements.browserMessage.textContent = "Web Serial tidak tersedia. Buka halaman melalui Chrome atau Edge desktop terbaru.";
  elements.browserMessage.classList.remove("hidden");
}

updateControls();
renderPreview();
