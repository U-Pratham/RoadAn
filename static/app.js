// RoadSight AI - Frontend Controller

let map = null;
let markersLayer = null;
let currentMode = "upload";
let selectedFile = null;
let currentStream = null;

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  checkHealth();
  loadSamples();
  setupEventListeners();
  loadHistory();
});

// 1. Initialize Leaflet Map
function initMap() {
  map = L.map("map").setView([12.9716, 77.5946], 12);

  // CartoDB Dark Matter tiles for modern high-tech map aesthetic
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: "abcd",
    maxZoom: 19
  }).addTo(map);

  markersLayer = L.layerGroup().addTo(map);
}

// 2. Check Backend Health & GPU Status
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const gpuText = document.getElementById("gpuText");
    if (data.gpu_accelerated) {
      gpuText.textContent = `${data.hardware_target} • Active`;
    } else {
      gpuText.textContent = `CPU Execution Active`;
    }
  } catch (err) {
    document.getElementById("gpuText").textContent = "Backend Offline";
  }
}

// 3. Load Samples
async function loadSamples() {
  try {
    const res = await fetch("/api/samples");
    const data = await res.json();
    const select = document.getElementById("sampleSelect");
    select.innerHTML = '<option value="">-- Choose a test road sample --</option>';
    data.samples.forEach(s => {
      const opt = document.createElement("option");
      opt.value = s;
      opt.textContent = s;
      select.appendChild(opt);
    });
  } catch (err) {
    console.error("Failed to load samples", err);
  }
}

// 4. Setup Event Listeners
function setupEventListeners() {
  const btnUpload = document.getElementById("btnUploadMode");
  const btnSample = document.getElementById("btnSampleMode");
  const btnWebcam = document.getElementById("btnWebcamMode");
  const sampleGroup = document.getElementById("sampleGroup");
  const dropzone = document.getElementById("dropzone");
  const cameraPanel = document.getElementById("cameraPanel");
  const fileInput = document.getElementById("fileInput");
  const confSlider = document.getElementById("confSlider");
  const confValue = document.getElementById("confValue");
  const btnAnalyze = document.getElementById("btnAnalyze");
  const btnRefresh = document.getElementById("btnRefreshHistory");

  // Mode Switching
  btnUpload.addEventListener("click", () => {
    setMode("upload");
    btnUpload.classList.add("active");
    btnSample.classList.remove("active");
    btnWebcam.classList.remove("active");
    dropzone.classList.remove("hidden");
    sampleGroup.classList.add("hidden");
    cameraPanel.classList.add("hidden");
    stopCamera();
  });

  btnSample.addEventListener("click", () => {
    setMode("sample");
    btnSample.classList.add("active");
    btnUpload.classList.remove("active");
    btnWebcam.classList.remove("active");
    dropzone.classList.add("hidden");
    sampleGroup.classList.remove("hidden");
    cameraPanel.classList.add("hidden");
    stopCamera();
  });

  btnWebcam.addEventListener("click", () => {
    setMode("webcam");
    btnWebcam.classList.add("active");
    btnUpload.classList.remove("active");
    btnSample.classList.remove("active");
    dropzone.classList.add("hidden");
    sampleGroup.classList.add("hidden");
    cameraPanel.classList.remove("hidden");
    startCamera();
  });

  // Slider change
  confSlider.addEventListener("input", (e) => {
    confValue.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // File Dropzone
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      selectedFile = e.dataTransfer.files[0];
      previewUploadedFile(selectedFile);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      selectedFile = e.target.files[0];
      previewUploadedFile(selectedFile);
    }
  });

  // Sample Selection
  document.getElementById("sampleSelect").addEventListener("change", (e) => {
    if (e.target.value) {
      analyzeSample(e.target.value);
    }
  });

  // Analyze Button
  btnAnalyze.addEventListener("click", () => {
    if (currentMode === "upload") {
      if (!selectedFile) {
        alert("Please drop or select a road photograph first.");
        return;
      }
      uploadAndAnalyze(selectedFile);
    } else if (currentMode === "sample") {
      const sample = document.getElementById("sampleSelect").value;
      if (!sample) {
        alert("Please select a sample road from the dropdown.");
        return;
      }
      analyzeSample(sample);
    }
  });

  // Webcam Capture Button
  document.getElementById("btnCapture")?.addEventListener("click", captureFromWebcam);

  // Refresh Map History
  btnRefresh.addEventListener("click", loadHistory);
}

function setMode(mode) {
  currentMode = mode;
}

function previewUploadedFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = document.getElementById("annotatedImage");
    img.src = e.target.result;
    img.classList.remove("hidden");
    document.getElementById("viewerPlaceholder").classList.add("hidden");
  };
  reader.readAsDataURL(file);
}

// 5. Trigger Analysis for Upload
async function uploadAndAnalyze(file) {
  const conf = document.getElementById("confSlider").value;
  const formData = new FormData();
  formData.append("file", file);
  formData.append("conf_thresh", conf);

  setLoadingState(true);
  try {
    const res = await fetch("/api/detect", {
      method: "POST",
      body: formData
    });
    if (!res.ok) throw new Error("Detection failed");
    const data = await res.json();
    renderResults(data);
    loadHistory();
  } catch (err) {
    alert("Inference error: " + err.message);
  } finally {
    setLoadingState(false);
  }
}

// 6. Trigger Analysis for Sample
async function analyzeSample(sampleName) {
  const conf = document.getElementById("confSlider").value;
  const formData = new FormData();
  formData.append("sample_name", sampleName);
  formData.append("conf_thresh", conf);

  setLoadingState(true);
  try {
    const res = await fetch("/api/detect", {
      method: "POST",
      body: formData
    });
    if (!res.ok) throw new Error("Detection failed");
    const data = await res.json();
    renderResults(data);
    loadHistory();
  } catch (err) {
    alert("Inference error: " + err.message);
  } finally {
    setLoadingState(false);
  }
}

// 7. Render AI & Risk Output
function renderResults(data) {
  // Annotated Image
  const img = document.getElementById("annotatedImage");
  img.src = data.annotated_image;
  img.classList.remove("hidden");
  document.getElementById("viewerPlaceholder").classList.add("hidden");

  // Latency & Counts Badges
  document.getElementById("badgeLatency").textContent = `${data.inference_ms} ms (${data.provider.replace('ExecutionProvider', '')})`;
  document.getElementById("badgeCount").textContent = `${data.detections_count} Hazards Found`;

  // Risk Overview
  const risk = data.risk;
  document.getElementById("riskScore").textContent = risk.score;
  const circle = document.getElementById("riskCircle");
  circle.style.borderColor = risk.color;
  circle.style.boxShadow = `0 0 18px ${risk.color}55`;

  const badge = document.getElementById("riskLevelBadge");
  badge.textContent = risk.level + " RISK";
  badge.style.backgroundColor = risk.color;
  badge.style.color = (risk.level === "MODERATE" || risk.level === "LOW") ? "#0b0f19" : "#ffffff";

  document.getElementById("riskAction").textContent = risk.action;
  document.getElementById("pciScore").textContent = `${risk.pci} / 100`;

  // Defect Counters
  document.getElementById("cntPothole").textContent = risk.defect_counts.pothole || 0;
  document.getElementById("cntAlligator").textContent = risk.defect_counts.alligator_crack || 0;
  document.getElementById("cntLongitudinal").textContent = risk.defect_counts.longitudinal_crack || 0;
  document.getElementById("cntTransverse").textContent = risk.defect_counts.transverse_crack || 0;

  // Geo info
  document.getElementById("geoText").textContent = `GPS: Lat ${data.latitude.toFixed(5)}, Lon ${data.longitude.toFixed(5)}`;

  // Center map on this inspection
  map.setView([data.latitude, data.longitude], 14);
}

function setLoadingState(loading) {
  const btn = document.getElementById("btnAnalyze");
  if (loading) {
    btn.disabled = true;
    btn.innerHTML = `<span class="status-dot"></span> Analyzing...`;
    document.getElementById("badgeLatency").textContent = "Inferencing...";
  } else {
    btn.disabled = false;
    btn.innerHTML = `<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Analyze Road`;
  }
}

// 8. Load GIS Map History
async function loadHistory() {
  try {
    const res = await fetch("/api/history");
    const data = await res.json();
    markersLayer.clearLayers();

    if (!data.history || data.history.length === 0) return;

    data.history.forEach(item => {
      if (!item.latitude || !item.longitude) return;

      const color = item.risk_score >= 70 ? "#ef4444" :
                    item.risk_score >= 45 ? "#f97316" :
                    item.risk_score >= 20 ? "#eab308" : "#10b981";

      const circleMarker = L.circleMarker([item.latitude, item.longitude], {
        radius: 8 + (item.risk_score / 12),
        fillColor: color,
        color: "#ffffff",
        weight: 1.5,
        opacity: 0.9,
        fillOpacity: 0.75
      });

      const popupContent = `
        <div style="font-family: 'Inter', sans-serif; color: #1e293b; min-width: 170px;">
          <h4 style="margin: 0 0 4px 0; font-weight: 700; color: ${color};">${item.risk_level} RISK (Score: ${item.risk_score})</h4>
          <p style="margin: 0 0 4px 0; font-size: 12px; color: #64748b;">${item.timestamp}</p>
          <p style="margin: 0; font-size: 13px;"><b>${item.defect_count}</b> road hazards detected.</p>
        </div>
      `;

      circleMarker.bindPopup(popupContent);
      markersLayer.addLayer(circleMarker);
    });
  } catch (err) {
    console.error("Failed to load map history", err);
  }
}

// 9. Camera Support
async function startCamera() {
  try {
    currentStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    document.getElementById("webcam").srcObject = currentStream;
  } catch (err) {
    alert("Camera access denied or unavailable: " + err.message);
  }
}

function stopCamera() {
  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop());
    currentStream = null;
  }
}

function captureFromWebcam() {
  const video = document.getElementById("webcam");
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob((blob) => {
    const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
    selectedFile = file;
    uploadAndAnalyze(file);
  }, "image/jpeg", 0.9);
}
