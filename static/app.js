/**
 * RoadSight AI — Municipal Road Hazard & Risk Intelligence Controller
 * Fully connects interactive dashboard to FastAPI backend & SQLite database.
 */

// Application State
const state = {
  selectedFile: null,
  currentInspection: null,
  allHistory: [],
  filteredHistory: [],
  statistics: null,
  map: null,
  markersGroup: null,
  activeViewMode: "annotated", // "annotated" | "original"
  modalActiveViewMode: "annotated"
};

// Priority Colors & Mapping
const PRIORITY_CONFIG = {
  Critical: { color: "#ef4444", bgClass: "priority-critical", borderClass: "border-critical", label: "CRITICAL" },
  High: { color: "#f97316", bgClass: "priority-high", borderClass: "border-high", label: "HIGH" },
  Moderate: { color: "#eab308", bgClass: "priority-moderate", borderClass: "border-moderate", label: "MODERATE" },
  Low: { color: "#22c55e", bgClass: "priority-low", borderClass: "border-low", label: "LOW" }
};

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  setupEventListeners();
  checkHealth();
  refreshAllData();
});

/* ==========================================================================
   1. MAP INITIALIZATION & GIS LAYER
   ========================================================================== */
function initMap() {
  const mapElement = document.getElementById("map");
  if (!mapElement) return;

  // Center on India / default view
  state.map = L.map("map", {
    center: [12.9716, 77.5946],
    zoom: 6,
    zoomControl: true
  });

  // Dark OpenStreetMap CartoDB Tiles
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: "abcd",
    maxZoom: 19
  }).addTo(state.map);

  state.markersGroup = L.layerGroup().addTo(state.map);
}

function updateMapMarkers(inspections) {
  if (!state.map || !state.markersGroup) return;

  state.markersGroup.clearLayers();

  const geolocated = inspections.filter(item => item.latitude !== null && item.longitude !== null && !isNaN(item.latitude) && !isNaN(item.longitude));
  const geolocatedCountElem = document.getElementById("mapGeolocatedCount");
  const emptyOverlay = document.getElementById("mapEmptyOverlay");

  if (geolocatedCountElem) {
    geolocatedCountElem.textContent = `${geolocated.length} Geotagged Records`;
  }

  if (geolocated.length === 0) {
    if (emptyOverlay) emptyOverlay.classList.remove("hidden");
    return;
  } else {
    if (emptyOverlay) emptyOverlay.classList.add("hidden");
  }

  const bounds = L.latLngBounds();

  geolocated.forEach(item => {
    const lat = parseFloat(item.latitude);
    const lon = parseFloat(item.longitude);
    const priority = item.priority_category || "Low";
    const cfg = PRIORITY_CONFIG[priority] || PRIORITY_CONFIG.Low;

    // Custom SVG circle marker
    const markerIcon = L.divIcon({
      className: "custom-map-pin",
      html: `<div style="
        width: 18px;
        height: 18px;
        background-color: ${cfg.color};
        border: 2px solid #ffffff;
        border-radius: 50%;
        box-shadow: 0 0 10px ${cfg.color};
      "></div>`,
      iconSize: [18, 18],
      iconAnchor: [9, 9]
    });

    const marker = L.marker([lat, lon], { icon: markerIcon });

    const popupHtml = `
      <div style="font-family: var(--font-main); color: #0b0f19; min-width: 200px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
          <strong style="font-size:14px;">Inspection #${item.id}</strong>
          <span style="font-size:11px; font-weight:700; color:#fff; background-color:${cfg.color}; padding:2px 6px; border-radius:4px;">${priority.toUpperCase()}</span>
        </div>
        <div style="font-size:12px; color:#475569; margin-bottom:8px;">
          <div><strong>File:</strong> ${escapeHtml(item.original_filename)}</div>
          <div><strong>Risk Score:</strong> ${item.overall_risk_score} / 100</div>
          <div><strong>Hazards:</strong> ${item.detection_count || 0} defects</div>
          <div><strong>Road:</strong> ${escapeHtml(item.road_category || "unknown")}</div>
        </div>
        <button onclick="openInspectionModal(${item.id})" style="
          width:100%;
          background:#0284c7;
          color:#fff;
          border:none;
          border-radius:4px;
          padding:6px;
          font-size:12px;
          font-weight:600;
          cursor:pointer;
        ">View Detailed Audit</button>
      </div>
    `;

    marker.bindPopup(popupHtml);
    state.markersGroup.addLayer(marker);
    bounds.extend([lat, lon]);
  });

  if (geolocated.length > 0) {
    state.map.fitBounds(bounds, { padding: [50, 50], maxZoom: 15 });
  }
}

/* ==========================================================================
   2. HEALTH & HARDWARE STATUS
   ========================================================================== */
async function checkHealth() {
  const statusPill = document.getElementById("hardwareStatusPill");
  const statusText = document.getElementById("hardwareStatusText");
  const statusDot = document.getElementById("statusDot");
  const kpiBadge = document.getElementById("kpiHardwareBadge");

  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();

    if (data.status === "healthy") {
      const isGPU = data.gpu_accelerated;
      statusText.textContent = `${data.hardware_target}`;
      if (statusDot) {
        statusDot.className = "status-dot";
      }
      if (kpiBadge) {
        kpiBadge.textContent = isGPU ? "RTX 2050 (DirectML)" : "CPU Execution";
      }
    }
  } catch (err) {
    console.error("Health check error:", err);
    if (statusText) statusText.textContent = "Backend Offline";
    if (statusDot) statusDot.className = "status-dot dot-error";
    showAlert("Cannot connect to RoadSight backend API. Ensure server is running on port 8000.", "error");
  }
}

/* ==========================================================================
   3. STATISTICS & OVERVIEW
   ========================================================================== */
async function loadStatistics() {
  try {
    const res = await fetch("/api/statistics");
    if (!res.ok) throw new Error("Failed to load statistics");
    const stats = await res.json();
    state.statistics = stats;

    // Update KPI Card values
    updateText("kpiTotalAnalyses", stats.total_analyses_count);
    updateText("kpiTotalAnalysesSub", `${stats.total_analyses_count} inspection audits`);
    updateText("kpiTotalHazards", stats.total_detections_count);
    
    const priCounts = stats.analyses_by_priority || {};
    updateText("kpiCriticalCount", priCounts["Critical"] || 0);
    updateText("kpiHighCount", priCounts["High"] || 0);
    
    updateText("kpiAvgRisk", (stats.average_risk_score !== null) ? stats.average_risk_score.toFixed(1) : "0.0");
    updateText("kpiAvgLatency", (stats.average_inference_latency_ms !== null) ? `${stats.average_inference_latency_ms.toFixed(1)} ms` : "0 ms");

    // Defect Class Distribution
    const clsCounts = stats.detections_by_class || {};
    updateText("defectCountPothole", clsCounts["pothole"] || 0);
    updateText("defectCountAlligator", clsCounts["alligator_crack"] || 0);
    updateText("defectCountLongitudinal", clsCounts["longitudinal_crack"] || 0);
    updateText("defectCountTransverse", clsCounts["transverse_crack"] || 0);

  } catch (err) {
    console.error("Statistics error:", err);
  }
}

/* ==========================================================================
   4. INSPECTION HISTORY & UNLOCATED RECORDS
   ========================================================================== */
async function loadHistory() {
  try {
    const res = await fetch("/api/history?limit=100&offset=0");
    if (!res.ok) throw new Error("Failed to load history");
    const data = await res.json();
    state.allHistory = data.items || [];
    
    applyHistoryFilters();
    updateMapMarkers(state.allHistory);
    updateUnlocatedList(state.allHistory);

  } catch (err) {
    console.error("History error:", err);
  }
}

function updateUnlocatedList(inspections) {
  const container = document.getElementById("unlocatedList");
  const badge = document.getElementById("unlocatedCountBadge");
  if (!container) return;

  const unlocated = inspections.filter(item => item.latitude === null || item.longitude === null);

  if (badge) {
    badge.textContent = `${unlocated.length} records without GPS`;
  }

  if (unlocated.length === 0) {
    container.innerHTML = '<div class="empty-list-placeholder">All inspections contain GPS coordinates</div>';
    return;
  }

  container.innerHTML = unlocated.map(item => {
    const priority = item.priority_category || "Low";
    const cfg = PRIORITY_CONFIG[priority] || PRIORITY_CONFIG.Low;

    return `
      <div class="unlocated-item" onclick="openInspectionModal(${item.id})">
        <div class="unlocated-info">
          <span class="unlocated-name">#${item.id} — ${escapeHtml(item.original_filename)}</span>
          <span class="unlocated-meta">${item.upload_timestamp} • ${item.road_category || 'unknown'}</span>
        </div>
        <div style="text-align:right;">
          <span class="priority-badge ${cfg.bgClass}" style="font-size:0.7rem; padding:2px 6px;">${priority}</span>
          <div style="font-size:0.75rem; font-family:var(--font-mono); color:var(--text-muted); margin-top:2px;">
            Risk: ${item.overall_risk_score}
          </div>
        </div>
      </div>
    `;
  }).join("");
}

function applyHistoryFilters() {
  const searchInput = document.getElementById("historySearchInput");
  const priorityFilter = document.getElementById("historyPriorityFilter");
  const sortSelector = document.getElementById("historySortSelector");

  const query = (searchInput ? searchInput.value : "").trim().toLowerCase();
  const priority = priorityFilter ? priorityFilter.value : "";
  const sort = sortSelector ? sortSelector.value : "newest";

  let list = [...state.allHistory];

  // 1. Text Search Filter
  if (query) {
    list = list.filter(item => 
      String(item.id).includes(query) ||
      (item.original_filename && item.original_filename.toLowerCase().includes(query)) ||
      (item.road_category && item.road_category.toLowerCase().includes(query))
    );
  }

  // 2. Priority Filter
  if (priority) {
    list = list.filter(item => item.priority_category === priority);
  }

  // 3. Sorting
  if (sort === "newest") {
    list.sort((a, b) => b.id - a.id);
  } else if (sort === "oldest") {
    list.sort((a, b) => a.id - b.id);
  } else if (sort === "risk_high") {
    list.sort((a, b) => b.overall_risk_score - a.overall_risk_score);
  } else if (sort === "risk_low") {
    list.sort((a, b) => a.overall_risk_score - b.overall_risk_score);
  }

  state.filteredHistory = list;
  renderHistoryTable();
}

function renderHistoryTable() {
  const tbody = document.getElementById("historyTableBody");
  const emptyState = document.getElementById("historyEmptyState");
  if (!tbody) return;

  if (state.filteredHistory.length === 0) {
    tbody.innerHTML = "";
    if (emptyState) emptyState.classList.remove("hidden");
    return;
  }

  if (emptyState) emptyState.classList.add("hidden");

  tbody.innerHTML = state.filteredHistory.map(item => {
    const priority = item.priority_category || "Low";
    const cfg = PRIORITY_CONFIG[priority] || PRIORITY_CONFIG.Low;
    const hasGPS = (item.latitude !== null && item.longitude !== null);
    const coordsText = hasGPS ? `${parseFloat(item.latitude).toFixed(4)}, ${parseFloat(item.longitude).toFixed(4)}` : "No GPS";

    return `
      <tr>
        <td><strong>#${item.id}</strong></td>
        <td>
          <img 
            src="/api/images/${item.id}?type=annotated" 
            alt="thumb" 
            class="history-thumb"
            onclick="openInspectionModal(${item.id})"
            onerror="this.src='/static/style.css'"
          >
        </td>
        <td style="font-weight:600;">${escapeHtml(item.original_filename)}</td>
        <td style="font-family:var(--font-mono); font-size:0.78rem; color:var(--text-muted);">${item.upload_timestamp}</td>
        <td><span class="badge-tag">${escapeHtml(item.road_category || 'unknown')}</span></td>
        <td style="font-family:var(--font-mono); font-size:0.78rem;">${coordsText}</td>
        <td><span class="badge-tag">${item.detection_count || 0} defects</span></td>
        <td><strong style="font-family:var(--font-mono); font-size:0.95rem; color:${cfg.color}">${item.overall_risk_score}</strong></td>
        <td><span class="priority-badge ${cfg.bgClass}">${priority}</span></td>
        <td>
          <button class="btn btn-secondary btn-sm" onclick="openInspectionModal(${item.id})">
            Inspect
          </button>
        </td>
      </tr>
    `;
  }).join("");
}

/* ==========================================================================
   5. DETECTION WORKSPACE & INGESTION
   ========================================================================== */
function setupEventListeners() {
  // Navigation smooth scroll
  document.querySelectorAll(".nav-link").forEach(link => {
    link.addEventListener("click", e => {
      e.preventDefault();
      document.querySelectorAll(".nav-link").forEach(l => l.classList.remove("active"));
      link.classList.add("active");
      const targetId = link.getAttribute("data-target");
      const elem = document.getElementById(targetId);
      if (elem) elem.scrollIntoView({ behavior: "smooth" });
    });
  });

  // Dropzone drag & drop
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const btnRemoveFile = document.getElementById("btnRemoveFile");

  if (dropzone && fileInput) {
    ["dragenter", "dragover"].forEach(name => {
      dropzone.addEventListener(name, e => {
        e.preventDefault();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach(name => {
      dropzone.addEventListener(name, e => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
      });
    });

    dropzone.addEventListener("drop", e => {
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", e => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
      }
    });
  }

  if (btnRemoveFile) {
    btnRemoveFile.addEventListener("click", e => {
      e.stopPropagation();
      clearSelectedFile();
    });
  }

  // Confidence Threshold Slider (Advanced Settings)
  const confSlider = document.getElementById("confThreshold");
  const confBadge = document.getElementById("confBadge");
  const advancedSummaryBadge = document.getElementById("advancedSummaryBadge");
  if (confSlider && confBadge) {
    confSlider.addEventListener("input", e => {
      const val = parseFloat(e.target.value).toFixed(2);
      confBadge.textContent = val;
      if (advancedSummaryBadge) {
        advancedSummaryBadge.textContent = `${val}${val === "0.20" ? " (Calibrated)" : ""}`;
      }
    });
  }

  // Coordinate Preset Chips
  document.querySelectorAll(".chip-btn").forEach(chip => {
    chip.addEventListener("click", () => {
      const lat = chip.getAttribute("data-lat");
      const lon = chip.getAttribute("data-lon");
      const road = chip.getAttribute("data-road");

      document.getElementById("latInput").value = lat;
      document.getElementById("lonInput").value = lon;
      if (road) document.getElementById("roadCategory").value = road;
    });
  });

  const btnClearGPS = document.getElementById("btnClearGPS");
  if (btnClearGPS) {
    btnClearGPS.addEventListener("click", () => {
      document.getElementById("latInput").value = "";
      document.getElementById("lonInput").value = "";
    });
  }

  // Detection Form Submit
  const detectForm = document.getElementById("detectForm");
  if (detectForm) {
    detectForm.addEventListener("submit", handleDetectionSubmit);
  }

  // View Mode Switcher
  const btnAnnotated = document.getElementById("btnShowAnnotated");
  const btnOriginal = document.getElementById("btnShowOriginal");
  if (btnAnnotated && btnOriginal) {
    btnAnnotated.addEventListener("click", () => switchViewMode("annotated"));
    btnOriginal.addEventListener("click", () => switchViewMode("original"));
  }

  // Refresh Stats Button
  const btnRefreshStats = document.getElementById("btnRefreshStats");
  if (btnRefreshStats) {
    btnRefreshStats.addEventListener("click", refreshAllData);
  }

  // Recenter Map Bounds Button
  const btnFitMapBounds = document.getElementById("btnFitMapBounds");
  if (btnFitMapBounds) {
    btnFitMapBounds.addEventListener("click", () => {
      if (state.allHistory.length > 0) updateMapMarkers(state.allHistory);
    });
  }

  // History Search & Filters
  const searchInput = document.getElementById("historySearchInput");
  const priorityFilter = document.getElementById("historyPriorityFilter");
  const sortSelector = document.getElementById("historySortSelector");

  if (searchInput) searchInput.addEventListener("input", applyHistoryFilters);
  if (priorityFilter) priorityFilter.addEventListener("change", applyHistoryFilters);
  if (sortSelector) sortSelector.addEventListener("change", applyHistoryFilters);

  // Modal Controls
  const btnModalClose = document.getElementById("btnModalClose");
  const btnModalCloseFooter = document.getElementById("btnModalCloseFooter");
  const modalBackdrop = document.getElementById("inspectionModal");

  if (btnModalClose) btnModalClose.addEventListener("click", closeModal);
  if (btnModalCloseFooter) btnModalCloseFooter.addEventListener("click", closeModal);
  if (modalBackdrop) {
    modalBackdrop.addEventListener("click", e => {
      if (e.target === modalBackdrop) closeModal();
    });
  }

  const btnModalAnnotated = document.getElementById("btnModalAnnotated");
  const btnModalOriginal = document.getElementById("btnModalOriginal");
  if (btnModalAnnotated && btnModalOriginal) {
    btnModalAnnotated.addEventListener("click", () => switchModalViewMode("annotated"));
    btnModalOriginal.addEventListener("click", () => switchModalViewMode("original"));
  }

  // Alert Close Button
  const alertCloseBtn = document.getElementById("alertCloseBtn");
  if (alertCloseBtn) {
    alertCloseBtn.addEventListener("click", () => {
      document.getElementById("alertBanner").classList.add("hidden");
    });
  }
}

function handleFileSelect(file) {
  if (!file) return;

  const validTypes = ["image/jpeg", "image/png", "image/webp"];
  if (!validTypes.includes(file.type)) {
    showAlert("Unsupported file format. Please upload a JPEG, PNG, or WebP road photograph.", "error");
    return;
  }

  if (file.size > 25 * 1024 * 1024) {
    showAlert("File size exceeds 25 MB upload limit.", "error");
    return;
  }

  state.selectedFile = file;

  // Show client preview
  const dropzonePrompt = document.getElementById("dropzonePrompt");
  const previewBox = document.getElementById("previewBox");
  const clientPreviewImg = document.getElementById("clientPreviewImg");
  const previewFilename = document.getElementById("previewFilename");

  if (clientPreviewImg && previewBox && dropzonePrompt) {
    clientPreviewImg.src = URL.createObjectURL(file);
    previewFilename.textContent = file.name;
    previewBox.classList.remove("hidden");
    dropzonePrompt.classList.add("hidden");
  }
}

function clearSelectedFile() {
  state.selectedFile = null;
  const fileInput = document.getElementById("fileInput");
  if (fileInput) fileInput.value = "";

  const dropzonePrompt = document.getElementById("dropzonePrompt");
  const previewBox = document.getElementById("previewBox");
  const clientPreviewImg = document.getElementById("clientPreviewImg");

  if (previewBox && dropzonePrompt) {
    previewBox.classList.add("hidden");
    dropzonePrompt.classList.remove("hidden");
  }
  if (clientPreviewImg) clientPreviewImg.src = "";
}

/* ==========================================================================
   6. SUBMIT DETECTION TO BACKEND API
   ========================================================================== */
async function handleDetectionSubmit(e) {
  e.preventDefault();

  if (!state.selectedFile) {
    showAlert("Please select or drop a road image photograph first.", "error");
    return;
  }

  const btnSubmit = document.getElementById("btnSubmitDetect");
  const btnSpinner = document.getElementById("btnSubmitSpinner");
  const btnText = document.getElementById("btnSubmitText");
  const viewerLoading = document.getElementById("viewerLoading");
  const viewerPlaceholder = document.getElementById("viewerPlaceholder");
  const riskContainer = document.getElementById("riskAssessmentContainer");

  // Disable button and activate loading states
  if (btnSubmit) btnSubmit.disabled = true;
  if (btnSpinner) btnSpinner.classList.remove("hidden");
  if (btnText) btnText.textContent = "Processing ONNX Inference...";
  if (viewerLoading) viewerLoading.classList.remove("hidden");
  if (viewerPlaceholder) viewerPlaceholder.classList.add("hidden");
  if (riskContainer) riskContainer.classList.add("hidden");

  const formData = new FormData();
  formData.append("file", state.selectedFile);

  const confVal = document.getElementById("confThreshold").value;
  formData.append("confidence_threshold", confVal);

  const roadCat = document.getElementById("roadCategory").value;
  if (roadCat) formData.append("road_category", roadCat);

  const latVal = document.getElementById("latInput").value.trim();
  const lonVal = document.getElementById("lonInput").value.trim();

  if (latVal !== "") {
    const latNum = parseFloat(latVal);
    if (isNaN(latNum) || latNum < -90 || latNum > 90) {
      showAlert("Latitude must be a valid number between -90 and 90.", "error");
      resetLoadingState();
      return;
    }
    formData.append("latitude", latVal);
  }

  if (lonVal !== "") {
    const lonNum = parseFloat(lonVal);
    if (isNaN(lonNum) || lonNum < -180 || lonNum > 180) {
      showAlert("Longitude must be a valid number between -180 and 180.", "error");
      resetLoadingState();
      return;
    }
    formData.append("longitude", lonVal);
  }

  try {
    const res = await fetch("/api/detect", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: "Inference failed" }));
      throw new Error(errData.detail || `Server returned HTTP ${res.status}`);
    }

    const data = await res.json();
    state.currentInspection = data;

    renderDetectionResult(data);
    showAlert(`Analysis complete for ${data.original_filename} (${data.detection_count} hazards detected)`, "success");

    // Refresh background datasets
    refreshAllData();

  } catch (err) {
    console.error("Detection submission error:", err);
    showAlert(`Inspection Error: ${err.message}`, "error");
    if (viewerPlaceholder) viewerPlaceholder.classList.remove("hidden");
  } finally {
    resetLoadingState();
  }
}

function resetLoadingState() {
  const btnSubmit = document.getElementById("btnSubmitDetect");
  const btnSpinner = document.getElementById("btnSubmitSpinner");
  const btnText = document.getElementById("btnSubmitText");
  const viewerLoading = document.getElementById("viewerLoading");

  if (btnSubmit) btnSubmit.disabled = false;
  if (btnSpinner) btnSpinner.classList.add("hidden");
  if (btnText) btnText.textContent = "Execute AI Road Analysis";
  if (viewerLoading) viewerLoading.classList.add("hidden");
}

/* ==========================================================================
   7. RENDER DETECTION RESULT & RISK ASSESSMENT
   ========================================================================== */
function renderDetectionResult(data) {
  const viewToggleGroup = document.getElementById("viewToggleGroup");
  const imgAnnotated = document.getElementById("imgAnnotated");
  const imgOriginal = document.getElementById("imgOriginal");
  const resultSubtitle = document.getElementById("resultSubtitle");
  const riskContainer = document.getElementById("riskAssessmentContainer");

  if (viewToggleGroup) viewToggleGroup.classList.remove("hidden");
  if (riskContainer) riskContainer.classList.remove("hidden");

  // Load Images
  const annotatedUrl = `/api/images/${data.analysis_id}?type=annotated&t=${Date.now()}`;
  const originalUrl = `/api/images/${data.analysis_id}?type=original&t=${Date.now()}`;

  if (imgAnnotated) {
    imgAnnotated.src = annotatedUrl;
    imgAnnotated.classList.remove("hidden");
  }
  if (imgOriginal) {
    imgOriginal.src = originalUrl;
    imgOriginal.classList.add("hidden");
  }
  switchViewMode("annotated");

  if (resultSubtitle) {
    resultSubtitle.textContent = `Analysis #${data.analysis_id} • ${data.original_filename} (${data.image_dimensions})`;
  }

  // Risk Score & Priority Gauge
  const priority = data.priority_category || "Low";
  const cfg = PRIORITY_CONFIG[priority] || PRIORITY_CONFIG.Low;

  const scoreCircle = document.getElementById("riskScoreCircle");
  const scoreVal = document.getElementById("riskScoreValue");
  const priBadge = document.getElementById("priorityBadge");
  const hazardCountTag = document.getElementById("hazardCountTag");
  const latencyTag = document.getElementById("latencyTag");
  const explanationText = document.getElementById("riskExplanationText");

  if (scoreCircle) {
    scoreCircle.className = `risk-score-circle ${cfg.borderClass}`;
  }
  if (scoreVal) scoreVal.textContent = data.overall_risk_score;
  if (priBadge) {
    priBadge.className = `priority-badge ${cfg.bgClass}`;
    if (data.detection_count === 0) {
      priBadge.textContent = "LOW (0 DETECTIONS)";
    } else {
      priBadge.textContent = `${cfg.label} PRIORITY`;
    }
  }
  if (hazardCountTag) {
    hazardCountTag.textContent = `${data.detection_count} Hazard${data.detection_count === 1 ? '' : 's'} Identified`;
  }
  if (latencyTag) {
    latencyTag.textContent = `${data.inference_latency_ms.toFixed(1)} ms inference`;
  }
  if (explanationText) {
    explanationText.textContent = data.summary_explanation;
  }

  // Meta row
  updateText("metaDimensions", `Dimensions: ${data.image_dimensions}`);
  updateText("metaRoadCategory", `Road: ${data.road_category}`);
  const hasGPS = (data.latitude !== null && data.longitude !== null);
  updateText("metaGPS", hasGPS ? `GPS: ${parseFloat(data.latitude).toFixed(4)}, ${parseFloat(data.longitude).toFixed(4)}` : "GPS: None");

  // Factors Breakdown
  // If detections exist, show factors of highest risk hazard; otherwise 0
  let topFactors = { severity: 0, area: 0, confidence: 0, road_importance: 0 };
  if (data.detections && data.detections.length > 0) {
    // Find highest individual score detection
    const topDet = data.detections.reduce((prev, curr) => (curr.individual_risk_score > prev.individual_risk_score) ? curr : prev, data.detections[0]);
    if (topDet.factors) {
      topFactors = topDet.factors;
    }
  }

  updateText("factorSeverityScore", `${topFactors.severity.toFixed(1)} / 40`);
  setBarWidth("barSeverity", (topFactors.severity / 40) * 100);

  updateText("factorAreaScore", `${topFactors.area.toFixed(1)} / 20`);
  setBarWidth("barArea", (topFactors.area / 20) * 100);

  updateText("factorConfScore", `${topFactors.confidence.toFixed(1)} / 15`);
  setBarWidth("barConfidence", (topFactors.confidence / 15) * 100);

  updateText("factorRoadScore", `${topFactors.road_importance.toFixed(1)} / 25`);
  setBarWidth("barRoad", (topFactors.road_importance / 25) * 100);

  // Individual Detections Table
  const tbody = document.getElementById("detectionTableBody");
  if (tbody) {
    if (!data.detections || data.detections.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; color:var(--text-muted); padding:1.5rem;">No damage hazards detected above confidence threshold. Note: Visual scan does not guarantee defect-free road surface.</td></tr>`;
    } else {
      tbody.innerHTML = data.detections.map((d, idx) => {
        const dPri = d.priority_category || "Low";
        const dCfg = PRIORITY_CONFIG[dPri] || PRIORITY_CONFIG.Low;
        const bbox = d.bounding_box || [0, 0, 0, 0];
        const bboxStr = `[${bbox.map(n => Math.round(n)).join(", ")}]`;

        return `
          <tr>
            <td><strong>#${idx + 1}</strong></td>
            <td><strong style="color:var(--text-main);">${escapeHtml(d.damage_class)}</strong></td>
            <td style="font-family:var(--font-mono);">${(d.confidence * 100).toFixed(1)}%</td>
            <td><span class="bbox-tag">${bboxStr}</span></td>
            <td><strong style="font-family:var(--font-mono); color:${dCfg.color}">${d.individual_risk_score}</strong></td>
            <td><span class="priority-badge ${dCfg.bgClass}" style="font-size:0.7rem; padding:2px 8px;">${dPri}</span></td>
          </tr>
        `;
      }).join("");
    }
  }
}

function switchViewMode(mode) {
  state.activeViewMode = mode;
  const btnAnnotated = document.getElementById("btnShowAnnotated");
  const btnOriginal = document.getElementById("btnShowOriginal");
  const imgAnnotated = document.getElementById("imgAnnotated");
  const imgOriginal = document.getElementById("imgOriginal");

  if (mode === "annotated") {
    btnAnnotated.classList.add("active");
    btnOriginal.classList.remove("active");
    if (imgAnnotated) imgAnnotated.classList.remove("hidden");
    if (imgOriginal) imgOriginal.classList.add("hidden");
  } else {
    btnOriginal.classList.add("active");
    btnAnnotated.classList.remove("active");
    if (imgOriginal) imgOriginal.classList.remove("hidden");
    if (imgAnnotated) imgAnnotated.classList.add("hidden");
  }
}

/* ==========================================================================
   8. INSPECTION DETAIL MODAL
   ========================================================================== */
async function openInspectionModal(analysisId) {
  const modal = document.getElementById("inspectionModal");
  if (!modal) return;

  try {
    const res = await fetch(`/api/history/${analysisId}`);
    if (!res.ok) throw new Error("Could not retrieve inspection detail");
    const item = await res.json();

    // Populate Modal Header
    updateText("modalTitle", `Inspection Audit #${item.id} — ${escapeHtml(item.original_filename)}`);
    updateText("modalSubtitle", `Uploaded on ${item.upload_timestamp}`);

    // Modal Badges
    const priority = item.priority_category || "Low";
    const cfg = PRIORITY_CONFIG[priority] || PRIORITY_CONFIG.Low;
    const modalBadges = document.getElementById("modalBadges");
    if (modalBadges) {
      const badgeText = item.detection_count === 0 ? "LOW (0 DEFECTS)" : `${priority} PRIORITY`;
      modalBadges.innerHTML = `
        <span class="priority-badge ${cfg.bgClass}">${badgeText}</span>
        <span class="badge-tag">Risk: ${item.overall_risk_score}/100</span>
        <span class="badge-tag">${item.detection_count || 0} Defects</span>
      `;
    }

    // Modal Images
    const modalImgAnnotated = document.getElementById("modalImgAnnotated");
    const modalImgOriginal = document.getElementById("modalImgOriginal");
    if (modalImgAnnotated) modalImgAnnotated.src = `/api/images/${item.id}?type=annotated`;
    if (modalImgOriginal) modalImgOriginal.src = `/api/images/${item.id}?type=original`;
    switchModalViewMode("annotated");

    // Modal Details
    updateText("modalExplanation", item.summary_explanation || "No diagnosis explanation provided.");
    updateText("modalRoadCat", item.road_category || "unknown");
    const coordsStr = (item.latitude !== null && item.longitude !== null) 
      ? `${parseFloat(item.latitude).toFixed(6)}, ${parseFloat(item.longitude).toFixed(6)}` 
      : "No GPS coordinates recorded";
    updateText("modalCoords", coordsStr);
    updateText("modalLatency", `${item.inference_ms.toFixed(1)} ms`);
    updateText("modalTotalTime", `${item.total_ms.toFixed(1)} ms`);

    // Modal Detections Table
    const tbody = document.getElementById("modalDetectionsBody");
    if (tbody) {
      if (!item.detections || item.detections.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:1.5rem; color:var(--text-muted);">No road damage hazards detected above confidence threshold. Note: Visual scan does not guarantee defect-free road surface.</td></tr>`;
      } else {
        tbody.innerHTML = item.detections.map(d => {
          const dPri = d.priority_category || "Low";
          const dCfg = PRIORITY_CONFIG[dPri] || PRIORITY_CONFIG.Low;
          const bbox = d.bounding_box || [0, 0, 0, 0];
          const bboxStr = `[${bbox.map(n => Math.round(n)).join(", ")}]`;

          return `
            <tr>
              <td><strong>${escapeHtml(d.damage_class)}</strong></td>
              <td style="font-family:var(--font-mono);">${(d.confidence * 100).toFixed(1)}%</td>
              <td><span class="bbox-tag">${bboxStr}</span></td>
              <td><strong style="color:${dCfg.color}; font-family:var(--font-mono);">${d.individual_risk_score}</strong></td>
              <td><span class="priority-badge ${dCfg.bgClass}" style="font-size:0.7rem; padding:2px 6px;">${dPri}</span></td>
              <td style="font-size:0.8rem; color:var(--text-muted);">${escapeHtml(d.explanation || '')}</td>
            </tr>
          `;
        }).join("");
      }
    }

    modal.classList.remove("hidden");

  } catch (err) {
    console.error("Modal detail error:", err);
    showAlert("Failed to retrieve inspection details from server.", "error");
  }
}

function switchModalViewMode(mode) {
  state.modalActiveViewMode = mode;
  const btnAnnotated = document.getElementById("btnModalAnnotated");
  const btnOriginal = document.getElementById("btnModalOriginal");
  const imgAnnotated = document.getElementById("modalImgAnnotated");
  const imgOriginal = document.getElementById("modalImgOriginal");

  if (mode === "annotated") {
    btnAnnotated.classList.add("active");
    btnOriginal.classList.remove("active");
    if (imgAnnotated) imgAnnotated.classList.remove("hidden");
    if (imgOriginal) imgOriginal.classList.add("hidden");
  } else {
    btnOriginal.classList.add("active");
    btnAnnotated.classList.remove("active");
    if (imgOriginal) imgOriginal.classList.remove("hidden");
    if (imgAnnotated) imgAnnotated.classList.add("hidden");
  }
}

function closeModal() {
  const modal = document.getElementById("inspectionModal");
  if (modal) modal.classList.add("hidden");
}

/* ==========================================================================
   9. HELPERS & UTILITIES
   ========================================================================== */
function refreshAllData() {
  loadStatistics();
  loadHistory();
}

function updateText(id, text) {
  const elem = document.getElementById(id);
  if (elem) elem.textContent = text;
}

function setBarWidth(id, pct) {
  const elem = document.getElementById(id);
  if (elem) elem.style.width = `${Math.min(100, Math.max(0, pct))}%`;
}

function showAlert(message, type = "info") {
  const banner = document.getElementById("alertBanner");
  const msgElem = document.getElementById("alertMessage");
  if (!banner || !msgElem) return;

  msgElem.textContent = message;
  banner.classList.remove("hidden");

  if (type === "error") {
    banner.style.backgroundColor = "rgba(239, 68, 68, 0.2)";
    banner.style.color = "#fca5a5";
  } else if (type === "success") {
    banner.style.backgroundColor = "rgba(34, 197, 94, 0.2)";
    banner.style.color = "#86efac";
  } else {
    banner.style.backgroundColor = "rgba(56, 189, 248, 0.2)";
    banner.style.color = "#7dd3fc";
  }

  // Auto-hide after 6 seconds
  setTimeout(() => {
    banner.classList.add("hidden");
  }, 6000);
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
