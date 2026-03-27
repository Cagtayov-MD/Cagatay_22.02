let state = null;
let selectedProfileId = null;
let selectedJobId = null;
let selectedNodeId = null;
let activeTab = "logs";
let drawerOpen = false;

const profileSelectEl = document.getElementById("profile-select");
const jobSelectEl = document.getElementById("job-select");
const statusCurrentNodeEl = document.getElementById("status-current-node");
const statusElapsedEl = document.getElementById("status-elapsed");
const statusBottleneckEl = document.getElementById("status-bottleneck");
const statusQueueEl = document.getElementById("status-queue");
const toolbarProfileChipEl = document.getElementById("toolbar-profile-chip");
const toolbarJobChipEl = document.getElementById("toolbar-job-chip");
const jobStatusPillEl = document.getElementById("job-status-pill");
const jobBottleneckPillEl = document.getElementById("job-bottleneck-pill");
const dagNodesEl = document.getElementById("dag-nodes");
const dagEdgesEl = document.getElementById("dag-edges");
const tooltipEl = document.getElementById("node-tooltip");
const drawerEl = document.getElementById("detail-drawer");
const drawerCloseEl = document.getElementById("drawer-close");
const nodeTitleEl = document.getElementById("node-title");
const nodeStatusPillEl = document.getElementById("node-status-pill");
const nodeScorePillEl = document.getElementById("node-score-pill");
const overviewGridEl = document.getElementById("overview-grid");
const rulesGridEl = document.getElementById("rules-grid");
const metricsGridEl = document.getElementById("metrics-grid");
const decisionLogEl = document.getElementById("decision-log");
const logsPanelEl = document.getElementById("logs-panel");
const artifactsPanelEl = document.getElementById("artifacts-panel");
const tracePanelEl = document.getElementById("trace-panel");
const tabButtons = Array.from(document.querySelectorAll(".dock-tab"));
const tabPanels = Array.from(document.querySelectorAll(".dock-panel"));

async function bootstrap() {
  const response = await fetch("/api/state", { cache: "no-store" });
  state = await response.json();

  selectedProfileId = state.profiles[0]?.id ?? null;
  selectedJobId = jobsForProfile(selectedProfileId)[0]?.id ?? null;
  selectedNodeId = currentJob()?.current_node_id ?? currentJob()?.dag.nodes[0]?.id ?? null;

  bindEvents();
  renderAll();
}

function bindEvents() {
  profileSelectEl.addEventListener("change", (event) => {
    selectedProfileId = event.target.value;
    selectedJobId = jobsForProfile(selectedProfileId)[0]?.id ?? null;
    selectedNodeId = currentJob()?.current_node_id ?? currentJob()?.dag.nodes[0]?.id ?? null;
    drawerOpen = false;
    renderAll();
  });

  jobSelectEl.addEventListener("change", (event) => {
    selectedJobId = event.target.value;
    selectedNodeId = currentJob()?.current_node_id ?? currentJob()?.dag.nodes[0]?.id ?? null;
    drawerOpen = false;
    renderAll();
  });

  drawerCloseEl.addEventListener("click", () => {
    drawerOpen = false;
    renderDrawer();
  });

  tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      activeTab = button.dataset.tab;
      renderTabs();
    });
  });
}

function currentProfile() {
  return state.profiles.find((profile) => profile.id === selectedProfileId) ?? null;
}

function jobsForProfile(profileId) {
  return state.jobs.filter((job) => job.profile_id === profileId);
}

function currentJob() {
  return state.jobs.find((job) => job.id === selectedJobId) ?? null;
}

function currentNode() {
  const job = currentJob();
  return job?.dag.nodes.find((node) => node.id === selectedNodeId) ?? null;
}

function renderAll() {
  renderToolbar();
  renderStatusRibbon();
  renderDag();
  renderDrawer();
  renderBottomDock();
  renderTabs();
}

function renderToolbar() {
  const profile = currentProfile();
  const job = currentJob();

  profileSelectEl.innerHTML = "";
  state.profiles.forEach((profileOption) => {
    const option = document.createElement("option");
    option.value = profileOption.id;
    option.textContent = profileOption.name;
    option.selected = profileOption.id === selectedProfileId;
    profileSelectEl.appendChild(option);
  });

  jobSelectEl.innerHTML = "";
  jobsForProfile(selectedProfileId).forEach((jobOption) => {
    const option = document.createElement("option");
    option.value = jobOption.id;
    option.textContent = jobOption.title;
    option.selected = jobOption.id === selectedJobId;
    jobSelectEl.appendChild(option);
  });

  toolbarProfileChipEl.textContent = profile?.name ?? "Profil";
  toolbarJobChipEl.textContent = job?.title ?? "Is";
  jobStatusPillEl.textContent = `Status: ${job?.status ?? "-"}`;
  jobBottleneckPillEl.textContent = `Bottleneck: ${job?.bottleneck ?? "-"}`;
}

function renderStatusRibbon() {
  const job = currentJob();
  const node = currentNode();
  const queueCount = jobsForProfile(selectedProfileId).length;
  const activeCount = state.jobs.filter((item) => item.status === "running").length;

  statusCurrentNodeEl.textContent = node?.label ?? "-";
  statusElapsedEl.textContent = job?.total_elapsed ?? "-";
  statusBottleneckEl.textContent = job?.bottleneck ?? "-";
  statusQueueEl.textContent = `${queueCount} / ${activeCount}`;
}

function renderDag() {
  const job = currentJob();
  if (!job) {
    return;
  }

  dagNodesEl.innerHTML = "";
  dagEdgesEl.innerHTML = "";

  const width = dagNodesEl.clientWidth || 1180;
  const height = dagNodesEl.clientHeight || 760;
  dagEdgesEl.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const nodeMap = new Map(job.dag.nodes.map((node) => [node.id, node]));

  job.dag.edges.forEach(([sourceId, targetId]) => {
    const source = nodeMap.get(sourceId);
    const target = nodeMap.get(targetId);
    if (!source || !target) {
      return;
    }

    const startX = source.x + 186;
    const startY = source.y + 47;
    const endX = target.x;
    const endY = target.y + 47;
    const curve = Math.max((endX - startX) * 0.42, 40);

    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", `M ${startX} ${startY} C ${startX + curve} ${startY}, ${endX - curve} ${endY}, ${endX} ${endY}`);
    path.setAttribute("stroke", "#485468");
    path.setAttribute("stroke-width", "2.1");
    path.setAttribute("fill", "none");
    dagEdgesEl.appendChild(path);
  });

  job.dag.nodes.forEach((node) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `dag-node ${node.id === selectedNodeId ? "selected" : ""}`;
    button.style.left = `${node.x}px`;
    button.style.top = `${node.y}px`;
    button.innerHTML = `
      <div class="node-status">
        <span class="node-dot status-${node.status}"></span>
        ${node.status.toUpperCase()}
      </div>
      <h3>${node.label}</h3>
      <div class="node-meta">
        <div>elapsed<strong>${node.elapsed}</strong></div>
        <div>score<strong>${node.score}</strong></div>
      </div>
    `;

    button.addEventListener("mouseenter", (event) => showTooltip(node, event));
    button.addEventListener("mousemove", moveTooltip);
    button.addEventListener("mouseleave", hideTooltip);
    button.addEventListener("click", () => {
      selectedNodeId = node.id;
      drawerOpen = true;
      renderDag();
      renderDrawer();
      renderBottomDock();
    });

    dagNodesEl.appendChild(button);
  });
}

function renderDrawer() {
  const node = currentNode();
  if (!node) {
    return;
  }

  nodeTitleEl.textContent = node.label;
  nodeStatusPillEl.textContent = `Status: ${node.status}`;
  nodeScorePillEl.textContent = `Score: ${node.score}`;
  renderKeyValueGrid(overviewGridEl, node.detail.overview);
  renderKeyValueGrid(rulesGridEl, node.detail.rules);
  renderKeyValueGrid(metricsGridEl, node.detail.metrics);
  decisionLogEl.textContent = node.detail.decision_log.join("\n");
  drawerEl.classList.toggle("open", drawerOpen);
}

function renderBottomDock() {
  const node = currentNode();
  const job = currentJob();
  if (!node || !job) {
    return;
  }

  logsPanelEl.textContent = node.detail.decision_log.join("\n");
  renderDockList(artifactsPanelEl, buildArtifacts(node));
  renderDockList(tracePanelEl, buildTrace(job));
}

function renderTabs() {
  tabButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === activeTab);
  });

  tabPanels.forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.panel === activeTab);
  });
}

function buildArtifacts(node) {
  const map = {
    input_video: [
      { title: "video_manifest.json", meta: "Girdi manifesti" }
    ],
    ffmpeg_splitter: [
      { title: "ocr_input_manifest.json", meta: "OCR hazir giris" },
      { title: "asr_input_manifest.json", meta: "ASR hazir giris" }
    ],
    ocr_worker: [
      { title: "ocr_raw.json", meta: "Ham OCR satirlari" },
      { title: "ocr_scores.json", meta: "OCR confidence verileri" }
    ],
    ocr_local_fix: [
      { title: "ada1.json", meta: "Local duzeltme izi" }
    ],
    ocr_gemini_fix: [
      { title: "ocr_gemini_fix.json", meta: "Gemini enrichment" }
    ],
    asr_worker: [
      { title: "transcript.json", meta: "ASR transcript" }
    ],
    asr_filter: [
      { title: "transcript_quality.json", meta: "ASR kalite filtresi" }
    ],
    summary_worker: [
      { title: "summary.json", meta: "Ozet cikisi" }
    ],
    verify_worker: [
      { title: "verify.json", meta: "Confidence + flags" }
    ],
    finalize_worker: [
      { title: "final_report.json", meta: "Final rapor" }
    ],
    sport_input_video: [
      { title: "sport_input_manifest.json", meta: "Spor yayin girdisi" }
    ],
    sport_segment_extract: [
      { title: "segment_windows.json", meta: "Ilk/son segmentler" }
    ],
    sport_asr_worker: [
      { title: "sport_transcript.json", meta: "Mac transcript cikisi" }
    ],
    sport_ocr_worker: [
      { title: "scoreboard_ocr.json", meta: "Skor OCR" }
    ],
    sport_analyze: [
      { title: "sport_analysis.json", meta: "Gemini capraz dogrulama" }
    ],
    sport_finalize: [
      { title: "sport_report.txt", meta: "Spor final cikti" }
    ]
  };

  return map[node.id] || [{ title: "artifact.json", meta: "Ornek artifact" }];
}

function buildTrace(job) {
  return job.trace_rows.map((row) => ({
    title: `${row[0]} -> ${row[2]}`,
    meta: `Ara: ${row[1]} | Karar: ${row[3]}`
  }));
}

function renderDockList(container, items) {
  container.innerHTML = "";

  items.forEach((item) => {
    const div = document.createElement("div");
    div.className = "dock-item";
    div.innerHTML = `
      <div class="dock-item-title">${item.title}</div>
      <div class="dock-item-meta">${item.meta}</div>
    `;
    container.appendChild(div);
  });
}

function renderKeyValueGrid(container, pairs) {
  container.innerHTML = "";

  if (!pairs || pairs.length === 0) {
    const row = document.createElement("div");
    row.className = "kv-row";
    row.innerHTML = `<span class="kv-key">veri</span><span class="kv-value">hazir degil</span>`;
    container.appendChild(row);
    return;
  }

  pairs.forEach(([key, value]) => {
    const row = document.createElement("div");
    row.className = "kv-row";
    row.innerHTML = `
      <span class="kv-key">${key}</span>
      <span class="kv-value">${value}</span>
    `;
    container.appendChild(row);
  });
}

function showTooltip(node, event) {
  tooltipEl.innerHTML = `
    <div><strong>${node.label}</strong></div>
    <div>Status: ${node.status}</div>
    <div>Elapsed: ${node.elapsed}</div>
    <div>Score: ${node.score}</div>
  `;
  tooltipEl.classList.remove("hidden");
  moveTooltip(event);
}

function moveTooltip(event) {
  tooltipEl.style.left = `${event.clientX + 14}px`;
  tooltipEl.style.top = `${event.clientY + 14}px`;
}

function hideTooltip() {
  tooltipEl.classList.add("hidden");
}

bootstrap().catch(() => {
  statusCurrentNodeEl.textContent = "yuklenemedi";
});
