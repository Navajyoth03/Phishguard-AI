// ╔══════════════════════════════════════════════════════════╗
// ║  PhishGuard Extension — popup.js                        ║
// ║  Dual-Mode Architecture: Smart Auto, Local, & Cloud     ║
// ╚══════════════════════════════════════════════════════════╝

'use strict';

// ── CONSTANTS ──────────────────────────────────────────────
const STORAGE_HISTORY_KEY = 'phishguard_history';
const STORAGE_MODE_KEY    = 'phishguard_server_mode';
const STORAGE_BACKEND_KEY = 'phishguard_active_backend';
const MAX_HISTORY         = 50;

const DEFAULT_LOCAL = 'http://127.0.0.1:5000';
const DEFAULT_CLOUD = 'https://phishguard-ai-j381.onrender.com';

const TIER = {
  safe: { color: '#10B981', glow: 'rgba(16,185,129,0.25)', icon: '🟢', label: 'SAFE', action: 'This website appears safe. No significant threats detected.' },
  low: { color: '#06B6D4', glow: 'rgba(6,182,212,0.25)', icon: '🔵', label: 'LOW RISK', action: 'Minimal risk detected. Remain cautious and verify if unsure.' },
  medium: { color: '#F59E0B', glow: 'rgba(245,158,11,0.25)', icon: '🟡', label: 'MEDIUM', action: 'Potential phishing indicators found. Verify this site carefully.' },
  high: { color: '#F97316', glow: 'rgba(249,115,22,0.25)', icon: '🟠', label: 'HIGH RISK', action: 'Strong phishing signals detected. Do NOT enter credentials.' },
  crit: { color: '#EF4444', glow: 'rgba(239,68,68,0.30)', icon: '🔴', label: 'CRITICAL', action: 'Coordinated phishing attack confirmed. DO NOT INTERACT.' },
};

let currentTabId = null;

function getTier(score) {
  if (score >= 0.80) return 'crit';
  if (score >= 0.60) return 'high';
  if (score >= 0.40) return 'medium';
  if (score >= 0.20) return 'low';
  return 'safe';
}

// ── TAB SWITCHING ─────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById(`tab${capitalize(name)}`).classList.add('active');
  document.getElementById(`panel${capitalize(name)}`).classList.add('active');
  if (name === 'history') renderHistory();
}

function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

// ── MAIN INIT ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Bind Tab Switching
  document.getElementById('tabCurrent').addEventListener('click', () => switchTab('current'));
  document.getElementById('tabHistory').addEventListener('click', () => switchTab('history'));
  const tabSim = document.getElementById('tabSim');
  if (tabSim) tabSim.addEventListener('click', () => switchTab('sim'));
  const tabImage = document.getElementById('tabImage');
  if (tabImage) tabImage.addEventListener('click', () => switchTab('image'));

  // Init Image Forensics Inspector
  initImageForensics();

  const btnClear = document.getElementById('clearBtn');
  if (btnClear) btnClear.addEventListener('click', clearHistory);

  // Bind Simulator Buttons
  const btnSafe = document.getElementById('btnSimSafe');
  if (btnSafe) btnSafe.addEventListener('click', () => runSimulation('safe'));
  const btnSpam = document.getElementById('btnSimSpam');
  if (btnSpam) btnSpam.addEventListener('click', () => runSimulation('spam'));
  const btnCrit = document.getElementById('btnSimCrit');
  if (btnCrit) btnCrit.addEventListener('click', () => runSimulation('crit'));

  // Bind Dual-Mode Server Selection Buttons
  const btnAuto = document.getElementById('modeBtnAuto');
  const btnLocal = document.getElementById('modeBtnLocal');
  const btnCloud = document.getElementById('modeBtnCloud');

  if (btnAuto)  btnAuto.addEventListener('click', () => setBackendMode('auto'));
  if (btnLocal) btnLocal.addEventListener('click', () => setBackendMode('local'));
  if (btnCloud) btnCloud.addEventListener('click', () => setBackendMode('cloud'));

  // Listen for storage updates (e.g. smart failover in background)
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === 'local' && changes[STORAGE_BACKEND_KEY]) {
      updateBackendDisplay(changes[STORAGE_BACKEND_KEY].newValue);
    }
  });

  // Query Active Tab
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (!tabs || !tabs[0]) return;
    const tab = tabs[0];
    currentTabId = tab.id;

    // Display URL
    const urlEl = document.getElementById('targetUrl');
    if (urlEl) urlEl.textContent = tab.url || '—';

    // Synchronize initial server backend status
    syncBackendStatus();

    // Look up stored analysis for this tab
    chrome.storage.local.get([tab.id.toString()], (result) => {
      const data = result[tab.id.toString()];

      if (data && !data.error && data.final_risk_score !== undefined) {
        setStatus('online', 'Protected');
        renderAnalysis(data);
      } else if (data && data.error) {
        setStatus('offline', 'Offline');
        renderOffline();
      } else {
        setStatus('busy', 'Analyzing');
        renderWaiting();
      }
    });
  });
});

// ── DUAL-MODE SERVER LOGIC ────────────────────────────────
function setBackendMode(mode) {
  highlightModePill(mode);
  setChipState('busy', 'Switching...');

  chrome.runtime.sendMessage({
    action: 'setServerMode',
    mode: mode,
    tabId: currentTabId
  }, (response) => {
    if (response && response.backend) {
      updateBackendDisplay(response.backend);
    }
    // Refresh current tab display if re-analyzing
    if (currentTabId) {
      setTimeout(() => {
        chrome.storage.local.get([currentTabId.toString()], (res) => {
          const tabData = res[currentTabId.toString()];
          if (tabData && !tabData.error) {
            renderAnalysis(tabData);
          }
        });
      }, 700);
    }
  });
}

async function syncBackendStatus() {
  const stored = await chrome.storage.local.get([STORAGE_MODE_KEY, STORAGE_BACKEND_KEY]);
  const mode = stored[STORAGE_MODE_KEY] || 'auto';
  highlightModePill(mode);

  const targetUrl = (mode === 'local') ? DEFAULT_LOCAL : DEFAULT_CLOUD;
  const isCloud = (mode !== 'local');

  setChipState('busy', isCloud ? 'Connecting...' : 'Checking...');

  // 1. Direct probe from popup for instant UI responsiveness
  try {
    const probeTimeout = isCloud ? 8000 : 1000;
    const res = await fetch(`${targetUrl}/health`, { signal: AbortSignal.timeout(probeTimeout) });
    if (res.ok) {
      const data = await res.json();
      const backendInfo = {
        mode: mode,
        active: isCloud ? 'cloud' : 'local',
        url: targetUrl,
        online: true,
        version: data.version || '3.0'
      };
      await chrome.storage.local.set({ [STORAGE_BACKEND_KEY]: backendInfo });
      updateBackendDisplay(backendInfo);
      return;
    }
  } catch (err) {
    // If auto mode and local failed, probe cloud
    if (mode === 'auto') {
      try {
        const cloudRes = await fetch(`${DEFAULT_CLOUD}/health`, { signal: AbortSignal.timeout(8000) });
        if (cloudRes.ok) {
          const cloudData = await cloudRes.json();
          const backendInfo = {
            mode: 'auto',
            active: 'cloud',
            url: DEFAULT_CLOUD,
            online: true,
            version: cloudData.version || '3.0'
          };
          await chrome.storage.local.set({ [STORAGE_BACKEND_KEY]: backendInfo });
          updateBackendDisplay(backendInfo);
          return;
        }
      } catch (cloudErr) {}
    }
  }

  // 2. Fallback to background query if direct probe had issues
  chrome.runtime.sendMessage({ action: 'getBackendStatus' }, (backend) => {
    if (backend) {
      updateBackendDisplay(backend);
    } else {
      updateBackendDisplay({ mode, active: isCloud ? 'cloud' : 'local', online: false });
    }
  });
}

function highlightModePill(mode) {
  const btnAuto = document.getElementById('modeBtnAuto');
  const btnLocal = document.getElementById('modeBtnLocal');
  const btnCloud = document.getElementById('modeBtnCloud');

  if (btnAuto)  btnAuto.classList.toggle('active', mode === 'auto');
  if (btnLocal) btnLocal.classList.toggle('active', mode === 'local');
  if (btnCloud) btnCloud.classList.toggle('active', mode === 'cloud');
}

function setChipState(stateClass, label) {
  const dot = document.getElementById('backendDot');
  const text = document.getElementById('backendText');
  if (dot) dot.className = 'backend-dot ' + stateClass;
  if (text) text.textContent = label;
}

function updateBackendDisplay(backend) {
  if (!backend) return;
  highlightModePill(backend.mode || 'auto');

  const dot = document.getElementById('backendDot');
  const text = document.getElementById('backendText');
  const tag = document.getElementById('activeBackendTag');

  const isLocal = backend.active === 'local';
  const isOnline = !!backend.online;

  if (dot) {
    if (!isOnline) {
      dot.className = 'backend-dot offline';
    } else {
      dot.className = isLocal ? 'backend-dot local' : 'backend-dot cloud';
    }
  }

  if (text) {
    if (!isOnline) {
      text.textContent = isLocal ? 'Local Offline' : 'Cloud Connecting';
    } else {
      text.textContent = isLocal ? 'Local (Fast)' : 'Cloud (Render)';
    }
  }

  if (tag) {
    updateBackendTag(isLocal ? 'local' : 'cloud');
  }

  if (isOnline) {
    setStatus('online', 'Protected');
  } else {
    setStatus('offline', 'Offline');
  }
}

function updateBackendTag(activeType) {
  const tag = document.getElementById('activeBackendTag');
  if (!tag) return;
  if (activeType === 'cloud') {
    tag.textContent = '☁️ Cloud Render';
    tag.style.background = 'rgba(56, 189, 248, 0.15)';
    tag.style.borderColor = 'rgba(56, 189, 248, 0.35)';
    tag.style.color = '#7DD3FC';
  } else {
    tag.textContent = '⚡ Local Host';
    tag.style.background = 'rgba(16, 185, 129, 0.15)';
    tag.style.borderColor = 'rgba(16, 185, 129, 0.35)';
    tag.style.color = '#A7F3D0';
  }
}

// ── STATUS INDICATOR ─────────────────────────────────────
function setStatus(state, label) {
  const dot = document.getElementById('statusDot');
  const lbl = document.getElementById('statusLabel');
  if (dot) { dot.className = 'status-dot ' + state; }
  if (lbl) { lbl.textContent = label; }
}

// ── OFFLINE STATE ─────────────────────────────────────────
function renderOffline(errMessage) {
  setGaugeScore(null, null);
  setText('riskLabel', 'UNAVAILABLE');
  setText('riskAction', errMessage ? errMessage : 'Start server.py for local demo, or toggle Cloud mode.');
  if (errMessage) {
    setText('geminiText', 'Network or Parse Error: ' + errMessage);
  } else {
    setText('geminiText', 'Selected PhishGuard backend is unreachable. Switch mode or start server.');
  }
}

// ── WAITING STATE ─────────────────────────────────────────
function renderWaiting() {
  setText('riskLabel', 'ANALYZING');
  setText('riskAction', 'Multi-modal analysis in progress across 5 modalities...');
  setText('geminiText', 'Awaiting PhishGuard cross-modal fusion engine...');
}

// ── SIMULATOR ─────────────────────────────────────────────
async function runSimulation(type) {
  switchTab('current');
  setStatus('busy', 'Simulating');
  renderWaiting();
  setGaugeScore(null, null);

  let url, text;
  if (type === 'safe') {
    url = 'https://www.icicibank.com/personal-banking/login';
    text = 'Welcome to ICICI Bank. Please enter your user ID to login securely. Manage your finances with ease.';
  } else if (type === 'spam') {
    url = 'http://best-offers-online-deal-55.xyz/click?id=12';
    text = 'You have a new promotional offer! Click here to view your exclusive deal now.';
  } else {
    url = 'http://paypal-auth-secure-update-17.tk/login';
    text = 'URGENT: Your PayPal account has been restricted due to unauthorized access. Call this number immediately or enter your OTP and password below or we will suspend it.';
  }

  // Update UI to reflect the simulated URL instead of the actual Chrome Tab
  const urlEl = document.getElementById('targetUrl');
  if (urlEl) urlEl.textContent = url.substring(0, 55) + (url.length > 55 ? '...' : '');

  // Get active backend URL dynamically
  const stored = await chrome.storage.local.get([STORAGE_MODE_KEY, STORAGE_BACKEND_KEY]);
  const mode = stored[STORAGE_MODE_KEY] || 'auto';
  let targetUrl = (mode === 'local') ? DEFAULT_LOCAL : DEFAULT_CLOUD;

  updateBackendTag(mode === 'local' ? 'local' : 'cloud');

  try {
    const res = await fetch(`${targetUrl}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: url, text: text }),
      signal: AbortSignal.timeout(20000)
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    setStatus('online', 'Protected');
    data._backend = { mode, active: mode === 'local' ? 'local' : 'cloud', url: targetUrl };
    renderAnalysis(data);
  } catch (err) {
    if (mode === 'auto' && targetUrl === DEFAULT_LOCAL) {
      try {
        const cloudRes = await fetch(`${DEFAULT_CLOUD}/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: url, text: text }),
          signal: AbortSignal.timeout(20000)
        });
        if (!cloudRes.ok) throw new Error(`HTTP ${cloudRes.status}`);
        const cloudData = await cloudRes.json();
        setStatus('online', 'Protected');
        cloudData._backend = { mode: 'auto', active: 'cloud', url: DEFAULT_CLOUD };
        updateBackendTag('cloud');
        renderAnalysis(cloudData);
        return;
      } catch (cloudErr) {}
    }
    console.error('Simulation error:', err.message || err);
    setStatus('offline', 'Offline');
    renderOffline(err.message || String(err));
  }
}

// ── MAIN RENDER ───────────────────────────────────────────
function renderAnalysis(data) {
  const score = data.final_risk_score || 0;
  const tier = getTier(score);
  const t = TIER[tier];

  // 0. Update Backend Tag
  if (data._backend?.active) {
    updateBackendTag(data._backend.active);
  }

  // 1. Gauge
  setGaugeScore(score, t.color);

  // 2. Risk badge
  const badge = document.getElementById('riskBadge');
  if (badge) {
    badge.style.background = `${t.color}18`;
    badge.style.borderColor = `${t.color}45`;
  }
  setText('riskIcon', t.icon);
  setStyledText('riskLabel', t.label, t.color);
  setText('riskAction', data.action_required || t.action);

  // 3. Modality bars
  const risks = data.individual_risks || {};
  renderBar('Url', risks.url ?? 0);
  renderBar('Text', risks.text ?? 0);
  renderBar('Image', risks.image ?? 0);
  renderBar('Audio', risks.audio ?? 0);
  renderBar('Video', risks.video ?? 0);

  // 4. Model info badges
  const infoRow = document.getElementById('modelInfoRow');
  if (infoRow) {
    infoRow.style.display = 'flex';
    const uMethod = (data.url_method || '').includes('ml') ? '✅ ML' : 'Rule';
    const tMethod = (data.text_method || '').includes('ml') ? '✅ ML' : 'Rule';
    setText('urlMethodBadge', `URL: ${uMethod}`);
    setText('textMethodBadge', `Text: ${tMethod}`);
  }

  // 5. Fusion agreement
  const fusionRow = document.getElementById('fusionRow');
  const fusionVal = document.getElementById('fusionValue');
  if (data.agreement_multiplier !== undefined && fusionRow && fusionVal) {
    fusionRow.style.display = 'flex';
    const mult = data.agreement_multiplier;
    fusionVal.textContent = mult.toFixed(3) + (mult >= 1 ? '  ↑ Agreement' : '  ↓ Conflict');
  }

  // 6. Red flags
  const allFlags = [
    ...(data.red_flags || []),
    ...(data.video_indicators || []).slice(0, 3),
    ...(data.url_indicators || []).slice(0, 3),
    ...(data.text_indicators || []).slice(0, 2),
    ...(data.image_indicators || []).slice(0, 2),
  ].filter(Boolean).slice(0, 7);

  const flagsCard = document.getElementById('flagsCard');
  const flagsList = document.getElementById('flagsList');
  if (allFlags.length > 0 && score >= 0.35 && flagsCard && flagsList) {
    flagsCard.style.display = 'block';
    flagsList.innerHTML = allFlags
      .map(f => `<li>${escapeHTML(f)}</li>`)
      .join('');
  } else if (flagsCard) {
    flagsCard.style.display = 'none';
  }

  // 6b. SHAP Analysis
  const shapCard = document.getElementById('shapCard');
  const shapCont = document.getElementById('shapContainer');
  if (data.shap_values && data.shap_values.length > 0 && shapCard && shapCont) {
    shapCard.style.display = 'block';
    shapCont.innerHTML = data.shap_values.map(s => {
      const isPos = s.direction === '+';
      const w = Math.min(Math.abs(s.impact) * 200, 48);
      return `
        <div class="shap-row" style="margin-top:10px;">
          <div class="shap-label ${isPos ? 'left' : 'right'}">${escapeHTML(s.feature)}</div>
          <div class="shap-bar ${isPos ? 'positive' : 'negative'}" style="width: ${w}%"></div>
        </div>
      `;
    }).join('');
  } else if (shapCard) {
    shapCard.style.display = 'none';
  }

  // 7. Gemini explanation
  setText('geminiText', data.ai_explanation || 'No explanation available.');
}

// ── GAUGE ─────────────────────────────────────────────────
function setGaugeScore(score, color) {
  const fill = document.getElementById('gaugeFill');
  const label = document.getElementById('gaugeScore');
  const ARC = 251.3;

  if (score === null || score === undefined) {
    if (fill) fill.style.strokeDashoffset = ARC;
    if (label) { label.textContent = '--'; label.style.fill = 'rgba(255,255,255,0.3)'; }
    return;
  }

  const offset = ARC - (ARC * score);
  if (fill) {
    fill.style.strokeDashoffset = offset;
  }
  if (label) {
    label.textContent = Math.round(score * 100) + '%';
    label.style.fill = color || 'white';
    label.style.filter = `drop-shadow(0 0 6px ${color || 'white'}66)`;
  }
}

// ── MODALITY BAR ──────────────────────────────────────────
function renderBar(name, score) {
  const bar = document.getElementById(`bar${name}`);
  const label = document.getElementById(`pct${name}`);
  if (!bar || !label) return;

  const tier = getTier(score);
  const color = TIER[tier].color;
  const pct = Math.round(score * 100);

  setTimeout(() => {
    bar.style.width = pct + '%';
    bar.style.background = color;
    bar.style.boxShadow = `0 0 6px ${color}80`;
  }, 280);

  label.textContent = pct + '%';
  label.style.color = color;
}

// ── HISTORY ───────────────────────────────────────────────
function renderHistory() {
  chrome.storage.local.get([STORAGE_HISTORY_KEY], (result) => {
    const items = result[STORAGE_HISTORY_KEY] || [];
    const list = document.getElementById('historyList');
    if (!list) return;

    if (items.length === 0) {
      list.innerHTML = '<div class="history-empty">No analyses yet. Browse some websites!</div>';
      return;
    }

    list.innerHTML = items.slice(0, MAX_HISTORY).map(item => {
      const tier = getTier(item.score || 0);
      const t = TIER[tier];
      const pct = Math.round((item.score || 0) * 100);
      const ts = item.timestamp
        ? new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        : '';
      const bTag = item.backend === 'cloud' ? '☁️ Cloud' : '⚡ Local';

      return `
        <div class="history-item">
          <div class="hist-dot" style="background:${t.color}; box-shadow:0 0 5px ${t.color}80;"></div>
          <div class="hist-body">
            <div class="hist-url">${escapeHTML((item.url || '').substring(0, 55))}</div>
            <div class="hist-time">${ts} · ${t.label} · <span style="opacity:0.8">${bTag}</span></div>
          </div>
          <div class="hist-score" style="color:${t.color}">${pct}%</div>
        </div>
      `;
    }).join('');
  });
}

function clearHistory() {
  chrome.storage.local.remove(STORAGE_HISTORY_KEY, () => {
    renderHistory();
  });
}

// ── HELPERS ───────────────────────────────────────────────
function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function setStyledText(id, value, color) {
  const el = document.getElementById(id);
  if (el) { el.textContent = value; el.style.color = color; }
}

function escapeHTML(str) {
  return (str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// ── IMAGE FORENSICS ENGINE ────────────────────────────────
function initImageForensics() {
  const dropzone = document.getElementById('imageDropzone');
  const fileInput = document.getElementById('imageFileInput');
  const btnToggleHeatmap = document.getElementById('btnToggleHeatmap');

  if (!dropzone || !fileInput) return;

  // Click dropzone to open file dialog
  dropzone.addEventListener('click', () => fileInput.click());

  // File input change
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleImageFile(e.target.files[0]);
    }
  });

  // Drag and drop events
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageFile(e.dataTransfer.files[0]);
    }
  });

  // Global paste handler for Ctrl+V
  document.addEventListener('paste', (e) => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (const item of items) {
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        const file = item.getAsFile();
        switchTab('image');
        handleImageFile(file);
        break;
      }
    }
  });

  // Heatmap toggle
  if (btnToggleHeatmap) {
    btnToggleHeatmap.addEventListener('click', () => {
      const heatmapContainer = document.getElementById('heatmapContainer');
      if (heatmapContainer) {
        const isHidden = heatmapContainer.style.display === 'none';
        heatmapContainer.style.display = isHidden ? 'block' : 'none';
        btnToggleHeatmap.innerHTML = isHidden
          ? '<span>🔥</span> Hide ELA Tampering Heatmap'
          : '<span>🔥</span> View ELA Tampering Heatmap';
      }
    });
  }

  // Check if background script inspected an image via context menu
  chrome.storage.local.get(['phishguard_last_image_result', 'phishguard_last_image_time'], (res) => {
    if (res && res.phishguard_last_image_result) {
      const data = res.phishguard_last_image_result;
      renderImageResult(data);
      // Auto-switch to image tab if inspected recently (< 10 minutes ago)
      const elapsed = Date.now() - (res.phishguard_last_image_time || 0);
      if (elapsed < 10 * 60 * 1000) {
        switchTab('image');
      }
    }
  });
}

async function handleImageFile(file) {
  if (!file) return;

  const previewCard = document.getElementById('imagePreviewCard');
  const previewImg = document.getElementById('imagePreviewImg');
  const previewFilename = document.getElementById('previewFilename');
  const previewDimensions = document.getElementById('previewDimensions');
  const spinner = document.getElementById('imageSpinner');
  const resultCard = document.getElementById('imageResultCard');

  previewFilename.textContent = file.name || 'Pasted Image';
  previewDimensions.textContent = `${(file.size / 1024).toFixed(1)} KB`;
  previewCard.style.display = 'block';
  spinner.style.display = 'flex';
  resultCard.style.display = 'none';

  // Read preview
  const reader = new FileReader();
  reader.onload = async (e) => {
    previewImg.src = e.target.result;

    try {
      const backend = await getActiveBackendUrl();
      const base64Data = e.target.result;

      const resp = await fetch(`${backend}/analyze_image_file`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: base64Data,
          filename: file.name || 'image.png'
        })
      });

      if (!resp.ok) {
        throw new Error(`Server returned HTTP ${resp.status}`);
      }

      const data = await resp.json();
      spinner.style.display = 'none';
      renderImageResult(data);
    } catch (err) {
      spinner.style.display = 'none';
      alert(`Forensic analysis failed: ${err.message}`);
    }
  };
  reader.readAsDataURL(file);
}

function renderImageResult(data) {
  if (!data) return;

  const previewCard = document.getElementById('imagePreviewCard');
  const previewDimensions = document.getElementById('previewDimensions');
  const previewFilename = document.getElementById('previewFilename');
  const resultCard = document.getElementById('imageResultCard');
  const badge = document.getElementById('imageResultBadge');
  const verdictIcon = document.getElementById('imageVerdictIcon');
  const verdictText = document.getElementById('imageVerdictText');
  const riskScore = document.getElementById('imageRiskScore');
  const classification = document.getElementById('imageClassification');

  const metricEla = document.getElementById('metricEla');
  const metricFft = document.getElementById('metricFft');
  const metricEdge = document.getElementById('metricEdge');
  const metricNoise = document.getElementById('metricNoise');
  const indicatorsList = document.getElementById('imageIndicatorsList');
  const heatmapImg = document.getElementById('heatmapImg');

  previewCard.style.display = 'block';
  resultCard.style.display = 'block';

  if (data.dimensions) previewDimensions.textContent = data.dimensions;
  if (data.filename) previewFilename.textContent = data.filename;

  const pct = Math.round((data.risk_score || 0) * 100);
  const isTampered = data.is_tampered;

  verdictIcon.textContent = isTampered ? '⚠️' : '✅';
  verdictText.textContent = data.verdict || (isTampered ? 'TAMPERED / AI' : 'AUTHENTIC IMAGE');
  riskScore.textContent = `${pct}% RISK`;

  if (isTampered) {
    badge.style.background = 'rgba(239, 68, 68, 0.15)';
    badge.style.color = '#EF4444';
    badge.style.border = '1px solid rgba(239, 68, 68, 0.3)';
    riskScore.style.color = '#EF4444';
  } else {
    badge.style.background = 'rgba(16, 185, 129, 0.15)';
    badge.style.color = '#10B981';
    badge.style.border = '1px solid rgba(16, 185, 129, 0.3)';
    riskScore.style.color = '#10B981';
  }

  classification.textContent = data.detection_type || (isTampered ? 'Likely Manipulated' : 'Authentic Camera Shot');

  // Metrics
  const metrics = data.metrics || {};
  if (metrics.ela) metricEla.textContent = `${metrics.ela.tile_disparity_ratio}x`;
  if (metrics.fft) metricFft.textContent = metrics.fft.azimuthal_variance;
  if (metrics.laplacian) metricEdge.textContent = `${metrics.laplacian.edge_disparity}x`;
  if (metrics.noise) metricNoise.textContent = `${metrics.noise.noise_disparity}x`;

  // Indicators
  indicatorsList.innerHTML = '';
  const indicators = data.indicators || [];
  indicators.forEach(ind => {
    const item = document.createElement('div');
    item.className = `image-indicator-item ${isTampered ? 'tampered' : 'authentic'}`;
    item.innerHTML = `<span>${isTampered ? '⚠️' : '✓'}</span> <span>${escapeHTML(ind)}</span>`;
    indicatorsList.appendChild(item);
  });

  // Heatmap
  if (data.heatmap_b64 && heatmapImg) {
    heatmapImg.src = data.heatmap_b64;
  }
}

async function getActiveBackendUrl() {
  const stored = await chrome.storage.local.get([STORAGE_BACKEND_KEY]);
  if (stored && stored[STORAGE_BACKEND_KEY] && stored[STORAGE_BACKEND_KEY].url) {
    return stored[STORAGE_BACKEND_KEY].url;
  }
  return DEFAULT_LOCAL;
}
