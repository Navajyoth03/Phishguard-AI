// PhishGuard Extension — background.js (Service Worker)
// Dual-Mode Architecture: Smart Auto-Failover + Cloud Production + Local Demo

'use strict';

const DEFAULT_LOCAL_URL = 'http://127.0.0.1:5000';
const DEFAULT_CLOUD_URL = 'https://phishguard-ai-j381.onrender.com';

const STORAGE_KEY_MODE           = 'phishguard_server_mode';        // 'auto' | 'local' | 'cloud'
const STORAGE_KEY_CUSTOM_CLOUD   = 'phishguard_custom_cloud_url';
const STORAGE_KEY_CUSTOM_LOCAL   = 'phishguard_custom_local_url';
const STORAGE_KEY_ACTIVE_BACKEND = 'phishguard_active_backend';
const HISTORY_KEY                = 'phishguard_history';
const MAX_HISTORY                = 50;

// ── ENDPOINT RESOLUTION & HEALTH ──────────────────────────────
async function getEndpoints() {
  const data = await chrome.storage.local.get([STORAGE_KEY_CUSTOM_LOCAL, STORAGE_KEY_CUSTOM_CLOUD]);
  return {
    local: (data[STORAGE_KEY_CUSTOM_LOCAL] || DEFAULT_LOCAL_URL).replace(/\/+$/, ''),
    cloud: (data[STORAGE_KEY_CUSTOM_CLOUD] || DEFAULT_CLOUD_URL).replace(/\/+$/, '')
  };
}

async function pingHealth(baseUrl, timeoutMs = 8000) {
  try {
    const res = await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(timeoutMs) });
    if (res.ok) {
      const json = await res.json();
      return { online: true, version: json.version || '3.0' };
    }
  } catch (e) {
    // offline or unreachable
  }
  return { online: false, version: null };
}

async function resolveActiveBackend(forceMode = null) {
  const endpoints = await getEndpoints();
  let mode = forceMode;
  if (!mode) {
    const stored = await chrome.storage.local.get([STORAGE_KEY_MODE]);
    mode = stored[STORAGE_KEY_MODE] || 'auto';
  }

  let activeTarget = 'local';
  let targetUrl = endpoints.local;
  let isOnline = false;

  if (mode === 'local') {
    activeTarget = 'local';
    targetUrl = endpoints.local;
    const health = await pingHealth(targetUrl, 1500);
    isOnline = health.online;
  } else if (mode === 'cloud') {
    activeTarget = 'cloud';
    targetUrl = endpoints.cloud;
    const health = await pingHealth(targetUrl, 12000);
    isOnline = health.online;
  } else {
    // 'auto' mode: Prioritize ultra-fast local backend; fall back to cloud if offline
    const localHealth = await pingHealth(endpoints.local, 800);
    if (localHealth.online) {
      activeTarget = 'local';
      targetUrl = endpoints.local;
      isOnline = true;
    } else {
      activeTarget = 'cloud';
      targetUrl = endpoints.cloud;
      const cloudHealth = await pingHealth(endpoints.cloud, 12000);
      isOnline = cloudHealth.online;
    }
  }

  const backendInfo = {
    mode: mode,
    active: activeTarget,
    url: targetUrl,
    online: isOnline,
    endpoints: endpoints,
    lastChecked: new Date().toISOString()
  };

  await chrome.storage.local.set({ [STORAGE_KEY_ACTIVE_BACKEND]: backendInfo });
  return backendInfo;
}

// ── STARTUP ───────────────────────────────────────────────────
function setupContextMenu() {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: 'phishguard-inspect-image',
      title: 'Inspect Image Forensics (PhishGuard)',
      contexts: ['image']
    });
  });
}

chrome.runtime.onInstalled.addListener(async () => {
  console.log('[PhishGuard] Extension installed in Dual-Mode.');
  setupContextMenu();
  const backend = await resolveActiveBackend();
  console.log(`[PhishGuard] Initial backend: ${backend.active} (${backend.url}) [${backend.online ? 'ONLINE' : 'OFFLINE'}]`);
});

chrome.runtime.onStartup.addListener(async () => {
  setupContextMenu();
  await resolveActiveBackend();
});

// ── CONTEXT MENU IMAGE FORENSICS ───────────────────────────────
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === 'phishguard-inspect-image') {
    const imgUrl = info.srcUrl;
    if (!imgUrl) return;

    // 1. Notify content script to display the loading HUD directly on the page
    if (tab && tab.id) {
      try {
        chrome.tabs.sendMessage(tab.id, {
          action: 'showImageForensicsToast',
          status: 'loading'
        });
      } catch (e) {}
    }

    try {
      const backend = await resolveActiveBackend();
      const res = await fetch(`${backend.url}/analyze_image_file`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_url: imgUrl })
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }

      const data = await res.json();
      await chrome.storage.local.set({
        'phishguard_last_image_result': data,
        'phishguard_last_image_time': Date.now()
      });

      const pct = Math.round((data.risk_score || 0) * 100);
      const isTampered = data.is_tampered;
      const title = isTampered ? `⚠️ ${pct}% Tampered / AI Detected` : `✅ ${100 - pct}% Authentic Image`;
      const message = isTampered
        ? (data.indicators && data.indicators[0] ? data.indicators[0] : 'Manipulated image elements detected.')
        : 'Uniform optical sensor noise and authentic compression verified.';

      // 2. Send completed forensic results to the on-page HUD
      if (tab && tab.id) {
        try {
          chrome.tabs.sendMessage(tab.id, {
            action: 'showImageForensicsToast',
            status: 'done',
            data: data
          });
        } catch (e) {}

        // Update badge on active tab
        chrome.action.setBadgeText({ tabId: tab.id, text: `${pct}%` });
        chrome.action.setBadgeBackgroundColor({
          tabId: tab.id,
          color: isTampered ? '#EF4444' : '#10B981'
        });
      }

      try {
        chrome.notifications.create({
          type: 'basic',
          iconUrl: 'icons/icon48.png',
          title: title,
          message: message
        });
      } catch (e) {}

    } catch (err) {
      console.error('[PhishGuard] Context menu image analysis failed:', err);
      if (tab && tab.id) {
        try {
          chrome.tabs.sendMessage(tab.id, {
            action: 'showImageForensicsToast',
            status: 'error',
            message: `Could not analyze image (${err.message}). Check server status in extension popup.`
          });
        } catch (e) {}
      }
    }
  }
});

// ── TAB ANALYSIS ──────────────────────────────────────────────
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status !== 'complete') return;
  if (!tab.url) return;
  if (!tab.url.startsWith('http://') && !tab.url.startsWith('https://') && !tab.url.startsWith('file://')) return;

  await analyzeTab(tabId, tab);
});

async function analyzeTab(tabId, tab, liveVideoFrames = null, playerRect = null) {
  // Give DOM a brief moment to render
  await sleep(400);

  let pageText = '';
  let domFeatures = {};
  let videoFrames = liveVideoFrames || [];

  try {
    const res = await chrome.scripting.executeScript({
      target: { tabId },
      func: async () => {
        try {
          let t = document.body.innerText || document.body.textContent || '';
          let textContent = t.replace(/\s+/g, ' ').trim().substring(0, 3500);

          let videoEl = document.querySelector('video');
          let hasVideo = !!videoEl;
          let frames = [];

          if (videoEl) {
            try {
              const canvas = document.createElement('canvas');
              canvas.width = 320;
              canvas.height = 240;
              const ctx = canvas.getContext('2d');

              for (let i = 0; i < 6; i++) {
                try {
                  ctx.drawImage(videoEl, 0, 0, 320, 240);
                  const dataUrl = canvas.toDataURL('image/jpeg', 0.5);
                  if (dataUrl && dataUrl.length > 200) {
                    frames.push(dataUrl.split(',')[1]);
                  }
                } catch (corsErr) {
                  break; // Cross-origin video, will fallback to tab capture
                }
                await new Promise(r => setTimeout(r, 100));
              }
            } catch (canvasErr) {}
          }

          // Visual & DOM Forensics
          const currentHost = window.location.hostname.toLowerCase();
          const imgs = Array.from(document.querySelectorAll('img'));
          let extImgCount = 0;
          let hasLogoMatch = false;
          const brandKeywords = ['paypal', 'google', 'amazon', 'microsoft', 'apple', 'netflix', 'sbi', 'hdfc', 'icici', 'paytm', 'login', 'signin', 'secure', 'verify'];

          imgs.forEach(img => {
            const src = (img.src || '').toLowerCase();
            const alt = (img.alt || '').toLowerCase();
            const cls = (img.className || '').toLowerCase();
            if (src.startsWith('http') && !src.includes(currentHost)) {
              extImgCount++;
            }
            if (brandKeywords.some(bk => src.includes(bk) || alt.includes(bk) || cls.includes(bk))) {
              hasLogoMatch = true;
            }
          });
          const extImgRatio = imgs.length > 0 ? (extImgCount / imgs.length) : 0.0;

          const faviconEl = document.querySelector('link[rel*="icon"]');
          const faviconUrl = faviconEl ? (faviconEl.href || '') : '';
          const faviconIsExternal = faviconUrl.startsWith('http') && !faviconUrl.toLowerCase().includes(currentHost);

          const hiddenCount = document.querySelectorAll('[style*="display: none"], [style*="display:none"], [hidden], [style*="visibility: hidden"]').length;
          const iframeCount = document.querySelectorAll('iframe').length;
          const inputCount = document.querySelectorAll('input').length;
          const hasLoginForm = document.querySelectorAll('form input[type="password"], form input[type="email"], form input[type="text"]').length >= 2;

          let hasFixedOverlay = false;
          try {
            hasFixedOverlay = Array.from(document.querySelectorAll('div, section, modal, dialog')).some(el => {
              const style = window.getComputedStyle(el);
              return (style.position === 'fixed' || style.position === 'absolute') && parseInt(style.zIndex || '0') >= 100;
            });
          } catch (_) {}

          // Audio & Vishing Forensics
          const audioEls = Array.from(document.querySelectorAll('audio'));
          const hasAudio = audioEls.length > 0;
          const hasAutoplayAudio = audioEls.some(a => a.autoplay || a.loop);
          const audioSrcs = audioEls.map(a => (a.src || '').toLowerCase());
          const hasAlarmAudio = audioSrcs.some(src => /alarm|siren|alert|warning|beep|emergency|police|critical/.test(src));
          const hasTtsAudio = audioSrcs.some(src => /tts|speech|voice|synthesiz|say/.test(src));
          const hasWebAudio = typeof window.AudioContext !== 'undefined' || typeof window.webkitAudioContext !== 'undefined';

          let features = {
            has_video: hasVideo,
            frames_captured: frames.length,
            has_audio: hasAudio,
            has_autoplay_audio: hasAutoplayAudio,
            has_alarm_audio: hasAlarmAudio,
            has_tts_audio: hasTtsAudio,
            has_web_audio: hasWebAudio,
            num_images: imgs.length,
            ext_img_ratio: extImgRatio,
            has_password: document.querySelectorAll('input[type="password"]').length > 0,
            has_login_form: hasLoginForm,
            has_logo_match: hasLogoMatch,
            favicon_is_external: faviconIsExternal,
            hidden_count: hiddenCount,
            iframe_count: iframeCount,
            input_count: inputCount,
            has_fixed_overlay: hasFixedOverlay
          };
          return { text: textContent, features: features, frames: frames };
        } catch { return { text: '', features: {}, frames: [] }; }
      },
    });

    if (res?.[0]?.result) {
      pageText = res[0].result.text;
      domFeatures = res[0].result.features;
      videoFrames = res[0].result.frames || [];
    }
  } catch (e) {
    console.warn(`[PhishGuard] Script injection note for tab ${tabId}:`, e.message);
  }

  // Cross-Origin / Screen Capture Fallback (e.g. YouTube, iframes, external CDNs)
  if (domFeatures.has_video && videoFrames.length === 0 && tab.windowId) {
    try {
      for (let i = 0; i < 4; i++) {
        const tabCap = await chrome.tabs.captureVisibleTab(tab.windowId, { format: 'jpeg', quality: 50 });
        if (tabCap && tabCap.includes(',')) {
          videoFrames.push(tabCap.split(',')[1]);
        }
        await sleep(150);
      }
    } catch (capErr) {
      console.warn('[PhishGuard] Tab capture fallback note:', capErr.message);
    }
  }

  const payload = {
    url: tab.url,
    text: pageText,
    dom_features: domFeatures,
    video_frames: videoFrames,
    player_rect: playerRect
  };

  // Resolve backend and execute request with smart failover
  let backend = await resolveActiveBackend();
  let data = null;

  try {
    const res = await fetch(`${backend.url}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(60000)
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
    data._backend = {
      mode: backend.mode,
      active: backend.active,
      url: backend.url
    };
  } catch (err) {
    // If in auto mode and local failed, seamlessly failover to Cloud Production!
    if (backend.mode === 'auto' && backend.active === 'local') {
      console.warn('[PhishGuard] Local backend unreachable. Auto-failing over to Cloud Production...');
      try {
        const endpoints = await getEndpoints();
        const cloudRes = await fetch(`${endpoints.cloud}/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: AbortSignal.timeout(60000)
        });
        if (!cloudRes.ok) throw new Error(`Cloud HTTP ${cloudRes.status}`);
        data = await cloudRes.json();
        data._backend = {
          mode: 'auto',
          active: 'cloud',
          url: endpoints.cloud
        };
        // Update stored active backend state
        await chrome.storage.local.set({
          [STORAGE_KEY_ACTIVE_BACKEND]: {
            mode: 'auto',
            active: 'cloud',
            url: endpoints.cloud,
            online: true,
            endpoints: endpoints,
            lastChecked: new Date().toISOString()
          }
        });
      } catch (cloudErr) {
        console.error('[PhishGuard] Cloud failover also failed:', cloudErr.message);
      }
    }
  }

  if (!data) {
    console.warn(`[PhishGuard] Analysis failed for ${tab.url}`);
    await chrome.storage.local.set({ [tabId.toString()]: { error: true, url: tab.url, _backend: backend } });
    setBadge(tabId, '?', '#6B7280');
    return;
  }

  if (data.status === 'ignored') return;

  // Retain Peak Video Threat across continuous playback throughout the video
  if (liveVideoFrames && liveVideoFrames.length > 0) {
    const existing = await chrome.storage.local.get([tabId.toString()]);
    const old = existing[tabId.toString()];
    if (old && old.individual_risks?.video) {
      const prevVideoRisk = old.individual_risks.video || 0;
      const currentVideoRisk = data.individual_risks?.video || 0;
      if (prevVideoRisk > currentVideoRisk) {
        data.individual_risks.video = prevVideoRisk;
        if (old.final_risk_score > data.final_risk_score) {
          data.final_risk_score = old.final_risk_score;
          data.risk_level = old.risk_level;
        }
        if (old.video_indicators && old.video_indicators.length > 0) {
          data.video_indicators = Array.from(new Set([...old.video_indicators, ...(data.video_indicators || [])]));
        }
      }
    }
  }

  // Save per-tab result for popup
  await chrome.storage.local.set({ [tabId.toString()]: data });

  // Append to global history
  await appendHistory({
    url: (tab.url || '').substring(0, 100),
    score: data.final_risk_score,
    level: data.risk_level,
    backend: data._backend?.active || 'local',
    timestamp: data.timestamp || new Date().toISOString(),
  });

  // Update badge
  const score = data.final_risk_score || 0;
  if (score >= 0.80) setBadge(tabId, '!!', '#EF4444');
  else if (score >= 0.60) setBadge(tabId, '!', '#F97316');
  else if (score >= 0.40) setBadge(tabId, '~', '#F59E0B');
  else setBadge(tabId, '✓', '#10B981');

  console.log(`[PhishGuard] [${data._backend?.active?.toUpperCase()}] ${tab.url.substring(0, 60)} → ${(score * 100).toFixed(0)}% ${data.risk_level}`);

  // Inject Warning Overlay ONLY for confirmed multi-modal critical threats
  // NEVER show full-screen blocking overlay on trusted sites or if only ONE model has high risk!
  let host = '';
  try { host = (new URL(tab.url || 'http://localhost')).hostname.toLowerCase(); } catch (_) {}

  const TRUSTED_PLATFORMS = [
    'youtube.com', 'google.com', 'google.co.in', 'wikipedia.org', 'wikimedia.org',
    'github.com', 'reddit.com', 'twitter.com', 'x.com', 'linkedin.com', 'instagram.com',
    'netflix.com', 'amazon.com', 'amazon.in', 'microsoft.com', 'apple.com',
    'stackoverflow.com', 'medium.com', 'spotify.com', 'hotstar.com', 'twitch.tv',
    'vimeo.com', 'huggingface.co', 'claude.ai', 'chatgpt.com', 'openai.com',
    'perplexity.ai', 'cursor.com', 'coursera.org'
  ];
  const isTrustedPlatform = TRUSTED_PLATFORMS.some(d => host === d || host.endsWith('.' + d));
  const isEducationalOrGov = /(\.edu|\.gov|\.mil|\.ac\.in|\.edu\.in|\.ac\.uk|\.edu\.au)$/i.test(host);

  // Count how many distinct models flagged high danger (risk >= 0.70)
  const risks = data.individual_risks || {};
  const dangerousModelsCount = Object.keys(risks).filter(k => risks[k] >= 0.70).length;

  const isVideoHighRisk = (data.modalities && data.modalities.video && data.modalities.video.risk_score >= 0.70);
  const isConfirmedDeepfakeExploit = (isVideoHighRisk && (data.video_indicators?.length >= 2 || data.is_deepfake));
  const hasMultiModalAgreement = (dangerousModelsCount >= 2);

  // Overlay triggers ONLY if:
  // 1. NOT a trusted public platform or educational/government domain
  // 2. AND requires MULTI-MODAL agreement (at least 2 independent models >= 70% and overall score >= 85%)
  //    OR a verified multi-indicator deepfake exploit
  const shouldShowOverlay = !isTrustedPlatform && !isEducationalOrGov && (
    (score >= 0.85 && hasMultiModalAgreement) ||
    isConfirmedDeepfakeExploit
  );

  if (shouldShowOverlay) {
    chrome.scripting.executeScript({
      target: { tabId },
      func: (riskScore, explanation, backendName) => {
        if (document.getElementById('phishguard-warning-overlay')) return;
        const overlay = document.createElement('div');
        overlay.id = 'phishguard-warning-overlay';
        Object.assign(overlay.style, {
          position: 'fixed', top: '0', left: '0', width: '100vw', height: '100vh',
          backgroundColor: 'rgba(20, 0, 0, 0.90)', color: '#fff',
          zIndex: '2147483647', display: 'flex', flexDirection: 'column',
          justifyContent: 'center', alignItems: 'center', fontFamily: 'system-ui, sans-serif',
          backdropFilter: 'blur(10px)', textAlign: 'center', padding: '20px', boxSizing: 'border-box'
        });
        overlay.innerHTML = `
          <div style="background: #111; padding: 40px; border-radius: 16px; max-width: 600px; box-shadow: 0 0 50px rgba(239,68,68,0.3); border: 2px solid #ef4444;">
            <div style="font-size: 4rem; margin-bottom: 10px;">🛡️</div>
            <h1 style="font-size: 2rem; margin: 0 0 10px 0; color: #ef4444;">PhishGuard Alert</h1>
            <h2 style="font-size: 1.2rem; margin: 0 0 10px 0; color: #fca5a5;">High Risk Detected: ${(riskScore * 100).toFixed(0)}%</h2>
            <div style="font-size: 0.8rem; color: #888; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;">Protected by PhishGuard AI (${backendName})</div>
            <p style="font-size: 1rem; color: #ccc; margin-bottom: 30px; background: rgba(239,68,68,0.1); padding: 15px; border-radius: 8px; line-height: 1.5; white-space: pre-line; text-align: left;">
              ${explanation || 'This page exhibits highly suspicious patterns and may be attempting to steal your information.'}
            </p>
            <button id="phishguard-dismiss-btn" style="background: #ef4444; color: white; border: none; padding: 12px 24px; font-size: 1rem; border-radius: 6px; cursor: pointer; font-weight: bold; width: 100%; transition: background 0.2s;">
              I understand the risks, proceed anyway
            </button>
          </div>
        `;
        document.body.appendChild(overlay);
        document.getElementById('phishguard-dismiss-btn').addEventListener('click', () => {
          overlay.remove();
        });
      },
      args: [
        Math.max(score, (data.individual_risks?.video || 0)),
        (data.video_indicators && data.video_indicators.length > 0)
          ? `⚠️ SYNTHETIC DEEPFAKE ARTIFACTS DETECTED:\n${data.video_indicators.map(i => `• ${i}`).join('\n')}`
          : (data.ai_explanation || 'This page exhibits highly suspicious patterns.'),
        data._backend?.active === 'cloud' ? 'Cloud Production' : 'Local Engine'
      ]
    }).catch(err => console.warn('[PhishGuard] Failed to inject warning overlay:', err));
  }
}

// ── MESSAGE LISTENER (POPUP CONTROLS) ─────────────────────────
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'setServerMode') {
    (async () => {
      const mode = request.mode; // 'auto', 'local', or 'cloud'
      await chrome.storage.local.set({ [STORAGE_KEY_MODE]: mode });
      const backend = await resolveActiveBackend(mode);

      // If active tab ID is provided, immediately re-analyze it
      if (request.tabId) {
        chrome.tabs.get(request.tabId, (tab) => {
          if (chrome.runtime.lastError || !tab) return;
          if (tab && tab.url) analyzeTab(tab.id, tab);
        });
      }
      sendResponse({ success: true, backend });
    })();
    return true; // Keep message channel open for async response
  }

  if (request.action === 'getBackendStatus') {
    (async () => {
      const backend = await resolveActiveBackend();
      sendResponse(backend);
    })();
    return true;
  }

  if (request.action === 'reanalyzeTab' && request.tabId) {
    (async () => {
      chrome.tabs.get(request.tabId, async (tab) => {
        if (chrome.runtime.lastError || !tab || !tab.url) {
          sendResponse({ success: false, error: 'No active tab or tab closed' });
          return;
        }
        await analyzeTab(tab.id, tab);
        const result = await chrome.storage.local.get([tab.id.toString()]);
        sendResponse({ success: true, data: result[tab.id.toString()] });
      });
    })();
    return true;
  }

  // ── AUTOMATIC LIVE VIDEO PLAYBACK SCAN ────────────────────────
  if (request.action === 'videoPlaybackStarted' && sender.tab?.id) {
    const tabId = sender.tab.id;
    const windowId = sender.tab.windowId;
    (async () => {
      try {
        const frames = [];
        for (let i = 0; i < 6; i++) {
          const tabCap = await chrome.tabs.captureVisibleTab(windowId, { format: 'jpeg', quality: 50 });
          if (tabCap && tabCap.includes(',')) {
            frames.push(tabCap.split(',')[1]);
          }
          await sleep(120);
        }
        if (frames.length >= 2) {
          chrome.tabs.get(tabId, async (tab) => {
            if (chrome.runtime.lastError || !tab || !tab.url) return;
            console.log(`[PhishGuard] Live streaming video playback re-analyzing tab ${tabId}...`);
            await analyzeTab(tab.id, tab, frames, request.rect);
          });
        }
      } catch (err) {
        console.debug('[PhishGuard] Live video capture note:', err.message);
      }
    })();
    return true;
  }
});

// ── HISTORY HELPERS ───────────────────────────────────────────
async function appendHistory(entry) {
  return new Promise(resolve => {
    chrome.storage.local.get([HISTORY_KEY], (result) => {
      if (chrome.runtime.lastError) return resolve();
      const items = result[HISTORY_KEY] || [];
      if (items.length > 0 && items[0].url === entry.url) {
        items[0] = entry; // update in place
      } else {
        items.unshift(entry);
      }
      const trimmed = items.slice(0, MAX_HISTORY);
      chrome.storage.local.set({ [HISTORY_KEY]: trimmed }, () => {
        void chrome.runtime.lastError;
        resolve();
      });
    });
  });
}

// ── CLEANUP ───────────────────────────────────────────────────
chrome.tabs.onRemoved.addListener((tabId) => {
  try {
    chrome.storage.local.remove(tabId.toString(), () => {
      void chrome.runtime.lastError;
    });
  } catch (_) {}
});

// ── UTILS ─────────────────────────────────────────────────────
function setBadge(tabId, text, color) {
  if (!tabId) return;
  try {
    chrome.action.setBadgeText({ text, tabId: Number(tabId) }, () => {
      void chrome.runtime.lastError;
    });
    chrome.action.setBadgeBackgroundColor({ color, tabId: Number(tabId) }, () => {
      void chrome.runtime.lastError;
    });
  } catch (_) {}
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}
