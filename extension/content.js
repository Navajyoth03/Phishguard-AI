// PhishGuard content.js - Continuous Live Video Monitoring & Periodic Extraction
'use strict';

(function () {
  let activeVideoEl = null;
  let monitorInterval = null;

  function sendVideoCrop(videoEl) {
    if (!videoEl || videoEl.paused || videoEl.ended || videoEl.readyState < 2) return;
    const rect = videoEl.getBoundingClientRect();
    if (rect.width < 120 || rect.height < 90) return;

    chrome.runtime.sendMessage({
      action: 'videoPlaybackStarted',
      rect: {
        x: Math.max(0, rect.left),
        y: Math.max(0, rect.top),
        width: rect.width,
        height: rect.height,
        viewport_width: window.innerWidth,
        viewport_height: window.innerHeight
      }
    }).catch(() => {});
  }

  function startMonitoring(videoEl) {
    activeVideoEl = videoEl;
    if (monitorInterval) clearInterval(monitorInterval);

    // Initial check ~1.2s after play starts (past black screen/spinner)
    setTimeout(() => {
      sendVideoCrop(videoEl);
    }, 1200);

    // Continuous checking: checks throughout playback every 5 seconds
    monitorInterval = setInterval(() => {
      if (activeVideoEl && !activeVideoEl.paused && !activeVideoEl.ended) {
        sendVideoCrop(activeVideoEl);
      } else if (!activeVideoEl || activeVideoEl.paused || activeVideoEl.ended) {
        clearInterval(monitorInterval);
        monitorInterval = null;
      }
    }, 5000);
  }

  // Use capture phase on document to catch 'play' and 'playing' events
  document.addEventListener('play', (e) => {
    if (e.target && e.target.tagName === 'VIDEO') {
      startMonitoring(e.target);
    }
  }, true);

  document.addEventListener('playing', (e) => {
    if (e.target && e.target.tagName === 'VIDEO') {
      startMonitoring(e.target);
    }
  }, true);

  document.addEventListener('pause', (e) => {
    if (e.target === activeVideoEl && monitorInterval) {
      clearInterval(monitorInterval);
      monitorInterval = null;
    }
  }, true);

  // ── IN-PAGE FLOATING FORENSICS HUD ─────────────────────────────
  function getOrCreateHUD() {
    let hud = document.getElementById('phishguard-forensics-hud');
    if (!hud) {
      hud = document.createElement('div');
      hud.id = 'phishguard-forensics-hud';
      hud.style.cssText = `
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 2147483647;
        width: 360px;
        max-width: calc(100vw - 48px);
        background: #0B1120;
        color: #F8FAFC;
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 14px;
        box-shadow: 0 20px 35px -10px rgba(0, 0, 0, 0.8), 0 0 20px rgba(56, 189, 248, 0.15);
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
        font-size: 13px;
        line-height: 1.5;
        overflow: hidden;
        animation: pgSlideUp 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        transition: opacity 0.25s ease;
      `;

      // Add CSS keyframe animation
      if (!document.getElementById('phishguard-hud-styles')) {
        const style = document.createElement('style');
        style.id = 'phishguard-hud-styles';
        style.textContent = `
          @keyframes pgSlideUp {
            from { opacity: 0; transform: translateY(20px) scale(0.96); }
            to { opacity: 1; transform: translateY(0) scale(1); }
          }
          @keyframes pgSpin {
            to { transform: rotate(360deg); }
          }
        `;
        document.head.appendChild(style);
      }

      document.body.appendChild(hud);
    }
    hud.style.display = 'block';
    hud.style.opacity = '1';
    return hud;
  }

  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.action === 'showImageForensicsToast') {
      const hud = getOrCreateHUD();

      if (msg.status === 'loading') {
        hud.innerHTML = `
          <div style="padding: 14px 16px; display: flex; align-items: center; gap: 12px;">
            <div style="width: 22px; height: 22px; border: 3px solid rgba(56, 189, 248, 0.2); border-top-color: #38BDF8; border-radius: 50%; animation: pgSpin 0.8s linear infinite; flex-shrink: 0;"></div>
            <div style="flex: 1;">
              <div style="font-weight: 700; color: #38BDF8; font-size: 13px;">🛡️ Inspecting Image Forensics</div>
              <div style="font-size: 11px; color: #94A3B8; margin-top: 2px;">Computing ELA compression & 2D-FFT noise...</div>
            </div>
            <button id="pg-hud-close" style="background:none; border:none; color:#64748B; font-size:16px; cursor:pointer; padding:4px;">✕</button>
          </div>
        `;
      } else if (msg.status === 'done' && msg.data) {
        const data = msg.data;
        const pct = Math.round((data.risk_score || 0) * 100);
        const isTampered = data.is_tampered;
        const color = isTampered ? '#EF4444' : '#10B981';
        const bgBadge = isTampered ? 'rgba(239, 68, 68, 0.15)' : 'rgba(16, 185, 129, 0.15)';
        const icon = isTampered ? '⚠️' : '✅';
        const verdict = data.verdict || (isTampered ? 'TAMPERED / AI' : 'AUTHENTIC IMAGE');

        const indicators = data.indicators || [];
        const topIndicator = indicators.length > 0 ? indicators[0] : (isTampered ? 'Manipulated image elements detected.' : 'Uniform sensor noise & authentic compression verified.');

        hud.innerHTML = `
          <div style="padding: 12px 14px; background: rgba(255,255,255,0.03); border-bottom: 1px solid rgba(255,255,255,0.08); display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 6px; font-weight: 700; font-size: 12px;">
              <span>🛡️ PhishGuard Image Inspector</span>
            </div>
            <button id="pg-hud-close" style="background:none; border:none; color:#94A3B8; font-size:14px; cursor:pointer; padding:2px;">✕</button>
          </div>
          <div style="padding: 14px 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
              <span style="background: ${bgBadge}; color: ${color}; border: 1px solid ${color}40; padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 800; letter-spacing: 0.5px;">
                ${icon} ${verdict}
              </span>
              <span style="font-size: 14px; font-weight: 800; color: ${color};">${pct}% RISK</span>
            </div>
            <div style="font-size: 11px; color: #CBD5E1; line-height: 1.4; background: rgba(0,0,0,0.25); padding: 8px 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 10px;">
              ${topIndicator}
            </div>
            <div style="font-size: 10px; color: #94A3B8; text-align: right;">
              Click PhishGuard icon in toolbar for full ELA heatmap
            </div>
          </div>
        `;
      } else if (msg.status === 'error') {
        hud.innerHTML = `
          <div style="padding: 14px 16px; display: flex; align-items: center; gap: 12px;">
            <div style="font-size: 20px;">⚠️</div>
            <div style="flex: 1;">
              <div style="font-weight: 700; color: #F59E0B; font-size: 13px;">Inspection Notice</div>
              <div style="font-size: 11px; color: #CBD5E1; margin-top: 2px;">${msg.message || 'Could not analyze image.'}</div>
            </div>
            <button id="pg-hud-close" style="background:none; border:none; color:#64748B; font-size:16px; cursor:pointer; padding:4px;">✕</button>
          </div>
        `;
      }

      const closeBtn = hud.querySelector('#pg-hud-close');
      if (closeBtn) {
        closeBtn.onclick = () => {
          hud.style.opacity = '0';
          setTimeout(() => { hud.style.display = 'none'; }, 250);
        };
      }

      // Auto dismiss after 12 seconds
      if (msg.status !== 'loading') {
        setTimeout(() => {
          if (hud && hud.style.display !== 'none') {
            hud.style.opacity = '0';
            setTimeout(() => { hud.style.display = 'none'; }, 250);
          }
        }, 12000);
      }
    }
  });
})();
