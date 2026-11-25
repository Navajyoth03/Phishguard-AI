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
    });
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
})();
