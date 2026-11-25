// PhishGuard content.js - Automatic Active Video Playback Detection & Live Extraction
'use strict';

(function () {
  let playDebounce = 0;

  function onVideoPlaying(videoEl) {
    if (!videoEl || videoEl.paused || videoEl.ended) return;

    // Give playback ~1.5s to get past initial black frames/buffering spinners
    setTimeout(() => {
      if (videoEl.paused || videoEl.ended) return;
      if (Date.now() - playDebounce < 8000) return; // Debounce scans to once per 8 seconds
      playDebounce = Date.now();

      const rect = videoEl.getBoundingClientRect();
      // Ensure player is visible and has significant size (not a 1x1 tracking pixel)
      if (rect.width < 120 || rect.height < 90) return;

      console.log('[PhishGuard] Live video playback detected! Initiating forensic stream scan...');
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
    }, 1500);
  }

  // Use capture phase on document to catch 'play' and 'playing' events
  // works across all dynamically injected videos, SPAs (YouTube, Vimeo, etc.)
  document.addEventListener('play', (e) => {
    if (e.target && e.target.tagName === 'VIDEO') {
      onVideoPlaying(e.target);
    }
  }, true);

  document.addEventListener('playing', (e) => {
    if (e.target && e.target.tagName === 'VIDEO') {
      onVideoPlaying(e.target);
    }
  }, true);
})();
