export function isSectionActive(sectionName) {
  const active = document.querySelector('.page-section.active');
  return !!active && active.dataset.section === sectionName;
}

const refreshCallbacks = new Set();
let listenersBound = false;

function ensureListeners() {
  if (listenersBound) return;
  listenersBound = true;
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState !== 'visible') return;
    refreshCallbacks.forEach((cb) => cb());
  });
  document.addEventListener('section:changed', () => {
    refreshCallbacks.forEach((cb) => cb());
  });
}

export function scheduleAutoRefresh({ intervalMs, isEnabled, task }) {
  if (!intervalMs || typeof task !== 'function') {
    return { stop: () => {} };
  }

  let inFlight = false;

  const run = async () => {
    if (inFlight) return;
    if (document.visibilityState === 'hidden') return;
    if (typeof isEnabled === 'function' && !isEnabled()) return;
    inFlight = true;
    try {
      await task();
    } catch (err) {
      // Ignore auto-refresh failures to avoid spamming the UI.
    } finally {
      inFlight = false;
    }
  };

  const timer = window.setInterval(run, intervalMs);
  ensureListeners();
  refreshCallbacks.add(run);

  return {
    stop() {
      window.clearInterval(timer);
      refreshCallbacks.delete(run);
    },
  };
}
