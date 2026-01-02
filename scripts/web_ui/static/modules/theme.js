const STORAGE_KEY = 'llm-madness-theme';

function getStoredTheme() {
  const value = localStorage.getItem(STORAGE_KEY);
  return value === 'dark' || value === 'light' ? value : null;
}

function getPreferredTheme() {
  if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    return 'dark';
  }
  return 'light';
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
}

function updateToggle(toggle, theme) {
  if (!toggle) return;
  toggle.textContent = theme === 'dark' ? 'Light mode' : 'Dark mode';
  toggle.setAttribute('aria-pressed', theme === 'dark' ? 'true' : 'false');
}

export function initTheme() {
  const toggle = document.getElementById('themeToggle');
  const stored = getStoredTheme();
  const initialTheme = stored || getPreferredTheme();
  applyTheme(initialTheme);
  updateToggle(toggle, initialTheme);

  if (!stored && window.matchMedia) {
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = (event) => {
      const nextTheme = event.matches ? 'dark' : 'light';
      applyTheme(nextTheme);
      updateToggle(toggle, nextTheme);
    };
    if (media.addEventListener) {
      media.addEventListener('change', handler);
    } else if (media.addListener) {
      media.addListener(handler);
    }
  }

  if (toggle) {
    toggle.addEventListener('click', () => {
      const current = document.documentElement.dataset.theme;
      const next = current === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      localStorage.setItem(STORAGE_KEY, next);
      updateToggle(toggle, next);
    });
  }
}
