export function emitEvent(name, detail) {
  document.dispatchEvent(new CustomEvent(name, { detail }));
}

export function onEvent(name, handler) {
  document.addEventListener(name, handler);
  return () => document.removeEventListener(name, handler);
}
