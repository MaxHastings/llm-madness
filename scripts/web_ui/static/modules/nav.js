import { emitEvent } from './events.js';

export function initNav() {
  const navButtons = Array.from(document.querySelectorAll('.nav-btn'));
  const sections = Array.from(document.querySelectorAll('.page-section'));

  function setActive(sectionName) {
    navButtons.forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.section === sectionName);
    });
    sections.forEach((section) => {
      section.classList.toggle('active', section.dataset.section === sectionName);
    });
    emitEvent('section:changed', { section: sectionName });
  }

  navButtons.forEach((btn) => {
    btn.addEventListener('click', () => setActive(btn.dataset.section));
  });
}
