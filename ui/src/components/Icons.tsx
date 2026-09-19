/** Inline stroke icons in the template's style: 24-box, currentColor, 2px stroke. */

const base = { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round" } as const;

export const IconHome = () => (
  <svg {...base}><path d="M3 11l9-8 9 8" /><path d="M5 10v10h14V10" /></svg>
);
export const IconMap = () => (
  <svg {...base}><path d="M9 4l6 2 6-2v14l-6 2-6-2-6 2V6z" /><path d="M9 4v14M15 6v14" /></svg>
);
export const IconList = () => (
  <svg {...base}><path d="M8 6h13M8 12h13M8 18h13" /><circle cx="4" cy="6" r="1" /><circle cx="4" cy="12" r="1" /><circle cx="4" cy="18" r="1" /></svg>
);
export const IconUser = () => (
  <svg {...base}><circle cx="12" cy="8" r="4" /><path d="M4 21a8 8 0 0 1 16 0" /></svg>
);
export const IconMessage = () => (
  <svg {...base}><path d="M4 5h16v11H8l-4 4z" /></svg>
);
export const IconChart = () => (
  <svg {...base}><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" /></svg>
);
export const IconSearch = () => (
  <svg {...base}><circle cx="11" cy="11" r="7" /><path d="M21 21l-4-4" /></svg>
);
export const IconAlert = () => (
  <svg {...base}><path d="M12 3l10 18H2z" /><path d="M12 10v5M12 18h.01" /></svg>
);
export const IconBolt = () => (
  <svg {...base}><path d="M13 2L4 14h7l-1 8 9-12h-7z" /></svg>
);
export const IconDrop = () => (
  <svg {...base}><path d="M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11z" /></svg>
);
export const IconSun = () => (
  <svg {...base}><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>
);
export const IconMail = () => (
  <svg {...base}><rect x="3" y="5" width="18" height="14" rx="2" /><path d="M3 7l9 6 9-6" /></svg>
);
export const IconArrow = () => (
  <svg {...base}><path d="M5 12h14M13 6l6 6-6 6" /></svg>
);
