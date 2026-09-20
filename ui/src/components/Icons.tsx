/** Inline stroke icons in the template's style: 24-box, currentColor, 2px stroke. */

const base = { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round" } as const;

export const IconHome = () => <svg {...base}><path d="M3 11l9-8 9 8" /><path d="M5 10v10h14V10" /></svg>;
export const IconCalendar = () => <svg {...base}><rect x="3" y="5" width="18" height="16" rx="2" /><path d="M3 10h18M8 3v4M16 3v4" /></svg>;
export const IconMap = () => <svg {...base}><path d="M9 4l6 2 6-2v14l-6 2-6-2-6 2V6z" /><path d="M9 4v14M15 6v14" /></svg>;
export const IconList = () => <svg {...base}><path d="M8 6h13M8 12h13M8 18h13" /><circle cx="4" cy="6" r="1" /><circle cx="4" cy="12" r="1" /><circle cx="4" cy="18" r="1" /></svg>;
export const IconUser = () => <svg {...base}><circle cx="12" cy="8" r="4" /><path d="M4 21a8 8 0 0 1 16 0" /></svg>;
export const IconMessage = () => <svg {...base}><path d="M4 5h16v11H8l-4 4z" /></svg>;
export const IconChart = () => <svg {...base}><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" /></svg>;
export const IconSearch = () => <svg {...base}><circle cx="11" cy="11" r="7" /><path d="M21 21l-4-4" /></svg>;
export const IconAlert = () => <svg {...base}><path d="M12 3l10 18H2z" /><path d="M12 10v5M12 18h.01" /></svg>;
export const IconBolt = () => <svg {...base}><path d="M13 2L4 14h7l-1 8 9-12h-7z" /></svg>;
export const IconDrop = () => <svg {...base}><path d="M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11z" /></svg>;
export const IconSun = () => <svg {...base}><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>;
export const IconMail = () => <svg {...base}><rect x="3" y="5" width="18" height="14" rx="2" /><path d="M3 7l9 6 9-6" /></svg>;
export const IconArrow = () => <svg {...base}><path d="M5 12h14M13 6l6 6-6 6" /></svg>;
export const IconBack = () => <svg {...base}><path d="M19 12H5M11 18l-6-6 6-6" /></svg>;
export const IconShield = () => <svg {...base}><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z" /><path d="M9 12l2 2 4-4" /></svg>;
export const IconCheck = () => <svg {...base}><path d="M5 12l5 5L20 7" /></svg>;
export const IconX = () => <svg {...base}><path d="M6 6l12 12M18 6L6 18" /></svg>;
export const IconDownload = () => <svg {...base}><path d="M12 3v12M6 11l6 6 6-6M4 21h16" /></svg>;
export const IconBuilding = () => <svg {...base}><rect x="4" y="3" width="16" height="18" rx="1" /><path d="M9 7h2M13 7h2M9 11h2M13 11h2M9 15h2M13 15h2M10 21v-3h4v3" /></svg>;
export const IconClock = () => <svg {...base}><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></svg>;
export const IconPhone = () => <svg {...base}><path d="M5 4h4l2 5-2.5 1.5a11 11 0 0 0 5 5L15 13l5 2v4a2 2 0 0 1-2 2A16 16 0 0 1 3 6a2 2 0 0 1 2-2z" /></svg>;
export const IconBook = () => <svg {...base}><path d="M4 5a2 2 0 0 1 2-2h12v18H6a2 2 0 0 1-2-2z" /><path d="M8 7h7M8 11h7" /></svg>;
export const IconSparkle = () => <svg {...base}><path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z" /><path d="M18 15l.8 2.2L21 18l-2.2.8L18 21l-.8-2.2L15 18l2.2-.8z" /></svg>;
